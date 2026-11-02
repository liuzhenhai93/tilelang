"""MPU-projection pass for cluster-SRAM kernels (SPMD script view).

The script is written as SPMD: one body for every context of the cluster
(4 IPU tasks + the MPU helper). The script owns everything -- the SRAM
buffer (``alloc_shared``), the GDRAM->SRAM fills, the ``sync_cluster``
barriers, the SRAM->NRAM consumption copies and the business compute.

This pass ONLY hides the MPU, anchored on two statement classes:

  - a copy whose DESTINATION buffer is SRAM (scope shared/shared.dyn)
    is the producer's fill -> MPU-only (removed from the IPU path);
  - ``__sync_*`` calls are kept on both paths;
  - everything else (the SRAM->NRAM consumption, gemms, softmax,
    epilogue) is IPU-only (removed from the MPU path, which owns no
    fragments).

The loop/if structure is preserved on both sides so the barrier sequence
matches by construction. The MPU's projection is additionally
WAVE-EXTENDED: its loop count is multiplied by taskDim/32 (the number of
waves) with the loop variable remapped modulo the original extent, so the
MPU stays alive across all waves instead of exiting after the first and
poisoning every later __sync_cluster with a ~340 us timeout wait.
"""

from __future__ import annotations

from tvm import tirx

_SHARED_SCOPES = ("shared", "shared.dyn")
# Fallback only; the pass reads cluster_dim x core_dim from the target
# attrs registered in src/mlu/target.cc (8 clusters x 4 IPUs = 32).
_IPUS_PER_WAVE_DEFAULT = 32


def _call(name, *args):
    return tirx.call_extern("void", name, *args)


def _memcpy_dst_var(stmt):
    """Return the destination buffer var of a memcpy call, else None."""
    if not (isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call)):
        return None
    call = stmt.value
    name = getattr(call.op, "name", "")
    if not (name.startswith("__memcpy") and len(call.args) >= 1):
        return None
    dst = call.args[0]
    if isinstance(dst, tirx.Call) and len(dst.args) >= 2:
        return dst.args[1]  # tvm_access_ptr(anno, var, ...)
    return None


def _is_sync(stmt):
    if not (isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call)):
        return False
    return getattr(stmt.value.op, "name", "").startswith("__sync")


class _Projection:
    """Classify statements and build the MPU/IPU projections."""

    def __init__(self):
        self.scopes = {}

    def scope_of(self, var):
        return self.scopes.get(var, None)

    def run(self, stmt):
        """Project ``stmt``; returns (mpu_stmt_or_None, ipu_stmt_or_None)."""
        if isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "storage_scope":
            self.scopes[stmt.node] = str(stmt.value)
            mpu, ipu = self.run(stmt.body)
            if mpu is None and ipu is None:
                return None, None
            both = mpu is not None and ipu is not None
            if both:
                return stmt, stmt
            if mpu is not None:
                return stmt, None
            return None, stmt

        if isinstance(stmt, tirx.AllocBuffer):
            self.scopes[stmt.buffer.data] = str(stmt.buffer.scope())
            return stmt, stmt  # the declaration: kept on both paths

        if isinstance(stmt, tirx.Allocate):
            self.scopes.setdefault(stmt.buffer_var, None)
            mpu, ipu = self.run(stmt.body)
            mpu = tirx.Allocate(stmt.buffer_var, stmt.dtype, stmt.extents,
                                stmt.condition, mpu) if mpu is not None else None
            ipu = tirx.Allocate(stmt.buffer_var, stmt.dtype, stmt.extents,
                                stmt.condition, ipu) if ipu is not None else None
            return mpu, ipu

        if isinstance(stmt, tirx.SeqStmt):
            mpu, ipu = [], []
            for s in stmt.seq:
                sm, si = self.run(s)
                if sm is not None:
                    mpu.append(sm)
                if si is not None:
                    ipu.append(si)
            return (tirx.SeqStmt(mpu) if mpu else None,
                    tirx.SeqStmt(ipu) if ipu else None)

        if isinstance(stmt, tirx.IfThenElse):
            mpu_t, ipu_t = self.run(stmt.then_case)
            mpu_e, ipu_e = (self.run(stmt.else_case) if stmt.else_case is not None
                            else (None, None))
            mpu = tirx.IfThenElse(stmt.condition, mpu_t, mpu_e) if (mpu_t is not None or mpu_e is not None) else None
            ipu = tirx.IfThenElse(stmt.condition, ipu_t, ipu_e) if (ipu_t is not None or ipu_e is not None) else None
            return mpu, ipu

        if isinstance(stmt, tirx.For):
            mpu_b, ipu_b = self.run(stmt.body)
            mpu = tirx.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                           mpu_b, stmt.thread_binding, stmt.annotations) if mpu_b is not None else None
            ipu = tirx.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                           ipu_b, stmt.thread_binding, stmt.annotations) if ipu_b is not None else None
            return mpu, ipu

        if isinstance(stmt, tirx.Evaluate):
            if _is_sync(stmt):
                return stmt, stmt  # syncs: both paths
            dst_var = _memcpy_dst_var(stmt)
            if dst_var is not None and self.scope_of(dst_var) in _SHARED_SCOPES:
                return stmt, None  # SRAM fill: MPU-only
            return None, stmt  # the business: IPU-only

        return None, stmt


def has_shared_fill(body: tirx.Stmt) -> bool:
    """True if the body contains a memcpy whose destination is SRAM."""
    found = [False]

    def walk(stmt):
        if found[0]:
            return
        if isinstance(stmt, tirx.AllocBuffer):
            if str(stmt.buffer.scope()) in _SHARED_SCOPES:
                found[0] = True
                return
        if isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "storage_scope":
            if str(stmt.value) in _SHARED_SCOPES:
                found[0] = True
                return
        if isinstance(stmt, tirx.SeqStmt):
            for s in stmt.seq:
                walk(s)
        elif isinstance(stmt, tirx.IfThenElse):
            walk(stmt.then_case)
            if stmt.else_case is not None:
                walk(stmt.else_case)
        elif isinstance(stmt, tirx.For):
            walk(stmt.body)

    walk(body)
    return found[0]


def _wave_extend(stmt, waves):
    """Multiply the MPU projection's loop counts by the wave count.

    For(v, lo, ext) -> For(rv, 0, ext*waves) with the body's uses of v
    remapped to FloorMod(rv, ext), so the panel/slot arithmetic cycles
    across waves exactly as the IPU tasks' rounds do.
    """
    if isinstance(stmt, tirx.For):
        rv = tirx.Var("r_wave", "int32")
        new_ext = tirx.Mul(stmt.extent, waves)
        new_body = tirx.stmt_functor.substitute(
            stmt.body, {stmt.loop_var: tirx.FloorMod(rv, stmt.extent)})
        return tirx.For(rv, stmt.min, new_ext, stmt.kind, new_body, None, {})
    if isinstance(stmt, tirx.SeqStmt):
        return tirx.SeqStmt([_wave_extend(s, waves) for s in stmt.seq])
    if isinstance(stmt, tirx.IfThenElse):
        return tirx.IfThenElse(stmt.condition,
                               _wave_extend(stmt.then_case, waves),
                               _wave_extend(stmt.else_case, waves)
                               if stmt.else_case is not None else None)
    return stmt


def lower(body: tirx.Stmt, ipus_per_wave: int = _IPUS_PER_WAVE_DEFAULT) -> tirx.Stmt:
    """Project an SPMD kernel body into the MPU/IPU paths."""
    proj = _Projection()
    mpu, ipu = proj.run(body)
    if mpu is None:
        return tirx.IfThenElse(tirx.Evaluate(0), tirx.Evaluate(0), ipu)
    if ipu is None:
        return tirx.IfThenElse(tirx.Evaluate(1), mpu, tirx.Evaluate(0))
    # wave-extend the MPU projection: its loops run once per wave
    waves = tirx.FloorDiv(tirx.call_extern("int32", "taskDim"),
                          tirx.const(ipus_per_wave, "int32"))
    mpu = _wave_extend(mpu, waves)
    is_mpu = tirx.NE(tirx.call_extern("int32", "__is_mpu"), tirx.const(0, "int32"))
    return tirx.IfThenElse(is_mpu, mpu, ipu)