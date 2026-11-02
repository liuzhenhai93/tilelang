"""MLU590 lowering for one per-IPU BANG GEMM tile.

The region passed to ``T.gemm`` is one IPU task's tile (M rows at most 64);
the whole fragment lives in that task's NRAM. The kernel's ``T.Kernel``
index selects the tile directly.
"""

from __future__ import annotations

from tilelang import language as T
from tilelang import tvm
from tilelang.layout import Fragment
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_fragment
from tvm import tirx

GEMM_INST_BANG = "mlu.bang_gemm"


def _access_ptr(buffer, region, rw_mask: int):
    offset = tirx.const(0, "int32")
    for axis, item in enumerate(region.region):
        stride = tirx.const(1, "int32")
        for shape in buffer.shape[axis + 1 :]:
            stride = stride * shape
        offset = offset + tirx.Cast("int32", item.min) * stride
    extent = tirx.const(1, "int32")
    for item in region.region:
        extent = extent * item.extent
    return tirx.op.tvm_access_ptr(
        tirx.op.type_annotation(buffer.dtype),
        buffer.data,
        offset,
        extent,
        tirx.IntImm("int32", rw_mask),
    )


def _const_dim(expr) -> int:
    if isinstance(expr, int):
        return expr
    value = getattr(expr, "value", None)
    if isinstance(value, int):
        return value
    raise TypeError(f"MLU GEMM layout requires a constant buffer dimension, got {expr}")


def _ptr_access(buf, count: int):
    """Flat tvm_access_ptr over the first ``count`` elements of ``buf``."""
    return tirx.op.tvm_access_ptr(
        tirx.op.type_annotation(buf.dtype), buf.data,
        tirx.const(0, "int32"), tirx.IntImm("int32", count),
        tirx.IntImm("int32", 1),
    )


def _nram_fragment(buffer) -> Fragment:
    """Row-major NRAM, wholly owned by one IPU task."""
    shape = [_const_dim(dim) for dim in buffer.shape]

    def forward(*indices):
        offset = 0
        for coord, extent in zip(indices, shape, strict=True):
            offset = offset * extent + coord
        return 0, offset

    return Fragment(shape, forward_fn=forward)


class GemmBang(GemmBase):
    """Lower one float32 per-IPU tile. M is the tile rows, at most 64."""

    def infer_layout(self, target, thread_nums: int):
        # Tiles are per-IPU: every fragment lives wholly in one task.
        return {buf: _nram_fragment(buf) for buf in (self.A, self.B, self.C) if is_fragment(buf)}

    def lower(self, layout_map, target, thread_bounds, thread_index, mbar_phase_expr=None):
        aligned = self.M % 16 == 0 and self.N % 64 == 0 and self.K % 64 == 0
        if self.M <= 0 or self.N <= 0 or self.K <= 0 or self.M > 256 or not aligned:
            raise ValueError(
                "MLU BANG GEMM tiles are per-IPU: M must be a positive multiple "
                "of 16 at most 64, and N and K positive multiples of 64; "
                f"got M={self.M}, N={self.N}, K={self.K}"
            )
        operands = (self.A, self.B, self.C)
        nram = all(is_fragment(buf) for buf in operands)
        if not nram and any(buf.scope() != "global" for buf in operands):
            raise ValueError("MLU BANG GEMM operands must all be global or all be fragments")
        if nram and self.trans_A:
            raise ValueError("MLU fragment GEMM does not support transpose_A")
        trans_b = self.trans_B if isinstance(self.trans_B, tirx.PrimExpr) else tirx.const(bool(self.trans_B), "bool")
        # Every helper scratch is allocated by the CALLER (TIR buffers land
        # in the uninit sections): helper-side statics would be
        # nram/wram_zero_init sections that the runtime clears via GDRAM
        # once per task (512 KB of task-switch write traffic).
        dtype_str = str(self.A.dtype)
        nk = self.N * self.K
        # The muladd form accumulates in place on the TFU, so a non-clear
        # gemm needs no c_part scratch at any M*N; the old M*N <= 8192 gate
        # only served the (now unreachable) full 13-arg scratch path.
        lean = nram and bool(self.trans_B) and nk <= 65536

        @T.prim_func
        def _gemm_bang() -> None:
            if nram and lean:
                wram = T.alloc_shared((nk,), dtype_str, scope="local.wram")
                wram_ptr = tirx.op.tvm_access_ptr(
                    tirx.op.type_annotation(wram.dtype), wram.data,
                    tirx.const(0, "int32"), tirx.IntImm("int32", nk),
                    tirx.IntImm("int32", 1),
                )
                call = tirx.call_intrin(
                    "void",
                    tirx.op.Op.get("tl.mlu_nram_gemm"),
                    _access_ptr(self.CRegion.buffer, self.CRegion, 2),
                    _access_ptr(self.ARegion.buffer, self.ARegion, 1),
                    _access_ptr(self.BRegion.buffer, self.BRegion, 1),
                    wram_ptr,
                    tirx.IntImm("int32", self.M),
                    tirx.IntImm("int32", self.N),
                    tirx.IntImm("int32", self.K),
                    self.clear_accum,
                    trans_b,
                    tirx.IntImm("int32", 0),
                )
            elif nram:
                wram = T.alloc_shared((nk,), dtype_str, scope="local.wram")
                a_panel = T.alloc_shared((nk,), dtype_str, scope="local.fragment")
                b_nk = T.alloc_shared((nk,), dtype_str, scope="local.fragment")
                c_part = T.alloc_shared((self.M * self.N,), "float32", scope="local.fragment")
                call = tirx.call_intrin(
                    "void",
                    tirx.op.Op.get("tl.mlu_nram_gemm"),
                    _access_ptr(self.CRegion.buffer, self.CRegion, 2),
                    _access_ptr(self.ARegion.buffer, self.ARegion, 1),
                    _access_ptr(self.BRegion.buffer, self.BRegion, 1),
                    _ptr_access(wram, nk), _ptr_access(a_panel, nk),
                    _ptr_access(b_nk, nk),
                    _ptr_access(c_part, self.M * self.N),
                    tirx.IntImm("int32", self.M),
                    tirx.IntImm("int32", self.N),
                    tirx.IntImm("int32", self.K),
                    self.clear_accum,
                    trans_b,
                    tirx.IntImm("int32", 0),
                )
            else:
                call = tirx.call_intrin(
                    "void",
                    tirx.op.Op.get("tl.mlu_bang_gemm"),
                    _access_ptr(self.CRegion.buffer, self.CRegion, 2),
                    _access_ptr(self.ARegion.buffer, self.ARegion, 1),
                    _access_ptr(self.BRegion.buffer, self.BRegion, 1),
                    tirx.IntImm("int32", self.M),
                    tirx.IntImm("int32", self.N),
                    tirx.IntImm("int32", self.K),
                    self.clear_accum,
                )
            T.evaluate(call)

        return _Simplify(_gemm_bang, inline_let=True)


register_gemm_impl(
    "mlu.Gemm",
    GEMM_INST_BANG,
    lambda target: target.kind.name == "mlu",
    GemmBang,
)
