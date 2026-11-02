"""T.conv.* — the MLU conv-engine matmul dialect.

The TFU conv engine executes 1x1 convolutions as matrix multiplies at a
rate the ``__bang_matmul`` path cannot approach at attention shapes
(~31 TF/core vs ~1.2 TF/core measured; see examples/mlu/
bench_conv_vs_gemm.py). The ops emit the raw ``conv.nram`` instruction
through the generic ``__tl_mlu_conv_gen`` preamble helper that
``CodeGenBANG`` prints unconditionally.

Like :func:`tilelang.mlu.language.sync`, each op returns the tirx
``call_extern`` PrimExpr, so a bare statement in a traced kernel is
auto-emitted by the eager builder -- write ``conv.matmul(...)`` as a
statement, no ``T.evaluate`` wrapper needed.

Hardware contract (all verified on-device, see examples/mlu/probe_conv*.py):

- encoding: ``conv.nram.rn.f32.bf16.bf16 [dst],[in],[filter], ci, rows,
  1,1,1,1,1, co, .mstride(1), .kstride(ci*2 bytes), .indilation(1,1),
  <fusion modifiers>``. ``rows`` is the per-call row count (one call
  covers all rows; the output is plain row-major f32). Verified: the
  ``.kstride`` modifier accepts a register operand, so ``ci`` is a free
  runtime value (probe_convgen.py).
- the filter is a ``(co, ci)`` bf16 matrix packed into WRAM by
  :meth:`reshape_filter` -- the same layout ``ld.3d`` /
  ``trans.tiling.wram.sram`` produce for Triton-MLU.
- the filter's WRAM base must be below 32 KB (payloads may cross that
  boundary); keep at most one 32 KB filter alive beside the 16 KB ones.
- the accumulator/output is f32; a bf16 dst is rejected by cnas (0117).
- ``.cvt.src.rn.f32()`` exists for f32 INPUTS (it poisons bf16 inputs);
  :meth:`matmul` selects it automatically when ``a`` is f32.
"""

from __future__ import annotations

from tvm import tirx


def _call(name: str, *args):
    return tirx.call_extern("void", name, *args)


def _addr(x):
    """Address of a fragment handle, BufferLoad, or BufferRegion slice
    (see bang._arg)."""
    from tilelang.language import address_of

    if isinstance(x, tirx.expr.BufferLoad):
        return address_of(x)
    if isinstance(x, tirx.BufferRegion):
        return address_of(
            tirx.BufferLoad(x.buffer, [r.min for r in x.region])
        )
    if isinstance(x, tirx.PrimExpr):
        return x
    return address_of(x[0, 0])


class ConvNamespace:
    """``T.conv.<op>(...)`` -- conv-engine matmul ops for the MLU backend."""

    def reshape_filter(self, dst, src, co: int, ci: int):
        """Pack a ``(co, ci)`` NRAM matrix into the conv-filter WRAM layout.

        ``dst`` must be a WRAM buffer (``T.alloc_wram``) of at least
        ``co * ci`` bf16 elements. Runs on the MOVE queue; follow with
        ``sync("move")`` before the consuming :meth:`matmul`.
        """
        return _call(
            "__bang_reshape_filter",
            _addr(dst),
            _addr(src),
            co,
            1,
            1,
            ci,
            tirx.call_extern("int32", "NRAM2WRAM"),
        )

    def matmul(
        self,
        dst,
        a,
        w,
        ci: int,
        rows: int,
        co: int,
        scale: float = 1.0,
        accumulate: bool = False,
        bias: float = 0.0,
    ):
        """1x1-conv matmul on the TFU: ``dst[rows, co] = a[rows, ci] @ w[co, ci]^T``.

        - ``dst``: f32 fragment; with ``accumulate`` the instruction adds
          into the existing contents (``.add.partial``), otherwise it
          clears them.
        - ``a``: bf16 or f32 fragment (f32 selects the ``.cvt.src``
          in-instruction convert, skipping a separate cast pass).
        - ``w``: WRAM filter from :meth:`reshape_filter`.
        - ``scale``: folds a multiply of the product into the instruction
          (``.mul.scalar``; 1.0 emits the no-scale form for accumulate).
        - ``bias``: folds a scalar add into the instruction
          (``.add.scalar.rn``, verified exact). Not combinable with
          ``accumulate``.
        """
        from tilelang.language import address_of

        is_f32 = str(getattr(a, "dtype", "bfloat16")) == "float32"
        addr = address_of(a[0, 0]) if not isinstance(a, tirx.PrimExpr) else a
        if bias != 0.0:
            assert not accumulate, "bias fusion is a clear-dst form"
            return _call(
                "__tl_mlu_conv_gen_b",
                _addr(dst),
                addr,
                _addr(w),
                scale,
                bias,
                ci,
                rows,
                co,
                1 if is_f32 else 0,
            )
        return _call(
            "__tl_mlu_conv_gen",
            _addr(dst),
            addr,
            _addr(w),
            scale,
            ci,
            rows,
            co,
            1 if accumulate else 0,
            1 if is_f32 else 0,
        )

    # Back-compat aliases over the pre-generalization API.
    def qk(self, dst, q, w, scale: float, rows: int, co: int):
        """``dst = (q[rows, 64] @ w[co, 64]^T) * scale`` (ci=64, clear)."""
        return self.matmul(dst, q, w, 64, rows, co, scale=scale)

    def pv(self, dst, p, w, rows: int, co: int):
        """``dst += p[rows, 128] @ w[co, 128]^T`` (ci=128, accumulate)."""
        return self.matmul(dst, p, w, 128, rows, co, accumulate=True)

    def pv_f32(self, dst, p, w, rows: int, co: int):
        """Like :meth:`pv` but ``p`` is f32 (``.cvt.src`` in-instruction)."""
        return self.matmul(dst, p, w, 128, rows, co, accumulate=True)


conv = ConvNamespace()