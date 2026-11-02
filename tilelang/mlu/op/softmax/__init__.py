"""Fused online-softmax panel ops for MLU590 (one NFU instruction each).

These are explicit tile ops for the chains that the elementwise lowering can
only fuse opportunistically. Calling them guarantees the fused BANG intrinsic
is emitted regardless of how the surrounding statement is written.

The ops run wholly on one IPU task: the panel rows are per-IPU fragment rows.
"""

from __future__ import annotations

from tilelang import language as T
from tvm import tirx

_CYCLE_SUB_MUL_EXP = "__bang_cycle_sub_mul_exp"


def _const_dim(expr) -> int:
    if isinstance(expr, int):
        return int(expr)
    value = getattr(expr, "value", None)
    if isinstance(value, int):
        return int(value)
    raise TypeError(f"panel_exp2_shift requires constant fragment dims, got {expr}")


def _buffer_of(x):
    """Trace-time fragment handles may wrap the tirx Buffer; unwrap."""
    buf = getattr(x, "buffer", None)
    return buf if buf is not None else x


def _access_ptr(buf, rw_mask: int):
    extent = 1
    for dim in buf.shape:
        extent *= _const_dim(dim)
    return tirx.op.tvm_access_ptr(
        tirx.op.type_annotation(buf.dtype),
        buf.data,
        tirx.const(0, "int32"),
        tirx.IntImm("int32", extent),
        tirx.IntImm("int32", rw_mask),
    )


def panel_exp2_shift(panel, row_vec, scale):
    """``panel[i, j] = exp2((panel[i, j] - row_vec[j]) * scale)`` in one NFU op.

    Matches BANG 592's ``__bang_cycle_sub_mul_exp``: ``row_vec`` cycles over
    the panel's row width, ``scale`` is the exp2-base multiplier (the FA
    ``scale`` already includes log2(e)). Both fragments must be float32 and
    row-major with ``panel`` 2-D; ``row_vec`` 1-D of length = panel cols.
    Emits in place on ``panel`` and returns it.
    """
    rows = _const_dim(panel.shape[0])
    cols = _const_dim(panel.shape[1])
    vec_len = 1
    for dim in row_vec.shape:
        vec_len *= _const_dim(dim)
    if vec_len != cols:
        raise ValueError(
            f"panel_exp2_shift: row_vec length {vec_len} != panel cols {cols}"
        )
    pbuf = _buffer_of(panel)
    vbuf = _buffer_of(row_vec)
    return tirx.call_extern(
        "void",
        _CYCLE_SUB_MUL_EXP,
        _access_ptr(pbuf, 2),
        _access_ptr(pbuf, 1),
        _access_ptr(vbuf, 1),
        scale,
        rows * cols,
        cols,
    )