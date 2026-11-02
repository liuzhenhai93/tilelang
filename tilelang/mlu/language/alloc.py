"""MLU-specific buffer allocation.

``alloc_wram`` allocates a WRAM (weight-RAM) buffer. WRAM is the
conv/matmul filter memory: per-IPU private (like NRAM, hence the
fragment allocation path) but a distinct hardware tier with its own bank
structure; the MLU codegen prints ``__wram__`` for the ``local.wram``
scope. Other backends do not have this tier, so the op lives in the MLU
dialect instead of the shared language surface.

Hardware note: the conv engine's filter operand addresses a window
starting at the WRAM base -- keep the *base offset* of a filter below
32 KB (payloads may cross that boundary); see
``examples/mlu/probe_wramoff.py``.
"""

from __future__ import annotations

from tvm.script import tirx as T

from tilelang.language.allocate import _with_span


def alloc_wram(shape, dtype: str):
    """Allocate a WRAM (weight-RAM) buffer, MLU only."""
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope="local.wram"))