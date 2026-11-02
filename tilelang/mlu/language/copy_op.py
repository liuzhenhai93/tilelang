"""MLU dialect of the copy operator."""

from __future__ import annotations

from typing import Any

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import copy as _common_copy

__all__ = ["copy"]


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    sync: bool = True,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
):
    """Copy between memory regions.

    ``sync=False`` lowers a contiguous panel to ``__memcpy_async`` and
    returns before the DMA finishes. The caller must ``T.sync("io")``
    before reading a GDRAM load or overwriting its destination.
    ``sync=True`` keeps the current heuristic: a single-slot load stays
    ``__memcpy``, a ring slot stays async.
    """
    ann: dict = dict(annotations) if annotations is not None else {}
    if "mlu_sync" not in ann and not sync:
        ann["mlu_sync"] = tirx.IntImm("int32", 0)
    return _common_copy(
        src,
        dst,
        coalesced_width=coalesced_width,
        annotations=ann or None,
        loop_layout=loop_layout,
    )
