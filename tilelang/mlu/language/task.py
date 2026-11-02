"""Task identifiers exposed to MLU kernels.

A cluster kernel's linear task id is ``cluster * 4 + ipu``; ``taskIdX % 4``
selects the IPU within the cluster. Kernels use it to partition per-IPU
work (e.g. the chunked softmax panel) that the layout system assigns to
individual IPUs.
"""

from __future__ import annotations

from tvm import tirx


def task_id() -> "tirx.PrimExpr":
    """The linear task id as an int32 expression."""
    return tirx.call_extern("int32", "taskIdX")

def task_dim() -> "tirx.PrimExpr":
    """The total task count (taskDim) as an int32 expression."""
    return tirx.call_extern("int32", "taskDim")
