"""MLU language dialect: common TileLang plus MLU extensions.

Mirrors ``tilelang/ascend/language``: kernels import the whole dialect as
one namespace -- ``import tilelang.mlu.language as T`` -- and get the
common surface (``T.Tensor``/``T.Kernel``/``T.copy``/``T.gemm``/...)
plus the MLU extensions (``T.conv.*``, ``T.bang.*``, ``T.alloc_wram``,
``T.sync``, ``T.sync_io_move_compute``, ``T.sync_queue_pair``,
``T.task_id``) from a single import.
"""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

from .alloc import alloc_wram  # noqa: F401
from .bang import bang  # noqa: F401 (exposed as T.bang.*)
from .conv import conv  # noqa: F401 (exposed as T.conv.*)
from .copy_op import copy  # noqa: F401 (shadows the common copy)
from .staging import stage_load  # noqa: F401
from .sync import sync, sync_io_move_compute, sync_queue_pair  # noqa: F401
from .task import task_dim, task_id  # noqa: F401

__tilelang_dialect__ = "mlu"

_MLU_API_ALL = (
    "alloc_wram",
    "bang",
    "conv",
    "copy",
    "stage_load",
    "sync",
    "sync_io_move_compute",
    "sync_queue_pair",
    "task_dim",
    "task_id",
)

__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_MLU_API_ALL)))