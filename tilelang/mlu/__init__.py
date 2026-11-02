"""MLU590 target selection.

``mlu``, ``mlu590`` and ``mtp_592`` all mean BANG arch mtp_592. Auto-detection
requires a live ``torch.mlu`` device; a CNToolkit install by itself does not
claim the host, so it does not shadow CUDA or Ascend.
"""

from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer

_MLU590_NAMES = frozenset({"mlu", "mlu590", "mtp_592"})


def _make_mlu_target() -> Target:
    """Build the MLU590 target. Other attributes use the C++ kind defaults."""
    return Target({"kind": "mlu", "mcpu": "mtp_592", "arch": "mtp_592"})


def check_mlu_availability() -> bool:
    """Return whether this process can see an MLU through ``torch.mlu``."""
    try:
        import torch

        mlu = getattr(torch, "mlu", None)
        return mlu is not None and bool(mlu.is_available())
    except Exception:
        return False


def normalize_mlu_target(target: TargetLike) -> Target | None:
    """Accept the MLU590 names. Return None for every other input."""
    if not isinstance(target, str) or target.strip().lower() not in _MLU590_NAMES:
        return None
    try:
        return _make_mlu_target()
    except Exception:
        return None


register_target_detector(
    "mlu",
    lambda: _make_mlu_target() if check_mlu_availability() else None,
    override=True,
)
register_target_normalizer("mlu", normalize_mlu_target, override=True)

from . import backend as backend  # noqa: E402,F401
from .op import gemm as gemm  # noqa: E402,F401
