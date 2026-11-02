"""T.bang.* - direct BANG intrinsic emission.

Mirrors ``tilelang/ascend/language/simd.py``: these functions emit raw BANG C
intrinsics so a kernel can express exactly what a hand-written mlu-ops kernel
expresses, without relying on the automatic pattern matchers.
``CodeGenBANG`` prints ``call_extern`` names verbatim, so no codegen changes
are needed.

All counts are element counts unless the intrinsic takes bytes (``move``).
NFU legality (128-byte alignment, counts divisible by 64 where required) is
the caller's responsibility, exactly as in mlu-ops.
"""

from __future__ import annotations

from tvm import tirx


def _call(name: str, *args):
    return tirx.call_extern("void", name, *args)


def _arg(x):
    """Address expressions pass through; BufferLoads and fragment handles
    get wrapped with ``address_of`` (the codegen prints a bare BufferLoad
    argument as a VALUE, not an address). BufferRegions (slices like
    ``buf[a:b, :]``) resolve to the sub-region's base address. Scalars
    pass through."""
    if isinstance(x, tirx.PrimExpr):
        return x
    if isinstance(x, (int, float, bool, str)):
        return x
    from tilelang.language import address_of

    if isinstance(x, tirx.Buffer):
        # full-rank zero index: the base address of the buffer
        return address_of(tirx.BufferLoad(x, [tirx.IntImm("int32", 0)] * len(x.shape)))
    if isinstance(x, tirx.BufferRegion):
        # slice view: the address of its first element
        return address_of(
            tirx.BufferLoad(x.buffer, [r.min for r in x.region])
        )
    try:
        return address_of(x[0, 0])
    except TypeError:
        return address_of(x[0])


class _BangIntrinsic:
    """A namespaced intrinsic: ``T.bang.add(dst, a, b, n)`` -> ``__bang_add``."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args):
        return _call(self._name, *[_arg(a) for a in args])


class BangNamespace:
    """``T.bang.<intrinsic>(...)`` emits ``__bang_<intrinsic>(...)``.

    Any BANG intrinsic works through ``T.bang.<name>``; ``T.bang.raw`` covers
    names that are not valid Python identifiers.
    """

    def __getattr__(self, name: str) -> _BangIntrinsic:
        return _BangIntrinsic("__bang_" + name)

    @staticmethod
    def raw(name: str, *args):
        """Emit ``__bang_<name>(...)`` for intrinsics without a wrapper."""
        return _call("__bang_" + name, *args)

    @staticmethod
    def move(dst, src, nbytes):
        """On-chip copy; ``nbytes`` is in BYTES."""
        return _call("__bang_move", _arg(dst), _arg(src), _arg(nbytes))

    @staticmethod
    def memcpy_async(dst, src, nbytes, direction):
        """Async GDRAM<->NRAM copy; ``direction`` is 0 (GDRAM2NRAM) or 1."""
        name = "GDRAM2NRAM" if direction == 0 else "NRAM2GDRAM"
        return _call(
            "__memcpy_async",
            _arg(dst),
            _arg(src),
            nbytes,
            tirx.call_extern("int32", name),
        )

    @staticmethod
    def memcpy(dst, src, nbytes, direction):
        """Synchronous GDRAM<->NRAM copy (drains the whole DMA queue)."""
        name = "GDRAM2NRAM" if direction == 0 else "NRAM2GDRAM"
        return _call(
            "__memcpy",
            _arg(dst),
            _arg(src),
            nbytes,
            tirx.call_extern("int32", name),
        )

    @staticmethod
    def transpose(dst, src, height, width):
        return _call("__bang_transpose", _arg(dst), _arg(src), height, width)


bang = BangNamespace()
