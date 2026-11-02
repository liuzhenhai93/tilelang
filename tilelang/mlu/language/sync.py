"""MLU queue and cluster synchronization primitives."""

from __future__ import annotations

from tvm import tirx


def sync(scope: str = "cluster"):
    """Queue/barrier synchronization for the MLU backend.

    ``"cluster"``  → ``__sync_cluster()``: cluster-wide barrier (MPU + IPUs).
    ``"io"``       → ``__sync_io()``: drain this IPU's async GDRAM↔NRAM copies.
    ``"move"``     → ``__sync_move()``: drain on-chip NRAM↔WRAM moves. Reserved:
                     ``T.gemm`` inserts these internally around its WRAM filter.
    ``"compute"``  → ``__sync_compute()``: drain the compute queues. Reserved:
                     ``T.gemm`` inserts these internally before reusing WRAM.

    ``sync`` only waits. To overlap a DMA with compute, issue the next
    ``T.copy`` early (the MLU lowering emits ``__memcpy_async`` for ring-buffer
    slots) and let a later ``sync("io")`` wait for it. Because ``T.gemm``
    internally drains the compute queues and copies its filter synchronously,
    a ring-buffer slot is safe to overwrite once the consuming ``T.gemm`` has
    returned.
    """
    if scope not in ("cluster", "io", "move", "compute"):
        raise ValueError(f"unknown sync scope: {scope}")
    return tirx.call_extern("void", "__sync_" + scope)


def sync_io_move_compute(
    ioproducer: bool = True,
    moveproducer: bool = True,
    computeproducer: bool = True,
    ioconsumer: bool = True,
    moveconsumer: bool = True,
    computeconsumer: bool = True,
):
    """``__sync_io_move_compute``: later consumer-queue ops wait for prior producer-queue ops.

    Arguments follow the BANG C order: three producer flags, then three
    consumer flags. Defaults are all true, which is the no-argument form:
    later IO, Move, and Compute wait until all three queues have drained.
    A queue that is both producer and consumer waits for its own prior ops.
    Producer flags cannot all be false, and consumer flags cannot all be false.

    ``T.sync_io_move_compute(True, False, False, False, True, False)`` is
    the MSDA edge: later Move waits for prior IO. Queues that are not
    consumers are not held, so the scalar keeps issuing them.
    """
    producers = (ioproducer, moveproducer, computeproducer)
    consumers = (ioconsumer, moveconsumer, computeconsumer)
    if not any(producers):
        raise ValueError("sync_io_move_compute requires a producer queue")
    if not any(consumers):
        raise ValueError("sync_io_move_compute requires a consumer queue")
    if all(producers) and all(consumers):
        return tirx.call_extern("void", "__sync_io_move_compute")
    return tirx.call_extern(
        "void",
        "__sync_io_move_compute",
        *[tirx.IntImm("int32", int(flag)) for flag in (*producers, *consumers)],
    )


def sync_queue_pair(consumer: str, producer: str):
    """Later ops on the first queue wait for prior ops on the second.

    ``T.sync_queue_pair("compute", "move")`` means Compute waits for Move.
    The queue being waited on is the second argument. Other queues keep
    issuing. Names are ``"io"``, ``"move"``, and ``"compute"``; passing the
    same name twice makes that queue wait for its own prior ops.
    """
    queues = ("io", "move", "compute")
    if consumer not in queues or producer not in queues:
        raise ValueError(
            f"sync_queue_pair queues must be io, move, or compute, got {consumer!r}, {producer!r}"
        )
    return sync_io_move_compute(
        *(name == producer for name in queues),
        *(name == consumer for name in queues),
    )
