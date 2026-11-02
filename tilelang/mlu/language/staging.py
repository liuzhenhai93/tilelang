"""Cluster SRAM staging markers.

``stage_load(dst_fragment, src_ptr, byte_count)`` declares that
``dst_fragment`` is fed with ``byte_count`` bytes read from ``src_ptr``
through the cluster's shared SRAM, one panel ahead. The
``LowerClusterStaging`` pass lowers the markers into the producer/consumer
protocol (the producer runs on the MPU context with wave-matched rounds;
the script never mentions the MPU or the SRAM buffers).
"""

from __future__ import annotations

from tvm import tirx


def stage_load(dst, src_ptr, byte_count) -> None:
    """Declare a cluster-staged load into ``dst`` (a fragment)."""
    return tirx.call_extern("void", "tl.mlu_stage_load", dst, src_ptr, byte_count)