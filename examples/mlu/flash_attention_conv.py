"""Flash Attention forward on MLU590 -- conv-engine execution model.

Identical to flash_attention.py (flat Union1 launch, per-IPU fragments,
hand-pipelined double-buffer ring, no-max exp2 softmax) except that both
matmuls run on the conv engine instead of __bang_matmul:

  QK: conv.nram ... .mul.scalar.rn(scale)   -- scale fused into the matmul
  PV: conv.nram ... .add.partial.rn([dst])  -- in-place accumulation

Measured on the pod (bench_conv_vs_gemm.py, 5000 iters, 4 tasks): the
gemm chain runs at 4.6 TF aggregate while the conv chain runs at 124 TF
with exact numerics -- the conv engine is ~20x faster per core at these
shapes and explains cnnl SDPA's 51.5 TF (its dump is conv-based too).

Encoding (decoded from the Triton-MLU fa_dump and verified exact):
  conv.nram.rn.f32.bf16.bf16 [dst],[in],[filter], ci, rows,
      1, 1, 1, 1, 1, co, .mstride(1), .kstride(ci*2 bytes),
      .indilation(1, 1), <fusion modifier>
param2 is the per-call row count (one call covers all rows; the output is
plain row-major f32). Filters are (co, ci) matrices packed by
__bang_reshape_filter (co, 1, 1, ci).

HARDWARE CONSTRAINT (probe_wramoff.py): the conv filter operand's WRAM
base must be below 32 KB (0 and 16 KB bases verified exact; 32/48 KB
bases fault with CNRT status 632015 even in a 256 KB arena), while the
payload may cross the 32 KB boundary. With a 16 KB K filter and a 32 KB
augmented V filter, only the K filter double-buffers: the lookahead
branch reshapes the next K panel into the single wk (the current panel's
K conv retired long before), and the next V panel is reshaped into the
single wv only after the current conv_pv retired (T.sync("io") between).
"""

import time

import torch
import tilelang
import tilelang.mlu.language as T


@tilelang.jit(target="mlu")
def flash_attention_conv(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    dim: int,
    dtype: str = "bfloat16",
    block_m: int = 128,
    block_n: int = 128,
    softmax_shift: float | None = None,
    pv_co_override: int | None = None,
):
    assert dtype == "bfloat16", "conv path is bf16-only for now"
    # WRAM base rule: wk = (block_n, dim) sits at base 0, wv at base
    # block_n*dim*2 which must stay below 32 KB AND avoid the silently-
    # wrong 8K base (probe_wramoff8k.py) -- the working combos are
    # (dim=64, block_n=128) and (dim=128, block_n=64), both wv@16K.
    assert (dim, block_n) in ((64, 128), (128, 64)), \
        "conv helpers are specialized to the WRAM-legal (dim, block_n) pairs"
    assert seq_q % block_m == 0 and seq_kv % block_n == 0
    assert (seq_q // block_m) % 4 == 0, "task-per-IPU needs tiles % 4 == 0"
    assert block_m % 16 == 0 and block_n % 64 == 0
    # Aligned with the mainline (flash_attention.py): f32-PV via .cvt.src
    # (no cast pass) and the shift softmax (conv bias folds -C, plain
    # single-pass exp2). The queue-pair sync structure is unchanged.
    scale = (1.0 / dim) ** 0.5 * 1.4426950408889634
    qk_scale = (
        scale if softmax_shift is None
        else (1.0 / dim) ** 0.5 * 1.4426950408889634
    )
    qk_bias = (
        0.0 if softmax_shift is None else -softmax_shift * 1.4426950408889634
    )
    # One ones row holds the row sum. reshape_filter needs co % 64 == 0,
    # so the filter width is dim + 1 rounded up, not dim * 2.
    # pv_co_override: diagnostic -- force the padded width. NOTE: pv_co
    # must be a POWER OF TWO here: a non-pow2 fragment extent (192 at
    # dim=128) degenerates the elementwise-transpose loop's layout and
    # scalarizes the whole panel (measured 43x: 5.06ms vs 0.117ms). The
    # mainline (flash_attention.py) has no elementwise loop over v (it
    # uses T.bang.transpose), so it keeps the tighter %64 rounding.
    pv_co = pv_co_override or (1 << int(dim).bit_length())

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch, heads, seq_q, dim), dtype),
        K: T.Tensor((batch, heads, seq_kv, dim), dtype),
        V: T.Tensor((batch, heads, seq_kv, dim), dtype),
        O: T.Tensor((batch, heads, seq_q, dim), dtype),
    ):
        ntiles = T.ceildiv(seq_q, block_m)
        tiles_per_task = max(1, -(-int(ntiles) // 32))
        with T.Kernel(T.ceildiv(ntiles, tiles_per_task), heads, batch) as (bx, by, bz):
            q = T.alloc_fragment((block_m, dim), dtype)
            # One score: softmax of this panel finishes before the next QK.
            score = T.alloc_fragment((block_m, block_n), "float32")
            output = T.alloc_fragment((block_m, pv_co), "float32")
            row_sum = T.alloc_fragment((block_m,), "float32")
            # Single K/V slots. Next K is loaded after QK, next V after PV.
            k = T.alloc_fragment((block_n, dim), dtype)
            v_t = T.alloc_fragment((block_n, dim), dtype)
            v = T.alloc_fragment((pv_co, block_n), dtype)
            # single-slot conv filters (WRAM base must stay below 32 KB;
            # see the module docstring)
            wk = T.alloc_wram((block_n, dim), dtype)
            wv = T.alloc_wram((pv_co, block_n), dtype)

            # rows [dim, pv_co) all = 1.0: column `dim` of the PV output is
            # the row sum; the padded columns are garbage-but-unused. The
            # full-range fill is required for fragment layout inference
            # (a single-row write leaves the fragment "not replicated").
            for i, j in T.Parallel(pv_co - dim, block_n):
                v[dim + i, j] = 1.0

            nko = T.ceildiv(seq_kv, block_n)
            for t in T.serial(tiles_per_task):
                row0 = (bx * tiles_per_task + t) * block_m
                T.copy(Q[bz, by, row0 : row0 + block_m, :], q, sync=False)
                T.copy(K[bz, by, 0 : block_n, :], k, sync=False)
                # K's filter transform begins only after its DMA (and Q's
                # DMA) has arrived; IO remains free after this edge issues.
                T.sync_queue_pair("move", "io")
                T.conv.reshape_filter(wk, k, block_n, dim)
                T.copy(V[bz, by, 0 : block_n, :], v_t, sync=False)
                # QK waits for K's filter. V's DMA stays in flight until the
                # loop transposes it.
                T.sync_queue_pair("compute", "move")
                T.conv.matmul(score, q, wk, dim, block_m, block_n,
                              scale=qk_scale, bias=qk_bias)
                T.fill(output, 0.0)
                T.fill(row_sum, 0.0)

                for ko in T.serial(nko):
                    # Transpose waits for V's DMA. The next K is a later IO,
                    # so it runs alongside the transpose and the QK still in flight.
                    T.sync_queue_pair("move", "io")
                    if ko + 1 < nko:
                        T.copy(
                            K[bz, by, (ko + 1) * block_n : (ko + 2) * block_n, :],
                            k,
                            sync=False,
                        )
                    for n, d in T.Parallel(block_n, dim):
                        v[d, n] = v_t[n, d]
                    T.conv.reshape_filter(wv, v, pv_co, block_n)
                    # exp2 reads score, so it waits for QK.
                    T.sync_queue_pair("move", "compute")
                    if softmax_shift is None:
                        for i, j in T.Parallel(block_m, block_n):
                            score[i, j] = T.exp2(T.min(score[i, j], 100.0))
                    else:
                        # the conv bias already shifted scores <= 0
                        for i, j in T.Parallel(block_m, block_n):
                            score[i, j] = T.exp2(score[i, j])
                    # PV reads the V filter produced by reshape_filter.
                    T.sync_queue_pair("compute", "move")
                    T.conv.matmul(
                        output, score, wv, block_n, block_m, pv_co, accumulate=True
                    )
                    if ko + 1 < nko:
                        # K's filter waits for its DMA; QK then waits for that filter.
                        T.sync_queue_pair("move", "io")
                        T.conv.reshape_filter(wk, k, block_n, dim)
                        T.sync_queue_pair("compute", "move")
                        T.copy(
                            V[bz, by, (ko + 1) * block_n : (ko + 2) * block_n, :],
                            v_t,
                            sync=False,
                        )
                        T.conv.matmul(score, q, wk, dim, block_m, block_n,
                                      scale=qk_scale, bias=qk_bias)

                # The last PV is compute; the reciprocal and the store read it.
                # T.sync_queue_pair("compute", "compute")
                for i in T.Parallel(block_m):
                    row_sum[i] = 1.0 / output[i, dim]
                for i, j in T.Parallel(block_m, pv_co):
                    output[i, j] = output[i, j] * row_sum[i]
                T.sync_queue_pair("io", "compute")
                out_full = T.alloc_fragment((block_m, pv_co), dtype)
                for i, j in T.Parallel(block_m, pv_co):
                    out_full[i, j] = T.cast(output[i, j], dtype)
                row_bytes = dim * 2
                for r in T.serial(block_m):
                    T.copy(out_full[r, 0:dim], O[bz, by, row0 + r, :])

            # Wait for all IO to complete.
            T.sync("io")

    return kernel


def main() -> None:
    import sys

    batch, heads, dim = 1, 1, 64
    scale = dim**-0.5
    block_m, block_n = 128, 128
    if len(sys.argv) > 2:
        shapes = ((int(sys.argv[1]), int(sys.argv[2])),)
    else:
        shapes = ((2048, 2048), (8192, 8192), (16384, 16384))
    for seq_q, seq_kv in shapes:
        kernel = flash_attention_conv(
            batch, heads, seq_q, seq_kv, dim, dtype="bfloat16",
            block_m=block_m, block_n=block_n,
        )
        q = torch.randn(batch, heads, seq_q, dim, device="mlu", dtype=torch.float32).to(torch.bfloat16)
        k = torch.randn(batch, heads, seq_kv, dim, device="mlu", dtype=torch.float32).to(torch.bfloat16)
        v = torch.randn(batch, heads, seq_kv, dim, device="mlu", dtype=torch.float32).to(torch.bfloat16)
        output = torch.empty_like(q)
        for _ in range(2):
            kernel(q, k, v, output)
        torch.mlu.synchronize()
        iters = 10
        start = time.perf_counter()
        for _ in range(iters):
            kernel(q, k, v, output)
        torch.mlu.synchronize()
        elapsed_ms = (time.perf_counter() - start) / iters * 1e3
        q32, k32, v32 = q.float(), k.float(), v.float()
        reference = torch.softmax(q32 @ k32.transpose(-1, -2) * scale, dim=-1) @ v32
        torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-2)
        tf = 4 * batch * heads * seq_q * seq_kv * dim / (elapsed_ms * 1e-3) / 1e12
        print(
            f"PASS seq_q={seq_q} seq_kv={seq_kv} block_m={block_m} "
            f"block_n={block_n} {elapsed_ms:.3f} ms {tf:.1f} TF"
        )


if __name__ == "__main__":
    main()