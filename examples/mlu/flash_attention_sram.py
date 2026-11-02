"""Flash Attention forward on MLU590 -- cluster-SRAM pipeline (SPMD).

taskDim = ntiles (one tile per task, 4 waves). Every task fills the
cluster's SRAM with the next K/V panel; the fills of the 4 tasks sharing
a cluster write IDENTICAL data, so they are idempotent and need no
cross-task coordination -- the cluster's GDRAM read traffic is a quarter
of the per-IPU form. Synchronization uses `T.sync("cluster")`, whose
barrier tags (pio/cio) drain every context's DMA queues and publish
SRAM writes cluster-wide; the LowerClusterStaging pass rewrites it for
the wave structure so the never-executing MPU contexts are excluded.

Per panel: barrier (publishes last iteration's fill, drains every
context's DMA queues) -> consume SRAM->NRAM -> refill lookahead for
panel ko+1 -> QK gemm -> no-max softmax -> PV gemm. The panel-0 fill
happens in the prologue; the barrier ordering guarantees no task
refills a slot while another task is still reading it.
"""

import time

import torch
import tilelang
import tilelang.mlu.language as T


@tilelang.jit(target="mlu")
def flash_attention_sram(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    dim: int,
    dtype: str = "bfloat16",
    block_m: int = 64,
    block_n: int = 128,
):
    assert dtype in ("float32", "bfloat16")
    assert dim % 64 == 0
    assert seq_q % block_m == 0 and seq_kv % block_n == 0
    assert block_m % 16 == 0 and block_n % 64 == 0
    scale = (1.0 / dim) ** 0.5 * 1.4426950408889634
    ntiles = seq_q // block_m
    panels = seq_kv // block_n
    elem = 2 if dtype == "bfloat16" else 4
    q_bytes = block_m * dim * elem
    row_bytes = dim * elem

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch, heads, seq_q, dim), dtype),
        K: T.Tensor((batch, heads, seq_kv, dim), dtype),
        V: T.Tensor((batch, heads, seq_kv, dim), dtype),
        O: T.Tensor((batch, heads, seq_q, dim), dtype),
    ):
        with T.Kernel(ntiles, heads, batch) as (bx, by, bz):
            q = T.alloc_fragment((block_m, dim), dtype)
            score = T.alloc_fragment((block_m, block_n), "float32")
            output = T.alloc_fragment((block_m, dim * 2), "float32")
            row_sum = T.alloc_fragment((block_m,), "float32")
            score_bf = T.alloc_fragment((block_m, block_n), "bfloat16")
            kbuf = T.alloc_fragment((block_n, dim), dtype)
            v_t = T.alloc_fragment((block_n, dim), dtype)
            v0 = T.alloc_fragment((dim * 2, block_n), dtype)
            out_full = T.alloc_fragment((block_m, dim * 2), dtype)
            # static cluster SRAM: plane = slot*2 + (0=K, 1=V); the 3-D
            # shape keeps every T.copy region a 2-D slice (DMA-matchable)
            sbuf = T.alloc_shared((4, block_n, dim), dtype, scope="shared")

            # constant augmented half of v0 (V^T + ones)
            for i, j in T.Parallel(dim, block_n):
                v0[dim + i, j] = 1.0
            # panel 0 must be filled before the loop: the loop only
            # refills panel ko+1, so without this the first consume
            # reads stale SRAM left over from the previous invocation
            T.copy(K[bz, by, 0:block_n, :], sbuf[0, :, :])
            T.copy(V[bz, by, 0:block_n, :], sbuf[1, :, :])
            # this task's Q tile (bx is the task id)
            T.copy(Q[bz, by, bx * block_m : (bx + 1) * block_m, :], q)
            T.fill(output, 0.0)
            T.fill(row_sum, 0.0)
            T.sync("io")

            for ko in T.serial(panels):
                # the barrier publishes the panel-ko fill issued at the
                # end of the previous iteration (or the prologue) and --
                # via its pio/cio tags -- drains every context's DMA
                # queues, so all tasks are done consuming panel ko-1's
                # slot before this iteration refills it
                T.sync("cluster")
                # consumption: on-die SRAM -> private NRAM
                T.copy(sbuf[(ko % 2) * 2, :, :], kbuf)
                T.copy(sbuf[(ko % 2) * 2 + 1, :, :], v_t)
                # refill lookahead: panel ko+1 into the slot just
                # vacated by panel ko-1; issued AFTER the barrier that
                # confirmed every task finished reading panel ko-1,
                # and published by next iteration's barrier
                if ko + 1 < panels:
                    nxt = (ko + 1) % 2
                    psrc = (ko + 1) * block_n
                    T.copy(
                        K[bz, by, psrc : psrc + block_n, :],
                        sbuf[nxt * 2, :, :],
                    )
                    T.copy(
                        V[bz, by, psrc : psrc + block_n, :],
                        sbuf[nxt * 2 + 1, :, :],
                    )
                T.sync("io")
                # V^T half of the augmented V panel (the ones half is
                # constant, filled in the prologue)
                for n, d in T.Parallel(block_n, dim):
                    v0[d, n] = v_t[n, d]
                # QK gemm + no-max softmax + PV gemm
                T.gemm(q, kbuf, score, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_m, block_n):
                    score[i, j] = T.exp2(T.min(score[i, j] * scale, 100.0))
                for i, j in T.Parallel(block_m, block_n):
                    score_bf[i, j] = T.cast(score[i, j], "bfloat16")
                T.gemm(score_bf, v0, output, transpose_B=True, clear_accum=False)

            # epilogue: normalize + store this task's tile
            for i in T.Parallel(block_m):
                # one scalar reciprocal per row (a subview divide would
                # lower to thousands of scalar fdiv)
                row_sum[i] = 1.0 / output[i, dim]
            for i, j in T.Parallel(block_m, dim * 2):
                output[i, j] = output[i, j] * row_sum[i]
            T.evaluate(T.sync("move"))
            for i, j in T.Parallel(block_m, dim * 2):
                out_full[i, j] = T.cast(output[i, j], dtype)
            for r in T.serial(block_m):
                T.copy(out_full[r, 0:dim], O[bz, by, bx * block_m + r, :])
    return kernel


def main() -> None:
    batch, heads, dim = 1, 1, 64
    scale = dim**-0.5
    block_m, block_n = 64, 128
    for seq_q, seq_kv in ((2048, 2048), (8192, 8192), (16384, 16384)):
        kernel = flash_attention_sram(
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