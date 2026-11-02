"""Differential MLISA sweep: recompile the mini-FA Triton kernel with one
knob changed at a time; each config dumps its own mlisa so the constant
operand fields of ld.3d / trans.tiling can be decoded by comparison.
"""

import os
import shutil
import sys

import torch
import torch_mlu
import triton
import triton.language as tl


@triton.jit
def fa_kernel(
    Q, K, V, O,
    seq_kv, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, DIM)
    q = tl.load(Q + pid * BLOCK_M * DIM + offs_m[:, None] * DIM + offs_d[None, :])
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, DIM], tl.float32)
    LOG2E: tl.constexpr = 1.4426950408889634
    for start in range(0, seq_kv, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        k = tl.load(K + offs_n[:, None] * DIM + offs_d[None, :])
        v = tl.load(V + offs_n[:, None] * DIM + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        p = tl.math.exp2((s - m_new[:, None]) * LOG2E)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + pid * BLOCK_M * DIM + offs_m[:, None] * DIM + offs_d[None, :], acc.to(tl.bfloat16))


CONFIGS = {
    # tag: (BLOCK_M, BLOCK_N, DIM)
    "base": (128, 128, 64),
    "bn64": (128, 64, 64),
    "bm64": (64, 128, 64),
    "dim128": (128, 128, 128),
}


def main() -> None:
    tag = sys.argv[1]
    block_m, block_n, dim = CONFIGS[tag]
    seq_q, seq_kv = 256, 2048
    q = (torch.randn(1, seq_q, dim, device="mlu", dtype=torch.float32) * 0.3).to(torch.bfloat16)
    k = (torch.randn(1, seq_kv, dim, device="mlu", dtype=torch.float32) * 0.3).to(torch.bfloat16)
    v = (torch.randn(1, seq_kv, dim, device="mlu", dtype=torch.float32) * 0.3).to(torch.bfloat16)
    o = torch.empty_like(q)
    fa_kernel[(seq_q // block_m,)](
        q, k, v, o, seq_kv, dim**-0.5,
        BLOCK_M=block_m, BLOCK_N=block_n, DIM=dim, num_warps=4,
    )
    torch.mlu.synchronize()
    dst = f"/tmp/fa_sweep/{tag}"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree("/tmp/fa_dump", dst)
    print(f"DUMPED {tag} -> {dst}")


if __name__ == "__main__":
    main()