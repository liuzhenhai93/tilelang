"""Per-IPU ``T.gemm`` on MLU590 with global operands.

``T.Kernel`` counts tasks along M: task ``bx`` owns the ``TILE_M``-row
tile ``bx * TILE_M``. Union1 launches are cluster-granular, so the task
count (``m / TILE_M``) must be a multiple of 4. M is a positive multiple
of ``TILE_M`` (at most 64 per gemm call); N and K are multiples of 64.

    python examples/mlu/example_gemm.py
"""

import torch

import tilelang
import tilelang.mlu.language as T


TILE_M = 64


@tilelang.jit(target="mlu")
def matmul(m: int = 128, n: int = 128, k: int = 128):
    """Build C[m, n] = A[m, k] @ B[k, n] as one cluster per M tile."""

    @T.prim_func
    def kernel(
        A: T.Tensor((m, k), "float32"),
        B: T.Tensor((k, n), "float32"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(T.ceildiv(m, TILE_M)) as bx:
            m0 = bx * TILE_M
            T.gemm(
                A[m0 : m0 + TILE_M, :],
                B,
                C[m0 : m0 + TILE_M, :],
                clear_accum=True,
            )

    return kernel


def main() -> None:
    m, n, k = 256, 128, 128
    kernel = matmul(m, n, k)
    a = torch.randn(m, k, device="mlu", dtype=torch.float32)
    b = torch.randn(k, n, device="mlu", dtype=torch.float32)
    output = torch.empty(m, n, device="mlu", dtype=torch.float32)
    kernel(a, b, output)
    torch.testing.assert_close(output, a @ b, rtol=1e-4, atol=1e-4)
    print(f"PASS matmul M={m} N={n} K={k}")


if __name__ == "__main__":
    main()
