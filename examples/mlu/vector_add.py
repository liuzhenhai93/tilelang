"""Add one to a length-1024 vector on MLU590.

Four tasks each own 256 elements. Compile and run with:

    python examples/mlu/vector_add.py
"""

import torch

import tilelang
import tilelang.mlu.language as T


@tilelang.jit(target="mlu")
def add_one(n: int = 1024):
    @T.prim_func
    def kernel(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(4) as bx:
            for i in T.serial(n // 4):
                B[bx * (n // 4) + i] = A[bx * (n // 4) + i] + 1.0

    return kernel


def main() -> None:
    n = 1024
    kernel = add_one(n)
    source = torch.arange(n, device="mlu", dtype=torch.float32)
    output = torch.empty(n, device="mlu", dtype=torch.float32)
    kernel(source, output)
    mismatches = int((output != source + 1).sum().item())
    print(f"{'PASS' if mismatches == 0 else 'FAIL'} mismatches={mismatches}")
    if mismatches != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
