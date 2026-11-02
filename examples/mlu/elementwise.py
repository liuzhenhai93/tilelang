"""BANG NFU elementwise lowering for contiguous NRAM tiles.

Two kernels:
- ``elementwise``: a scalar loop the compiler lowers to one vectorized
  ``__tl_mlu_ew_*`` chain (no explicit BANG calls in the script).
- ``exp_less0``: the raw ``__bang_active_exp_less_0`` intrinsic via
  ``T.bang.*`` -- semantically ``e^min(x, 0)`` (1.0 at and above 0, with
  an internal ~2^-24 output floor observed for very negative x). This is
  the building block of a one-pass no-max softmax:
  ``p = exp_less0(s*scale - C)`` with a caller-chosen C >= max logit.
"""

import torch

import tilelang
import tilelang.mlu.language as T


@tilelang.jit(target="mlu")
def elementwise(n: int = 64, dtype: str = "float32"):
    """Compute ``exp2((A - B) * 0.5)`` with NRAM-resident vectors."""
    assert n % 64 == 0

    @T.prim_func
    def kernel(
        A: T.Tensor((n,), dtype),
        B: T.Tensor((n,), dtype),
        O: T.Tensor((n,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_fragment((n,), dtype)
            b = T.alloc_fragment((n,), dtype)
            out = T.alloc_fragment((n,), dtype)

            T.copy(A, a)
            T.copy(B, b)
            for i in T.Parallel(n):
                out[i] = T.exp2((a[i] - b[i]) * 0.5)
            T.copy(out, O)

    return kernel


@tilelang.jit(target="mlu")
def exp_less0(n: int = 4096):
    """Compute ``exp_less0(x) = e^min(x, 0)`` via the raw intrinsic."""
    assert n % 64 == 0

    @T.prim_func
    def kernel(
        X: T.Tensor((n,), "float32"),
        O: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1):
            x = T.alloc_fragment((n,), "float32")
            out = T.alloc_fragment((n,), "float32")

            T.copy(X, x)
            T.sync("move")
            T.bang.active_exp_less_0(out, x, n)
            T.sync("io")
            T.copy(out, O)

    return kernel


def main() -> None:
    n = 64
    for dtype in ("float32", "bfloat16"):
        torch_dtype = getattr(torch, dtype)
        a = torch.randn(n, device="mlu", dtype=torch.float32).to(torch_dtype)
        b = torch.randn(n, device="mlu", dtype=torch.float32).to(torch_dtype)
        out = torch.empty_like(a)
        elementwise(n, dtype)(a, b, out)
        reference = torch.exp2((a.float() - b.float()) * 0.5)
        torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)
        print(f"PASS elementwise n={n} dtype={dtype}")

    n2 = 4096
    x = torch.tensor(
        [-30.0, -10.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 10.0, 30.0] + [0.0] * (n2 - 9),
        device="mlu", dtype=torch.float32,
    )
    out = torch.zeros(n2, device="mlu", dtype=torch.float32)
    exp_less0(n2)(x, out)
    y = out[:9].cpu().tolist()
    expected = [torch.exp(torch.tensor(min(v, 0.0))).item() for v in x[:9].cpu().tolist()]
    for xi, yi, ei in zip(x[:9].cpu().tolist(), y, expected):
        print(f"  exp_less0({xi:8.3g}) = {yi:.6g}  (e^min(x,0) = {ei:.6g})")
    ok = all(abs(yi - ei) <= 0.02 * max(1.0, abs(ei)) for yi, ei in zip(y, expected))
    print(("PASS" if ok else "MISMATCH"), "exp_less0 semantics (e^min(x,0))")


if __name__ == "__main__":
    main()