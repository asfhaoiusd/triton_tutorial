"""Triton 入门：向量加法。

运行（需 CUDA）：
    python kernel/triton/tutorial/01_vector_add/step01_add_1d.py
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前 program 编号
    pid = tl.program_id(axis=0)
    # 本 program 负责的全局下标
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 边界保护
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.numel() == y.numel()
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n)
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    n = 100_000
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)

    out = add(x, y)
    ref = x + y
    ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
    print(f"allclose: {ok}")
    print(f"max abs err: {(out - ref).abs().max().item():.3e}")

    ms = triton.testing.do_bench(lambda: add(x, y))
    print(f"triton add bench: {ms:.3f} ms")


if __name__ == "__main__":
    main()
