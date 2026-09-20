"""Triton：一维向量点积 x · y（对应 torch 的 x @ y / torch.dot）。"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
    # atomic_add 累加到同一标量；benchmark 前必须清零，否则结果被污染
    reset_to_zero=["out_ptr"],
)
@triton.jit
def dot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    # 本块内先求和，再原子加到标量输出（多 program 并行）
    acc = tl.sum(x * y, axis=0)
    tl.atomic_add(out_ptr, acc)


def dot_multiplication(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.ndim == 1 and y.ndim == 1 and x.numel() == y.numel()
    out = torch.zeros((), device=x.device, dtype=x.dtype)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    dot_kernel[grid](x, y, out, n)
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    n = 8128
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)

    o_naive = x @ y
    o_torch = torch.matmul(x, y)
    o_triton = dot_multiplication(x, y)

    print(f"naive : {o_naive.item():.6f}")
    print(f"torch : {o_torch.item():.6f}")
    print(f"triton: {o_triton.item():.6f}")
    print(
        f"allclose naive/triton: {torch.allclose(o_naive, o_triton, rtol=1e-4, atol=1e-4)}"
    )
    print(
        f"allclose torch/triton: {torch.allclose(o_torch, o_triton, rtol=1e-4, atol=1e-4)}"
    )
    print(f"shape: {(n,)}, bytes/tensor: {x.nbytes / 1024**2:.1f} MiB")
    print(f"torch  @     : {triton.testing.do_bench(lambda: x @ y):.3f} ms")
    print(f"torch matmul : {triton.testing.do_bench(lambda: torch.matmul(x, y)):.3f} ms")
    print(
        f"triton dot   : {triton.testing.do_bench(lambda: dot_multiplication(x, y)):.3f} ms"
    )


if __name__ == "__main__":
    main()
