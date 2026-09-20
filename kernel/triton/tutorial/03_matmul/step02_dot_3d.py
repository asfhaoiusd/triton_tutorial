"""Triton：三维张量点积 sum(x * y)（Frobenius 内积，结果为标量）。

对应 PyTorch: (x * y).sum() 或 torch.dot(x.flatten(), y.flatten())
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 8, "BLOCK_P": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_P": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_P": 8}, num_warps=8),
    ],
    key=["M", "N", "P"],
    # atomic_add 累加到同一标量；benchmark 前必须清零，否则结果被污染
    reset_to_zero=["out_ptr"],
)
@triton.jit
def dot_kernel_3d(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    P,
    stride_xm,
    stride_xn,
    stride_xp,
    stride_ym,
    stride_yn,
    stride_yp,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_p = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    offs_x = (
        offs_m[:, None, None] * stride_xm
        + offs_n[None, :, None] * stride_xn
        + offs_p[None, None, :] * stride_xp
    )
    offs_y = (
        offs_m[:, None, None] * stride_ym
        + offs_n[None, :, None] * stride_yn
        + offs_p[None, None, :] * stride_yp
    )
    mask = (
        (offs_m[:, None, None] < M)
        & (offs_n[None, :, None] < N)
        & (offs_p[None, None, :] < P)
    )

    x = tl.load(x_ptr + offs_x, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs_y, mask=mask, other=0.0)

    # 本块内三维求和，再原子加到同一个标量
    prod = x * y
    acc = tl.sum(tl.sum(tl.sum(prod, axis=0), axis=0), axis=0)
    tl.atomic_add(out_ptr, acc)


def dot_three_dimension(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape and x.ndim == 3
    out = torch.zeros((), device=x.device, dtype=x.dtype)
    M, N, P = x.shape
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        triton.cdiv(P, meta["BLOCK_P"]),
    )
    dot_kernel_3d[grid](
        x,
        y,
        out,
        M,
        N,
        P,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
    )
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    M, N, P = 128, 64, 32
    x = torch.randn(M, N, P, device="cuda", dtype=torch.float32)
    y = torch.randn(M, N, P, device="cuda", dtype=torch.float32)

    o_torch = (x * y).sum()
    o_flat = torch.dot(x.flatten(), y.flatten())
    o_triton = dot_three_dimension(x, y)

    print(f"torch sum(x*y): {o_torch.item():.6f}")
    print(f"torch flat dot: {o_flat.item():.6f}")
    print(f"triton        : {o_triton.item():.6f}")
    print(
        f"allclose: {torch.allclose(o_torch, o_triton, rtol=1e-3, atol=1e-3)}"
    )
    print(f"shape: {(M, N, P)}, bytes/tensor: {x.nbytes / 1024**2:.1f} MiB")
    print(f"naive : {triton.testing.do_bench(lambda: (x * y).sum()):.3f} ms")
    print(f"triton: {triton.testing.do_bench(lambda: dot_three_dimension(x, y)):.3f} ms")
    print(f"torch flat dot: {triton.testing.do_bench(lambda: torch.dot(x.flatten(), y.flatten())):.3f} ms")


if __name__ == "__main__":
    main()
