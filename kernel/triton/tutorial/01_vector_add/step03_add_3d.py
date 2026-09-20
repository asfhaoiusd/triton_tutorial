"""Triton 入门：三维张量加法。"""

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
)
@triton.jit
def add_kernel_three_dimension(
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
    stride_om,
    stride_on,
    stride_op,
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
    offs_o = (
        offs_m[:, None, None] * stride_om
        + offs_n[None, :, None] * stride_on
        + offs_p[None, None, :] * stride_op
    )
    mask = (
        (offs_m[:, None, None] < M)
        & (offs_n[None, :, None] < N)
        & (offs_p[None, None, :] < P)
    )

    x = tl.load(x_ptr + offs_x, mask=mask)
    y = tl.load(y_ptr + offs_y, mask=mask)
    tl.store(out_ptr + offs_o, x + y, mask=mask)


def add_three_dimension(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape and x.ndim == 3
    out = torch.empty_like(x)
    M, N, P = x.shape
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        triton.cdiv(P, meta["BLOCK_P"]),
    )
    add_kernel_three_dimension[grid](
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    M, N, P = 256, 128, 64
    x = torch.randn(M, N, P, device="cuda", dtype=torch.float32)
    y = torch.randn(M, N, P, device="cuda", dtype=torch.float32)
    out = add_three_dimension(x, y)
    ref = x + y
    ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
    print(f"allclose: {ok}")
    print(f"max abs err: {(out - ref).abs().max().item():.3e}")
    print(f"shape: {(M, N, P)}, bytes/tensor: {x.nbytes / 1024**2:.1f} MiB")

    ms_torch = triton.testing.do_bench(lambda: x + y)
    ms_triton = triton.testing.do_bench(lambda: add_three_dimension(x, y))
    print(f"torch  add:  {ms_torch:.3f} ms")
    print(f"triton add:  {ms_triton:.3f} ms")
    print(f"speedup (torch/triton): {ms_torch / ms_triton:.2f}x")


if __name__ == "__main__":
    main()
