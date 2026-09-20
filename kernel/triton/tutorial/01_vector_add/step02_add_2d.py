"""Triton 入门：二维矩阵加法。"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def add_kernel_two_dimension(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 广播成 [BLOCK_M, BLOCK_N]；各自用自己的 stride（转置/非连续时可能不同）
    offs_x = offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    offs_y = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    offs_o = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs_x, mask=mask)
    y = tl.load(y_ptr + offs_y, mask=mask)
    tl.store(out_ptr + offs_o, x + y, mask=mask)


def add_two_dimension(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape and x.ndim == 2
    out = torch.empty_like(x)
    M, N = x.shape
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    add_kernel_two_dimension[grid](
        x,
        y,
        out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    M, N = 1000, 800
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y = torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = add_two_dimension(x, y)
    ref = x + y
    ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
    print(f"allclose: {ok}")
    print(f"max abs err: {(out - ref).abs().max().item():.3e}")

    ms_torch = triton.testing.do_bench(lambda: x + y)
    ms_triton = triton.testing.do_bench(lambda: add_two_dimension(x, y))
    print(f"torch  add:  {ms_torch:.3f} ms")
    print(f"triton add:  {ms_triton:.3f} ms")
    print(f"speedup (torch/triton): {ms_torch / ms_triton:.2f}x")


if __name__ == "__main__":
    main()
