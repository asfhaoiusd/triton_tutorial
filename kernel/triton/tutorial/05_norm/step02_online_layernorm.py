"""Triton：分块 LayerNorm（沿 N 累加，类似 online Softmax 的两趟扫描）

Pass1 维护:
    sum_x  += sum(tile)
    sum_x2 += sum(tile * tile)

然后:
    mean = sum_x / N
    var  = sum_x2 / N - mean^2

Pass2:
    y = gamma * (x - mean) / sqrt(var + eps) + beta

适合 N 很大、不能一次装进寄存器的情况。
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def online_layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_ob,
    stride_om,
    stride_on,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = offs_m < M

    x_batch = x_ptr + pid_b * stride_xb
    o_batch = o_ptr + pid_b * stride_ob

    # 每行一对累加器 [BLOCK_M]
    sum_x = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # -------- Pass 1: 沿 N 累加 sum_x / sum_x2 --------
    for start in range(0, N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (offs_n[None, :] < N)

        x_ptrs = (
            x_batch + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        )
        x_tile = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_x += tl.sum(x_tile, axis=1)
        sum_x2 += tl.sum(x_tile * x_tile, axis=1)

    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # -------- Pass 2: 写回归一化结果 --------
    for start in range(0, N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (offs_n[None, :] < N)
        w_mask = offs_n < N

        x_ptrs = (
            x_batch + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        )
        o_ptrs = (
            o_batch + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        )
        w = tl.load(w_ptr + offs_n, mask=w_mask, other=0.0)
        bias = tl.load(b_ptr + offs_n, mask=w_mask, other=0.0)
        x_tile = tl.load(x_ptrs, mask=mask, other=0.0)

        y = (x_tile - mean[:, None]) * rstd[:, None] * w[None, :] + bias[None, :]
        tl.store(o_ptrs, y, mask=mask)


def online_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.ndim == 3
    assert weight.shape == (x.shape[-1],) and bias.shape == (x.shape[-1],)

    B, M, N = x.shape
    out = torch.empty_like(x)

    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))
    online_layernorm_kernel[grid](
        x,
        weight,
        bias,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        M,
        N,
        eps,
    )
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(42)
    B, M, N = 64, 256, 8192
    x = torch.randn(B, M, N, device="cuda", dtype=torch.float32)
    weight = torch.randn(N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)
    eps = 1e-5

    o_triton = online_layernorm(x, weight, bias, eps=eps)
    o_torch = torch.nn.functional.layer_norm(
        x, (N,), weight=weight, bias=bias, eps=eps
    )

    print(f"allclose: {torch.allclose(o_torch, o_triton, rtol=1e-4, atol=1e-4)}")
    print(f"max abs err: {(o_torch - o_triton).abs().max().item():.3e}")
    print(f"shape: {(B, M, N)}, BLOCK_N=256 → 约 {triton.cdiv(N, 256)} 块/行")

    ms_torch = triton.testing.do_bench(
        lambda: torch.nn.functional.layer_norm(
            x, (N,), weight=weight, bias=bias, eps=eps
        )
    )
    ms_online = triton.testing.do_bench(
        lambda: online_layernorm(x, weight, bias, eps=eps)
    )
    print(f"torch          : {ms_torch:.3f} ms")
    print(f"triton online  : {ms_online:.3f} ms")

    # 若同目录 naive 版可用，顺带对比
    try:
        from layernorm import layernorm as naive_layernorm

        ms_naive = triton.testing.do_bench(
            lambda: naive_layernorm(x, weight, bias, eps=eps)
        )
        print(f"triton naive   : {ms_naive:.3f} ms")
    except Exception as e:
        print(f"(skip naive compare: {e})")


if __name__ == "__main__":
    main()
