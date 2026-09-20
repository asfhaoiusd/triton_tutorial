"""Triton：分块 RMSNorm（对最后一维）

rms = sqrt(mean(x^2) + eps) = sqrt(sum(x^2)/N + eps)
y   = weight * x / rms

x: (B, M, N), weight: (N,), eps: 标量
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
def online_rmsnorm_kernel(
    x_ptr,
    w_ptr,
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

    sum_x2 = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Pass1: 累加 sum(x^2)
    for start in range(0, N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (offs_n[None, :] < N)
        x_ptrs = (
            x_batch + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        )
        x_tile = tl.load(x_ptrs, mask=mask, other=0.0)
        sum_x2 += tl.sum(x_tile * x_tile, axis=1)

    # rms = sqrt(mean(x^2) + eps)
    rstd = 1.0 / tl.sqrt(sum_x2 / N + eps)

    # Pass2: 写回
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
        x_tile = tl.load(x_ptrs, mask=mask, other=0.0)

        y = x_tile * rstd[:, None] * w[None, :]
        tl.store(o_ptrs, y, mask=mask)


def online_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    assert x.ndim == 3
    assert weight.ndim == 1 and weight.numel() == x.shape[-1]

    B, M, N = x.shape
    out = torch.empty_like(x)

    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))

    online_rmsnorm_kernel[grid](
        x,
        weight,
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


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """推理用 RMSNorm（无图）。x: (..., N)，weight: (N,)。"""
    orig_shape = x.shape
    n = orig_shape[-1]
    x3 = x.reshape(1, -1, n).contiguous()
    return online_rmsnorm(x3, weight, eps=eps).reshape(orig_shape)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_warps=4),
        triton.Config({"BLOCK_M": 4}, num_warps=4),
        triton.Config({"BLOCK_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 16}, num_warps=8),
    ],
    key=["M", "N"],
    reset_to_zero=["dw_ptr"],
)
@triton.jit
def rmsnorm_backward_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    dx_ptr,
    dw_ptr,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_dyb,
    stride_dym,
    stride_dyn,
    stride_dxb,
    stride_dxm,
    stride_dxn,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """xhat = x * rstd; g = w * dy; dx = rstd * (g - xhat * mean(xhat * g)); dw = Σ dy * xhat"""
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_batch = x_ptr + pid_b * stride_xb
    dy_batch = dy_ptr + pid_b * stride_dyb
    dx_batch = dx_ptr + pid_b * stride_dxb
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    w_mask = offs_n < N
    x = tl.load(
        x_batch + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    )
    dy = tl.load(
        dy_batch + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn,
        mask=mask,
        other=0.0,
    )
    w = tl.load(w_ptr + offs_n, mask=w_mask, other=0.0)
    var = tl.sum(x * x, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = x * rstd[:, None]
    g = w[None, :] * dy
    mean_gxhat = tl.sum(g * xhat, axis=1) / N
    dx = rstd[:, None] * (g - xhat * mean_gxhat[:, None])
    tl.store(
        dx_batch + offs_m[:, None] * stride_dxm + offs_n[None, :] * stride_dxn,
        dx,
        mask=mask,
    )
    dw_tile = tl.sum(tl.where(mask, dy * xhat, 0.0), axis=0)
    tl.atomic_add(dw_ptr + offs_n, dw_tile, mask=w_mask)


def rmsnorm_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
):
    assert x.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.shape == dy.shape and x.ndim == 3
    B, M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.zeros(N, device=x.device, dtype=torch.float32)
    BLOCK_N = triton.next_power_of_2(N)
    assert BLOCK_N <= 8192, f"N={N} 太大，RMSNorm bwd 教学核整行装不下"
    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))
    rmsnorm_backward_kernel[grid](
        x,
        weight,
        dy,
        dx,
        dw,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        dx.stride(0),
        dx.stride(1),
        dx.stride(2),
        M,
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )
    return dx, dw.to(dtype=weight.dtype)


class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig = x.shape
        n = orig[-1]
        x3 = x.reshape(1, -1, n).contiguous()
        y = online_rmsnorm(x3, weight, float(eps))
        ctx.save_for_backward(x3, weight)
        ctx.eps = float(eps)
        ctx.orig = orig
        return y.reshape(orig)

    @staticmethod
    def backward(ctx, dy):
        x3, weight = ctx.saved_tensors
        dy3 = dy.reshape(x3.shape).contiguous()
        if dy3.dtype != x3.dtype:
            dy3 = dy3.to(dtype=x3.dtype)
        dx, dw = rmsnorm_backward(x3, weight, dy3, eps=ctx.eps)
        return dx.reshape(ctx.orig), dw, None


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(42)
    B, M, N = 64, 256, 8192
    x = torch.randn(B, M, N, device="cuda", dtype=torch.float32)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    eps = 1e-6

    o_triton = online_rmsnorm(x, w, eps=eps)
    o_torch = torch.nn.functional.rms_norm(x, (N,), weight=w, eps=eps)

    print(f"allclose: {torch.allclose(o_torch, o_triton, rtol=1e-4, atol=1e-4)}")
    print(f"max abs err: {(o_torch - o_triton).abs().max().item():.3e}")
    print(f"shape: {(B, M, N)}")
    print(
        f"torch  rmsnorm: {triton.testing.do_bench(lambda: torch.nn.functional.rms_norm(x, (N,), weight=w, eps=eps)):.3f} ms"
    )
    print(
        f"triton rmsnorm: {triton.testing.do_bench(lambda: online_rmsnorm(x, w, eps=eps)):.3f} ms"
    )
    x2 = torch.randn(32, N, device="cuda", dtype=torch.float32)
    y2 = rmsnorm(x2, w, eps=eps)
    y2_ref = torch.nn.functional.rms_norm(x2, (N,), weight=w, eps=eps)
    print(f"rmsnorm 2d allclose: {torch.allclose(y2, y2_ref, rtol=1e-4, atol=1e-4)}")

    # 训练路径：Function + autograd.grad（整行 N 不宜过大）
    torch.manual_seed(0)
    xb = torch.randn(4, 64, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    wb = torch.randn(512, device="cuda", dtype=torch.float32, requires_grad=True)
    dy = torch.randn_like(xb)
    y_ref = torch.nn.functional.rms_norm(xb, (512,), weight=wb, eps=eps)
    dx_ref, dw_ref = torch.autograd.grad(y_ref, (xb, wb), dy)
    xt = xb.detach().requires_grad_(True)
    wt = wb.detach().requires_grad_(True)
    yt = TritonRMSNorm.apply(xt, wt, eps)
    dx_t, dw_t = torch.autograd.grad(yt, (xt, wt), dy)
    print(f"bwd dx allclose: {torch.allclose(dx_t, dx_ref, rtol=1e-3, atol=1e-3)}  "
          f"max abs {(dx_t - dx_ref).abs().max().item():.3e}")
    print(f"bwd dw allclose: {torch.allclose(dw_t, dw_ref, rtol=1e-3, atol=1e-3)}  "
          f"max abs {(dw_t - dw_ref).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
