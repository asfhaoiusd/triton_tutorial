"""Triton：LayerNorm（对最后一维）

y = gamma * (x - mean) / sqrt(var + eps) + beta

x: (B, M, N), gamma/beta: (N,)
每个 (b, m) 一行，在 N 上求 mean/var。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

_TUT = Path(__file__).resolve().parent.parent
if str(_TUT) not in sys.path:
    sys.path.insert(0, str(_TUT))
from kernel_tiles import launch, tiles as tile_override


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_warps=4),
        triton.Config({"BLOCK_M": 4}, num_warps=4),
        triton.Config({"BLOCK_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 32}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def layernorm_kernel(
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
    offs_n = tl.arange(0, BLOCK_N)

    x_batch = x_ptr + pid_b * stride_xb
    o_batch = o_ptr + pid_b * stride_ob

    x_ptrs = x_batch + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    o_ptrs = o_batch + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    w_ptrs = w_ptr + offs_n
    b_ptrs = b_ptr + offs_n

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    w_mask = offs_n < N

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    w = tl.load(w_ptrs, mask=w_mask, other=0.0)
    bias = tl.load(b_ptrs, mask=w_mask, other=0.0)

    # 行归约：axis=1
    mean = tl.sum(x, axis=1) / N
    x_center = x - mean[:, None]
    # 越界位置 other=0，不贡献 sum；有效长度按 N（与 F.layer_norm 一致用全长）
    var = tl.sum(x_center * x_center, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    y = x_center * rstd[:, None] * w[None, :] + bias[None, :]
    tl.store(o_ptrs, y, mask=mask)


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.ndim == 3
    assert weight.ndim == 1 and bias.ndim == 1
    assert x.shape[-1] == weight.numel() == bias.numel()

    B, M, N = x.shape
    out = torch.empty_like(x)

    # 归约维一次装整行；只 autotune BLOCK_M / num_warps
    BLOCK_N = triton.next_power_of_2(N)
    assert BLOCK_N <= 65536, f"N={N} 太大，需要分块 LayerNorm"
    ov = tile_override()
    if ov:
        bm = ov["ln_block_m"]
        grid = (B, triton.cdiv(M, bm))
        launch(
            layernorm_kernel,
            grid,
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
            block_kwargs={
                "BLOCK_M": bm,
                "BLOCK_N": BLOCK_N,
                "num_warps": ov["ln_num_warps"],
                "num_stages": ov.get("ln_num_stages", 1),
            },
        )
        return out

    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))
    layernorm_kernel[grid](
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
        BLOCK_N=BLOCK_N,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_warps=4),
        triton.Config({"BLOCK_M": 4}, num_warps=4),
        triton.Config({"BLOCK_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 32}, num_warps=8),
    ],
    key=["M", "N"],
    reset_to_zero=["dw_ptr", "db_ptr"],
)
@triton.jit
def layernorm_backward_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    dx_ptr,
    dw_ptr,
    db_ptr,
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
    """
    g = γ * dy
    dx = rstd * (g - mean(g) - xhat * mean(g * xhat))
    dγ 沿行累加 dy * xhat；dβ 沿行累加 dy。
    """
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x_batch = x_ptr + pid_b * stride_xb
    dy_batch = dy_ptr + pid_b * stride_dyb
    dx_batch = dx_ptr + pid_b * stride_dxb

    x_ptrs = x_batch + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    dy_ptrs = dy_batch + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn
    dx_ptrs = dx_batch + offs_m[:, None] * stride_dxm + offs_n[None, :] * stride_dxn
    w_ptrs = w_ptr + offs_n

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    w_mask = offs_n < N

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0)
    w = tl.load(w_ptrs, mask=w_mask, other=0.0)

    mean = tl.sum(x, axis=1) / N
    x_center = x - mean[:, None]
    var = tl.sum(x_center * x_center, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = x_center * rstd[:, None]

    g = w[None, :] * dy
    mean_g = tl.sum(g, axis=1) / N
    mean_gxhat = tl.sum(g * xhat, axis=1) / N
    dx = rstd[:, None] * (g - mean_g[:, None] - xhat * mean_gxhat[:, None])
    tl.store(dx_ptrs, dx, mask=mask)

    # 本 tile 若干行对 (N,) 的贡献；多个 program 写同一 dw/db
    dw_tile = tl.sum(tl.where(mask, dy * xhat, 0.0), axis=0)
    db_tile = tl.sum(tl.where(mask, dy, 0.0), axis=0)
    tl.atomic_add(dw_ptr + offs_n, dw_tile, mask=w_mask)
    tl.atomic_add(db_ptr + offs_n, db_tile, mask=w_mask)


def layernorm_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-5,
):
    """返回 dx, dweight, dbias。bias 不算 dx，不必传入。"""
    assert x.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.shape == dy.shape and x.ndim == 3
    assert weight.ndim == 1 and weight.numel() == x.shape[-1]

    B, M, N = x.shape
    dx = torch.empty_like(x)
    dweight = torch.zeros(N, device=x.device, dtype=torch.float32)
    dbias = torch.zeros(N, device=x.device, dtype=torch.float32)

    BLOCK_N = triton.next_power_of_2(N)
    assert BLOCK_N <= 65536, f"N={N} 太大，需要分块 LayerNorm backward"
    ov = tile_override()
    args = (
        x,
        weight,
        dy,
        dx,
        dweight,
        dbias,
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
    )
    if ov:
        bm = ov["ln_block_m"]
        launch(
            layernorm_backward_kernel,
            (B, triton.cdiv(M, bm)),
            *args,
            block_kwargs={
                "BLOCK_M": bm,
                "BLOCK_N": BLOCK_N,
                "num_warps": ov["ln_num_warps"],
                "num_stages": ov.get("ln_num_stages", 1),
            },
        )
        return dx, dweight.to(dtype=weight.dtype), dbias.to(dtype=weight.dtype)

    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))
    layernorm_backward_kernel[grid](*args, BLOCK_N=BLOCK_N)
    return dx, dweight.to(dtype=weight.dtype), dbias.to(dtype=weight.dtype)


class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        orig = x.shape
        n = orig[-1]
        x3 = x.reshape(1, -1, n).contiguous()
        y = layernorm(x3, weight, bias, eps=float(eps))
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
        dx, dweight, dbias = layernorm_backward(x3, weight, dy3, eps=ctx.eps)
        return dx.reshape(ctx.orig), dweight, dbias, None


def _check(x, weight, bias, dy, eps, tag):
    y_ref = torch.nn.functional.layer_norm(
        x, (x.shape[-1],), weight=weight, bias=bias, eps=eps
    )
    dx_ref, dw_ref, db_ref = torch.autograd.grad(y_ref, (x, weight, bias), dy)

    x_t = x.detach().requires_grad_(True)
    w_t = weight.detach().requires_grad_(True)
    b_t = bias.detach().requires_grad_(True)
    y_t = TritonLayerNorm.apply(x_t, w_t, b_t, eps)
    dx_t, dw_t, db_t = torch.autograd.grad(y_t, (x_t, w_t, b_t), dy)

    rtol, atol = 1e-4, 1e-4
    print(
        f"[{tag}] fwd allclose: {torch.allclose(y_t, y_ref, rtol=rtol, atol=atol)}  "
        f"max abs err: {(y_t - y_ref).abs().max().item():.3e}"
    )
    print(
        f"[{tag}] dx  allclose: {torch.allclose(dx_t, dx_ref, rtol=rtol, atol=atol)}  "
        f"max abs err: {(dx_t - dx_ref).abs().max().item():.3e}"
    )
    print(
        f"[{tag}] dw  allclose: {torch.allclose(dw_t, dw_ref, rtol=rtol, atol=atol)}  "
        f"max abs err: {(dw_t - dw_ref).abs().max().item():.3e}"
    )
    print(
        f"[{tag}] db  allclose: {torch.allclose(db_t, db_ref, rtol=rtol, atol=atol)}  "
        f"max abs err: {(db_t - db_ref).abs().max().item():.3e}"
    )
    print(f"[{tag}] shape: x{tuple(x.shape)}")
    return x_t, w_t, b_t, y_t


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(42)
    # 整行装 N，先用中等宽度对齐梯度
    B, M, N = 8, 256, 2048
    eps = 1e-5
    x = torch.randn(B, M, N, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
    dy = torch.randn(B, M, N, device="cuda", dtype=torch.float32)

    x_t, w_t, b_t, y_t = _check(x, weight, bias, dy, eps, "LN")

    print(
        f"torch  fwd: {triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)):.3f} ms"
    )
    print(
        f"triton fwd: {triton.testing.do_bench(lambda: TritonLayerNorm.apply(x_t, w_t, b_t, eps)):.3f} ms"
    )

    y_torch = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)

    def torch_bwd():
        # grads = torch.autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)
        torch.autograd.grad(y_torch, (x, weight, bias), dy, retain_graph=True)

    print(f"torch  bwd: {triton.testing.do_bench(torch_bwd):.3f} ms")
    print(
        f"triton bwd: {triton.testing.do_bench(lambda: layernorm_backward(x.detach(), weight.detach(), dy, eps)):.3f} ms"
    )


if __name__ == "__main__":
    main()
