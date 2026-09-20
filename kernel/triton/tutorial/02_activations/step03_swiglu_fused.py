"""Triton：融合版三维 SwiGLU（matmul + SiLU + mul 都在一个 kernel 里）

公式:
    out = SiLU(x @ W1) ⊙ (x @ W2)
        = (a * sigmoid(a)) * b
    其中 a = x @ W1, b = x @ W2

形状:
    x:  (B, M, K)
    W1: (K, N)   # 各 batch 共享
    W2: (K, N)
    out:(B, M, N)

每个 program 负责一块 out[b, offs_m, offs_n]：
  沿 K 循环：同一份 x tile 分别与 W1/W2 tile 做 tl.dot 累加，
  最后 silu(acc1)*acc2 写回。不物化中间 a、b。
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
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def swiglu_fused_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    o_ptr,
    B,
    M,
    N,
    K,
    stride_xb,
    stride_xm,
    stride_xk,
    stride_w1k,
    stride_w1n,
    stride_w2k,
    stride_w2n,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 当前 batch 的 x / out 基址；W1/W2 无 batch 维
    x_batch = x_ptr + pid_b * stride_xb
    o_batch = o_ptr + pid_b * stride_ob

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # x @ W1
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # x @ W2

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        x_ptrs = x_batch + (
            offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        )
        w1_ptrs = w1_ptr + (
            k_offs[:, None] * stride_w1k + offs_n[None, :] * stride_w1n
        )
        w2_ptrs = w2_ptr + (
            k_offs[:, None] * stride_w2k + offs_n[None, :] * stride_w2n
        )

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w1_tile = tl.load(w1_ptrs, mask=w_mask, other=0.0)
        w2_tile = tl.load(w2_ptrs, mask=w_mask, other=0.0)

        # 同一份 x_tile 复用两次，少读一次 x
        # ieee: 便于和 PyTorch FP32 对齐；追求速度可改 "tf32"
        acc1 += tl.dot(x_tile, w1_tile, input_precision="ieee")
        acc2 += tl.dot(x_tile, w2_tile, input_precision="ieee")

    out = acc1 * tl.sigmoid(acc1) * acc2

    o_ptrs = o_batch + (
        offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, out, mask=o_mask)


def swiglu_pro(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """融合 SwiGLU：kernel 内完成两次 matmul + SiLU + 逐元素乘。"""
    assert x.is_cuda and w1.is_cuda and w2.is_cuda
    assert x.ndim == 3 and w1.ndim == 2 and w2.ndim == 2
    assert x.shape[-1] == w1.shape[0] == w2.shape[0]
    assert w1.shape == w2.shape

    B, M, K = x.shape
    _, N = w1.shape
    out = torch.empty((B, M, N), device=x.device, dtype=x.dtype)
    args = (
        x,
        w1,
        w2,
        out,
        B,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w1.stride(0),
        w1.stride(1),
        w2.stride(0),
        w2.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    ov = tile_override()
    if ov:
        bm, bn, bk = ov["swiglu_block_m"], ov["swiglu_block_n"], ov["swiglu_block_k"]
        launch(
            swiglu_fused_kernel,
            (B, triton.cdiv(M, bm), triton.cdiv(N, bn)),
            *args,
            block_kwargs={
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "BLOCK_K": bk,
                "num_warps": ov["swiglu_num_warps"],
                "num_stages": ov.get("swiglu_num_stages", 2),
            },
        )
        return out

    grid = lambda meta: (
        B,
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    swiglu_fused_kernel[grid](*args)
    return out


def swiglu_torch(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x @ w1) * (x @ w2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def swiglu_act_backward_kernel(
    a_ptr,
    b_ptr,
    dy_ptr,
    da_ptr,
    db_ptr,
    B,
    M,
    N,
    stride_ab,
    stride_am,
    stride_an,
    stride_bb,
    stride_bm,
    stride_bn,
    stride_dyb,
    stride_dym,
    stride_dyn,
    stride_dab,
    stride_dam,
    stride_dan,
    stride_dbb,
    stride_dbm,
    stride_dbn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """y = silu(a)*b；silu'(a) = σ(a)*(1 + a*(1-σ(a)))；db = dy*silu(a)；da = dy*b*silu'(a)"""
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a_batch = a_ptr + pid_b * stride_ab
    b_batch = b_ptr + pid_b * stride_bb
    dy_batch = dy_ptr + pid_b * stride_dyb
    da_batch = da_ptr + pid_b * stride_dab
    db_batch = db_ptr + pid_b * stride_dbb
    a = tl.load(
        a_batch + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    b = tl.load(
        b_batch + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dy = tl.load(
        dy_batch + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sig = tl.sigmoid(a)
    silu = a * sig
    db = dy * silu
    dsilu = dy * b
    da = dsilu * sig * (1.0 + a * (1.0 - sig))
    tl.store(
        da_batch + offs_m[:, None] * stride_dam + offs_n[None, :] * stride_dan,
        da,
        mask=mask,
    )
    tl.store(
        db_batch + offs_m[:, None] * stride_dbm + offs_n[None, :] * stride_dbn,
        db,
        mask=mask,
    )


def swiglu_act_backward(a: torch.Tensor, b: torch.Tensor, dy: torch.Tensor):
    assert a.shape == b.shape == dy.shape and a.ndim == 3
    B, M, N = a.shape
    da = torch.empty_like(a, dtype=torch.float32)
    db = torch.empty_like(b, dtype=torch.float32)
    args = (
        a,
        b,
        dy,
        da,
        db,
        B,
        M,
        N,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        da.stride(0),
        da.stride(1),
        da.stride(2),
        db.stride(0),
        db.stride(1),
        db.stride(2),
    )
    ov = tile_override()
    if ov:
        bm, bn = ov["swiglu_bwd_block_m"], ov["swiglu_bwd_block_n"]
        launch(
            swiglu_act_backward_kernel,
            (B, triton.cdiv(M, bm), triton.cdiv(N, bn)),
            *args,
            block_kwargs={
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "num_warps": ov["swiglu_bwd_num_warps"],
                "num_stages": ov.get("swiglu_bwd_num_stages", 2),
            },
        )
        return da, db
    grid = lambda meta: (
        B,
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    swiglu_act_backward_kernel[grid](*args)
    return da, db


def swiglu_pro_backward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    dy: torch.Tensor,
):
    """重算 a=x@W1、b=x@W2；SiLU' 在 fp32，GEMM 跟输入 dtype（bf16 走 Tensor Core）。"""
    w1c = w1.to(dtype=x.dtype)
    w2c = w2.to(dtype=x.dtype)
    a = torch.matmul(x, w1c)
    b = torch.matmul(x, w2c)
    da, db = swiglu_act_backward(a.float(), b.float(), dy.float())
    da = da.to(dtype=x.dtype)
    db = db.to(dtype=x.dtype)
    dx = da @ w1c.transpose(0, 1) + db @ w2c.transpose(0, 1)
    dw1 = torch.einsum("bmk,bmn->kn", x, da)
    dw2 = torch.einsum("bmk,bmn->kn", x, db)
    return dx, dw1.to(dtype=w1.dtype), dw2.to(dtype=w2.dtype)


class TritonSwiGLU(torch.autograd.Function):
    """fwd 走融合 swiglu_pro；bwd 重算 a/b 后对 silu(x@W1)*(x@W2) 回传。"""

    @staticmethod
    def forward(ctx, x, w1, w2):
        orig = x.shape
        k = orig[-1]
        n = w1.shape[1]
        x3 = x.reshape(1, -1, k).contiguous()
        w1c = w1.contiguous()
        w2c = w2.contiguous()
        if w1c.dtype != x3.dtype:
            w1c = w1c.to(dtype=x3.dtype)
            w2c = w2c.to(dtype=x3.dtype)
        y = swiglu_pro(x3, w1c, w2c)
        ctx.save_for_backward(x3, w1.contiguous(), w2.contiguous())
        ctx.orig = orig
        ctx.y_shape = y.shape
        return y.reshape(*orig[:-1], n)

    @staticmethod
    def backward(ctx, dy):
        x3, w1, w2 = ctx.saved_tensors
        dy3 = dy.reshape(ctx.y_shape).contiguous()
        dx, dw1, dw2 = swiglu_pro_backward(x3, w1, w2, dy3)
        return dx.reshape(ctx.orig), dw1, dw2


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    B, M, K, N = 4, 128, 64, 256
    x = torch.randn(B, M, K, device="cuda", dtype=torch.float32)
    w1 = torch.randn(K, N, device="cuda", dtype=torch.float32)
    w2 = torch.randn(K, N, device="cuda", dtype=torch.float32)

    o_torch = swiglu_torch(x, w1, w2)
    o_triton = swiglu_pro(x, w1, w2)

    ok = torch.allclose(o_torch, o_triton, rtol=1e-3, atol=1e-3)
    print(f"allclose: {ok}")
    print(f"max abs err: {(o_torch - o_triton).abs().max().item():.3e}")
    print(f"shape: x{(B, M, K)} , W*{(K, N)} -> out{(B, M, N)}")
    print(f"torch          : {triton.testing.do_bench(lambda: swiglu_torch(x, w1, w2)):.3f} ms")
    print(f"triton fused   : {triton.testing.do_bench(lambda: swiglu_pro(x, w1, w2)):.3f} ms")

    xg = x.detach().requires_grad_(True)
    w1g = w1.detach().requires_grad_(True)
    w2g = w2.detach().requires_grad_(True)
    dy = torch.randn_like(o_torch)
    y_ref = swiglu_torch(xg, w1g, w2g)
    dx_r, dw1_r, dw2_r = torch.autograd.grad(y_ref, (xg, w1g, w2g), dy)
    xt = x.detach().requires_grad_(True)
    w1t = w1.detach().requires_grad_(True)
    w2t = w2.detach().requires_grad_(True)
    yt = TritonSwiGLU.apply(xt, w1t, w2t)
    dx_t, dw1_t, dw2_t = torch.autograd.grad(yt, (xt, w1t, w2t), dy)
    print(f"bwd dx  allclose: {torch.allclose(dx_t, dx_r, rtol=1e-3, atol=1e-3)}  "
          f"max abs {(dx_t - dx_r).abs().max().item():.3e}")
    print(f"bwd dw1 allclose: {torch.allclose(dw1_t, dw1_r, rtol=1e-3, atol=1e-3)}  "
          f"max abs {(dw1_t - dw1_r).abs().max().item():.3e}")
    print(f"bwd dw2 allclose: {torch.allclose(dw2_t, dw2_r, rtol=1e-3, atol=1e-3)}  "
          f"max abs {(dw2_t - dw2_r).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
