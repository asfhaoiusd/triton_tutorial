"""Triton：行 Softmax 反向（对最后一维）

正向: y = softmax(x)
反向: dx = y * (dy - sum(y * dy))     # sum 沿最后一维，再广播

形状 / 策略与 naive_softmax 相同:
    - 支持 2D (M, N) 或 3D (B, M, N)，展成 n_rows × n_cols
    - 一个 program 处理 BLOCK_ROWS 行
    - BLOCK_COLS 覆盖整行（不切归约维）
    - 沿 axis=1 做 sum

反向用的是 y，不是 x。TritonSoftmax.forward 里 save_for_backward(y)。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from step01_naive import naive_softmax


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": 1}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 4}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 8}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 16}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 32}, num_warps=8),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
# def naive_softmax_backward_kernel(
#     y_ptr,
#     dy_ptr,
#     dx_ptr,
#     y_row_stride,
#     y_col_stride,
#     dy_row_stride,
#     dy_col_stride,
#     dx_row_stride,
#     dx_col_stride,
#     n_rows,
#     n_cols,
#     BLOCK_ROWS: tl.constexpr,
#     BLOCK_COLS: tl.constexpr,
# ):
#     row_pid = tl.program_id(axis=0)

#     offs_rows = row_pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
#     offs_cols = tl.arange(0, BLOCK_COLS)

#     y_ptrs = (
#         y_ptr
#         + offs_rows[:, None] * y_row_stride
#         + offs_cols[None, :] * y_col_stride
#     )
#     dy_ptrs = (
#         dy_ptr
#         + offs_rows[:, None] * dy_row_stride
#         + offs_cols[None, :] * dy_col_stride
#     )
#     dx_ptrs = (
#         dx_ptr
#         + offs_rows[:, None] * dx_row_stride
#         + offs_cols[None, :] * dx_col_stride
#     )

#     mask = (offs_rows[:, None] < n_rows) & (offs_cols[None, :] < n_cols)

#     # 越界填 0，不贡献 sum(y * dy)
#     y = tl.load(y_ptrs, mask=mask, other=0.0)
#     dy = tl.load(dy_ptrs, mask=mask, other=0.0)

#     # s 每行一个标量：Σ_i y_i dy_i
#     s = tl.sum(y * dy, axis=1)
#     dx = y * (dy - s[:, None])

#     tl.store(dx_ptrs, dx, mask=mask)

def naive_softmax_backward_kernel(
    y_ptr,
    dy_ptr,
    dx_ptr,
    y_row_stride,
    y_col_stride,
    dy_row_stride,
    dy_col_stride,
    dx_row_stride,
    dx_col_stride,
    n_rows,
    n_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_pid = tl.program_id(axis = 0)

    offs_rows = row_pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    offs_cols = tl.arange(0, BLOCK_COLS)

    y_ptrs = y_ptr + offs_rows[:, None] * y_row_stride + offs_cols[None, :] * y_col_stride
    dy_ptrs = dy_ptr + offs_rows[:, None] * dy_row_stride + offs_cols[None, :] * dy_col_stride
    dx_ptrs = dx_ptr + offs_rows[:, None] * dx_row_stride + offs_cols[None, :] * dx_col_stride

    mask = (offs_rows[:, None] < n_rows) & (offs_cols[None, :] < n_cols)

    y = tl.load(y_ptrs, mask = mask, other = 0.0)
    dy = tl.load(dy_ptrs, mask = mask, other = 0.0)

    dx = y * (dy - tl.sum(y * dy, axis = 1)[:, None])
    tl.store(dx_ptrs, dx, mask = mask)

def naive_softmax_backward(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """给定 softmax 输出 y 和上游梯度 dy，计算 dx。"""
    assert y.is_cuda and dy.is_cuda
    assert y.shape == dy.shape
    assert y.ndim in (2, 3), "入门版支持 2D/3D"

    y_contig = y.contiguous()
    dy_contig = dy.contiguous()
    dx = torch.empty_like(y_contig)

    n_cols = y_contig.shape[-1]
    n_rows = y_contig.numel() // n_cols

    BLOCK_COLS = triton.next_power_of_2(n_cols)
    assert BLOCK_COLS <= 65536, f"N={n_cols} 太大，需要分块 Softmax backward"

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_ROWS"]),)
    naive_softmax_backward_kernel[grid](
        y_contig,
        dy_contig,
        dx,
        y_contig.stride(-2),
        y_contig.stride(-1),
        dy_contig.stride(-2),
        dy_contig.stride(-1),
        dx.stride(-2),
        dx.stride(-1),
        n_rows,
        n_cols,
        BLOCK_COLS=BLOCK_COLS,
    )
    return dx


class TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = naive_softmax(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        dx = naive_softmax_backward(y, dy)
        return dx


def _check(x: torch.Tensor, dy: torch.Tensor, tag: str):
    y_ref = torch.nn.functional.softmax(x, dim=-1)
    (dx_ref,) = torch.autograd.grad(y_ref, x, dy)

    x_t = x.detach().requires_grad_(True)
    y_t = TritonSoftmax.apply(x_t)
    (dx_t,) = torch.autograd.grad(y_t, x_t, dy)

    fwd_ok = torch.allclose(y_t, y_ref, rtol=1e-4, atol=1e-4)
    bwd_ok = torch.allclose(dx_t, dx_ref, rtol=1e-4, atol=1e-4)
    print(f"[{tag}] fwd allclose: {fwd_ok}  max abs err: {(y_t - y_ref).abs().max().item():.3e}")
    print(f"[{tag}] bwd allclose: {bwd_ok}  max abs err: {(dx_t - dx_ref).abs().max().item():.3e}")
    print(f"[{tag}] shape: {tuple(x.shape)}")
    return x_t, y_t, dy, dx_t


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(42)

    M, N = 512, 512
    x2 = torch.randn(M, N, device="cuda", dtype=torch.float32, requires_grad=True)
    dy2 = torch.randn(M, N, device="cuda", dtype=torch.float32)
    x2_t, y2_t, dy2, _ = _check(x2, dy2, "2D")

    B, S, D = 4, 512, 512
    x3 = torch.randn(B, S, D, device="cuda", dtype=torch.float32, requires_grad=True)
    dy3 = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    x3_t, y3_t, dy3, _ = _check(x3, dy3, "3D")

    print(
        f"torch  fwd: {triton.testing.do_bench(lambda: torch.nn.functional.softmax(x3, dim=-1)):.3f} ms"
    )
    print(f"triton fwd: {triton.testing.do_bench(lambda: TritonSoftmax.apply(x3_t)):.3f} ms")

    y_torch = torch.nn.functional.softmax(x3, dim=-1)
    y_det = y3_t.detach()

    def torch_bwd():
        (dx,) = torch.autograd.grad(y_torch, x3, dy3, retain_graph=True)
        return dx

    print(f"torch  bwd: {triton.testing.do_bench(torch_bwd):.3f} ms")
    print(
        f"triton bwd: {triton.testing.do_bench(lambda: naive_softmax_backward(y_det, dy3)):.3f} ms"
    )


if __name__ == "__main__":
    main()
