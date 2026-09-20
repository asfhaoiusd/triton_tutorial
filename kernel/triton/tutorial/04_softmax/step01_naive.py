"""Triton：行 Softmax（对最后一维）

公式（数值稳定）:
    m = max(x)
    y = exp(x - m)
    out = y / sum(y)

形状:
    支持 2D (M, N) 或 3D (B, M, N)
    展成 n_rows 行、每行 n_cols 列。

策略:
    - 一个 program 处理 BLOCK_ROWS 行
    - BLOCK_COLS 覆盖整行（不切归约维）
    - 沿 axis=1 做 max / sum
"""

import torch
import triton
import triton.language as tl


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
def naive_softmax_kernel(
    x_ptr,
    o_ptr,
    x_row_stride,
    x_col_stride,
    o_row_stride,
    o_col_stride,
    n_rows,
    n_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    # 只沿「行」分 program；列维一次装整行
    row_pid = tl.program_id(axis=0)

    offs_rows = row_pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    offs_cols = tl.arange(0, BLOCK_COLS)

    x_ptrs = (
        x_ptr
        + offs_rows[:, None] * x_row_stride
        + offs_cols[None, :] * x_col_stride
    )
    o_ptrs = (
        o_ptr
        + offs_rows[:, None] * o_row_stride
        + offs_cols[None, :] * o_col_stride
    )

    mask = (offs_rows[:, None] < n_rows) & (offs_cols[None, :] < n_cols)

    # 越界用 -inf，不影响行内 max
    x_tile = tl.load(x_ptrs, mask=mask, other=-float("inf"))

    # 每行一个 max / sum  → axis=1
    x_max = tl.max(x_tile, axis=1)
    num = tl.exp(x_tile - x_max[:, None])
    num = tl.where(mask, num, 0.0)
    den = tl.sum(num, axis=1)
    y = num / den[:, None]

    tl.store(o_ptrs, y, mask=mask)


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """对最后一维做 Softmax。"""
    assert x.is_cuda
    assert x.ndim in (2, 3), "入门版支持 2D/3D"

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_cols = x_contig.shape[-1]
    n_rows = x_contig.numel() // n_cols

    # 连续布局下：下一行跨 n_cols，下一列跨 1
    x_row_stride = x_contig.stride(-2)
    x_col_stride = x_contig.stride(-1)
    o_row_stride = out.stride(-2)
    o_col_stride = out.stride(-1)

    # 归约维必须一次装整行；只 autotune BLOCK_ROWS / num_warps
    BLOCK_COLS = triton.next_power_of_2(n_cols)
    assert BLOCK_COLS <= 65536, f"N={n_cols} 太大，需要分块 Softmax"

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_ROWS"]),)
    naive_softmax_kernel[grid](
        x_contig,
        out,
        x_row_stride,
        x_col_stride,
        o_row_stride,
        o_col_stride,
        n_rows,
        n_cols,
        BLOCK_COLS=BLOCK_COLS,
    )
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(42)

    # 2D
    M, N = 512, 512
    x2 = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y2_torch = torch.nn.functional.softmax(x2, dim=-1)
    y2_triton = naive_softmax(x2)
    print(
        f"[2D] allclose: {torch.allclose(y2_torch, y2_triton, rtol=1e-4, atol=1e-4)}"
    )
    print(f"[2D] max abs err: {(y2_torch - y2_triton).abs().max().item():.3e}")
    print(f"[2D] shape: {(M, N)}")
    print(f"[2D] torch softmax: {triton.testing.do_bench(lambda: torch.nn.functional.softmax(x2, dim=-1)):.3f} ms")
    print(f"[2D] triton softmax: {triton.testing.do_bench(lambda: naive_softmax(x2)):.3f} ms")

    # 3D: (batch, sequence, dim)·
    B, S, D = 4, 256, 256
    x3 = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    y3_torch = torch.nn.functional.softmax(x3, dim=-1)
    y3_triton = naive_softmax(x3)
    print(
        f"[3D] allclose: {torch.allclose(y3_torch, y3_triton, rtol=1e-4, atol=1e-4)}"
    )
    print(f"[3D] max abs err: {(y3_torch - y3_triton).abs().max().item():.3e}")
    print(f"[3D] shape: {(B, S, D)}")

    print(
        f"torch  softmax: {triton.testing.do_bench(lambda: torch.nn.functional.softmax(x3, dim=-1)):.3f} ms"
    )
    print(f"triton softmax: {triton.testing.do_bench(lambda: naive_softmax(x3)):.3f} ms")


if __name__ == "__main__":
    main()
