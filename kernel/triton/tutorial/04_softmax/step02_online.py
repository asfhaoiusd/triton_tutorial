"""Triton：Online Softmax（对最后一维，沿列分块）

维护可合并统计量 (m, d)：
    m = 目前见过的 max
    d = sum(exp(x - m))

新块 (m_blk, d_blk) 合并：
    m_new = max(m, m_blk)
    d_new = d * exp(m - m_new) + d_blk * exp(m_blk - m_new)

两趟扫描同一行：
    1) 沿列块更新 (m, d)
    2) 再沿列块写 y = exp(x - m) / d

适合 N 很大、不能一次装进 BLOCK 的情况。
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 128}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 8, "BLOCK_COLS": 256}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 16, "BLOCK_COLS": 256}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 32, "BLOCK_COLS": 256}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 32, "BLOCK_COLS": 512}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 64, "BLOCK_COLS": 128}, num_warps=8),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def online_softmax_kernel(
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
    row_pid = tl.program_id(axis = 0)
    offs_rows = row_pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = offs_rows < n_rows

    m_i = tl.full((BLOCK_ROWS,), -float("inf"), dtype=tl.float32)
    d_i = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    for start in range(0, n_cols, BLOCK_COLS):
        offs_cols = start + tl.arange(0, BLOCK_COLS)
        mask = row_mask[:, None] & (offs_cols[None, :] < n_cols)

        x_ptrs = x_ptr + offs_rows[:, None] * x_row_stride + offs_cols[None, :] * x_col_stride
        x_tile = tl.load(x_ptrs, mask = mask, other = -float("inf"))

        max = tl.max(x_tile, axis = 1)
        new_max = tl.maximum(max, m_i)
        num = tl.exp(x_tile - new_max[:, None])

        d_blk = tl.sum(num, axis = 1)
        d_i = d_i * tl.exp(m_i - new_max) + d_blk
        m_i = new_max
    
    for start in range(0, n_cols, BLOCK_COLS):
        offs_cols = start + tl.arange(0, BLOCK_COLS)
        mask = row_mask[:, None] & (offs_cols[None, :] < n_cols)

        x_ptrs = x_ptr + offs_rows[:, None] * x_row_stride + offs_cols[None, :] * x_col_stride
        o_ptrs = o_ptr + offs_rows[:, None] * o_row_stride + offs_cols[None, :] * o_col_stride

        x_tile = tl.load(x_ptrs, mask = mask, other = -float("inf"))
        y = tl.exp(x_tile - m_i[:, None]) / d_i[:, None]
        tl.store(o_ptrs, y, mask = mask)

def online_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.ndim in (2, 3)

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_cols = x_contig.shape[-1]
    n_rows = x_contig.numel() // n_cols

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_ROWS"]),)
    online_softmax_kernel[grid](
        x_contig,
        out,
        x_contig.stride(-2),
        x_contig.stride(-1),
        out.stride(-2),
        out.stride(-1),
        n_rows,
        n_cols,
    )

    return out

def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)

    # 用较长的 N，突出分块意义
    M, N = 256, 4096
    x2 = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y2_torch = torch.nn.functional.softmax(x2, dim=-1)
    y2_triton = online_softmax(x2)
    print(
        f"[2D] allclose: {torch.allclose(y2_torch, y2_triton, rtol=1e-4, atol=1e-4)}"
    )
    print(f"[2D] max abs err: {(y2_torch - y2_triton).abs().max().item():.3e}")
    print(f"[2D] shape: {(M, N)}, BLOCK_COLS=64 → 约 {triton.cdiv(N, 64)} 块/行")

    B, S, D = 4, 256, 8192
    x3 = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    y3_torch = torch.nn.functional.softmax(x3, dim=-1)
    y3_triton = online_softmax(x3)
    print(
        f"[3D] allclose: {torch.allclose(y3_torch, y3_triton, rtol=1e-4, atol=1e-4)}"
    )
    print(f"[3D] max abs err: {(y3_torch - y3_triton).abs().max().item():.3e}")
    print(f"[3D] shape: {(B, S, D)}")

    print(
        f"torch  : {triton.testing.do_bench(lambda: torch.nn.functional.softmax(x3, dim=-1)):.3f} ms"
    )
    print(f"triton : {triton.testing.do_bench(lambda: online_softmax(x3)):.3f} ms")


if __name__ == "__main__":
    main()


