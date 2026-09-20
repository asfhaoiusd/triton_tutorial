import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["m", "n", "K"],
)
@triton.jit
def matmul_kernel_three_dimension(
    x_ptr,
    y_ptr,
    o_ptr,
    b,
    m,
    n,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(axis = 0)
    pid_m = tl.program_id(axis = 1)
    pid_n = tl.program_id(axis = 2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_batch = x_ptr + pid_b * stride_ab
    b_batch = y_ptr + pid_b * stride_bb
    c_batch = o_ptr + pid_b * stride_cb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        a_ptrs = a_batch + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_batch + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < m) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < n)

        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a_tile, b_tile)

    c_ptrs = c_batch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptrs, acc, mask = c_mask)


def matrix_multiplication_three_dimension(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.ndim == 3 and y.ndim == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == y.shape[1]

    b, m, k = x.shape
    _, k2, n = y.shape

    o = torch.empty((b, m, n),device = x.device, dtype = x.dtype)
    grid = lambda meta: (
        b,
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )
    matmul_kernel_three_dimension[grid](
        x,
        y,
        o,
        b,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
    )
    return o

def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    b, m, k, n = 16, 512, 64, 256
    x = torch.randn(b, m, k, device = "cuda", dtype = torch.float32)
    y = torch.randn(b, k, n, device = "cuda", dtype = torch.float32)

    o_naive = x @ y
    o_triton = matrix_multiplication_three_dimension(x, y)
    o_torch = torch.matmul(x, y)
    print(f"allclose: {torch.allclose(o_naive, o_triton, rtol = 5e-2, atol = 5e-2)}")
    print(f"max abs err: {(o_naive - o_triton).abs().max().item():.3e}")
    print(f"shape: A{(b, m, k)} @ B{(b, k, n)} -> C{(b, m, n)}")
    print(f"naive : {triton.testing.do_bench(lambda: x @ y):.3f} ms")
    print(f"torch: {triton.testing.do_bench(lambda: torch.matmul(x, y)):.3f} ms")
    print(f"triton: {triton.testing.do_bench(lambda: matrix_multiplication_three_dimension(x, y)):.3f} ms")


if __name__ == "__main__":
    main()