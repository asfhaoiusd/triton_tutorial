"""Triton：二维矩阵乘 C = A @ B
A: (M, K), B: (K, N), C: (M, N)
"""

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
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿 K 维分块累加
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
        b_ptrs = b_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matrix_multiplication(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0], "需要 A:(M,K) 与 B:(K,N)"

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    M, K, N = 256, 256, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    o_naive = a @ b
    o_triton = matrix_multiplication(a, b)
    o_torch = torch.matmul(a, b)
    # tl.dot 在部分 GPU 上可能走 TF32，误差会比纯 FP32 略大
    # allclose 只能两两比较
    ok_torch = torch.allclose(o_torch, o_triton, rtol=5e-2, atol=5e-2)
    ok_naive = torch.allclose(o_naive, o_triton, rtol=5e-2, atol=5e-2)
    print(f"allclose torch/triton: {ok_torch}")
    print(f"allclose naive/triton: {ok_naive}")
    print(f"max abs err: {(o_torch - o_triton).abs().max().item():.3e}")
    print(f"shape: A{(M, K)} @ B{(K, N)} -> C{(M, N)}")
    print(f"naive : {triton.testing.do_bench(lambda: a @ b):.3f} ms")
    print(f"triton: {triton.testing.do_bench(lambda: matrix_multiplication(a, b)):.3f} ms")
    print(f"torch: {triton.testing.do_bench(lambda: torch.matmul(a, b)):.3f} ms")


if __name__ == "__main__":
    main()
