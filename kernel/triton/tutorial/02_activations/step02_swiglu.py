"""Triton：三维 SwiGLU

公式:
    SwiGLU(x; W1, W2) = SiLU(x @ W1) ⊙ (x @ W2)
    SiLU(z) = z * sigmoid(z)

这里把「激活 + 逐元素乘」融成一个 3D kernel：
    out = silu(a) * b
线性投影 x@W1 / x@W2 在 Host 侧完成（可用 torch 或你的 matmul kernel）。

张量形状:
    x:  (B, M, K)
    W1: (K, N)
    W2: (K, N)
    a,b,out: (B, M, N)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 1, "BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_B": 1, "BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_B": 1, "BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_B": 1, "BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
    ],
    key=["B", "M", "N"],
)
@triton.jit
def swiglu_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    B,
    M,
    N,
    stride_ab,
    stride_am,
    stride_an,
    stride_bb,
    stride_bm,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_a = (
        offs_b[:, None, None] * stride_ab
        + offs_m[None, :, None] * stride_am
        + offs_n[None, None, :] * stride_an
    )
    offs_b_ = (
        offs_b[:, None, None] * stride_bb
        + offs_m[None, :, None] * stride_bm
        + offs_n[None, None, :] * stride_bn
    )
    offs_o = (
        offs_b[:, None, None] * stride_ob
        + offs_m[None, :, None] * stride_om
        + offs_n[None, None, :] * stride_on
    )
    mask = (
        (offs_b[:, None, None] < B)
        & (offs_m[None, :, None] < M)
        & (offs_n[None, None, :] < N)
    )

    a = tl.load(a_ptr + offs_a, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_b_, mask=mask, other=0.0)

    # SiLU(a) * b
    out = a * tl.sigmoid(a) * b
    tl.store(o_ptr + offs_o, out, mask=mask)


def swiglu_act(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """融合激活: silu(a) * b，a/b 均为 (B, M, N)。"""
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape and a.ndim == 3
    out = torch.empty_like(a)
    B, M, N = a.shape
    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_B"]),
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    swiglu_kernel[grid](
        a,
        b,
        out,
        B,
        M,
        N,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


def swiglu(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """完整 SwiGLU: silu(x @ w1) * (x @ w2)。"""
    assert x.ndim == 3 and w1.ndim == 2 and w2.ndim == 2
    assert x.shape[-1] == w1.shape[0] == w2.shape[0]
    assert w1.shape[1] == w2.shape[1]
    a = torch.matmul(x, w1)  # (B, M, N)
    b = torch.matmul(x, w2)
    return swiglu_act(a, b)


def swiglu_torch(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x @ w1) * (x @ w2)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    B, M, K, N = 4, 128, 64, 256
    x = torch.randn(B, M, K, device="cuda", dtype=torch.float32)
    w1 = torch.randn(K, N, device="cuda", dtype=torch.float32)
    w2 = torch.randn(K, N, device="cuda", dtype=torch.float32)

    o_torch = swiglu_torch(x, w1, w2)
    o_triton = swiglu(x, w1, w2)

    ok = torch.allclose(o_torch, o_triton, rtol=1e-4, atol=1e-4)
    print(f"allclose: {ok}")
    print(f"max abs err: {(o_torch - o_triton).abs().max().item():.3e}")
    print(f"shape: x{(B, M, K)} , W*{(K, N)} -> out{(B, M, N)}")
    print(f"torch : {triton.testing.do_bench(lambda: swiglu_torch(x, w1, w2)):.3f} ms")
    print(f"triton: {triton.testing.do_bench(lambda: swiglu(x, w1, w2)):.3f} ms")
    # 只比融合激活部分
    a = x @ w1
    b = x @ w2
    print(
        f"act torch : {triton.testing.do_bench(lambda: torch.nn.functional.silu(a) * b):.3f} ms"
    )
    print(f"act triton: {triton.testing.do_bench(lambda: swiglu_act(a, b)):.3f} ms")


if __name__ == "__main__":
    main()
