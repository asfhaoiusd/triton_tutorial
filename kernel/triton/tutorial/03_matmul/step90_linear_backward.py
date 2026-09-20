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
def Linear_forward_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    b,
    m,
    n,
    K,
    stride_xb,
    stride_xm,
    stride_xk,
    stride_wb,
    stride_wk,
    stride_wn,
    stride_ob,
    stride_om,
    stride_on,
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

    x_batch = x_ptr + pid_b * stride_xb
    w_batch = w_ptr + pid_b * stride_wb
    o_batch = o_ptr + pid_b * stride_ob

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for k in range(K, BLOCK_K):
        k_offs = k * BLOCK_K + offs_k

        x_ptrs = x_batch + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_batch + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn

        x_mask = (offs_m[:, None] < m) & (k_offs[None, :] < K)
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < n)

        x_tile = tl.load(x_ptrs, mask = x_mask, other = 0.0)
        w_tile = tl.load(w_ptrs, mask = w_mask, other = 0.0)
        
        acc += tl.dot(x_tile, w_tile)

    o_ptrs = o_batch + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    o_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(o_ptrs, acc, mask = o_mask)

def linear_forward(x, w):
    b, m, n = x.shape
    _, n, K = w.shape

    o = torch.empty((b, m, n), device = x.device, dtype = x.dtype)

    # BLOCK_* 由 @triton.autotune 填，不要在启动时再传一遍
    grid = lambda meta: (
        b,
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )
    Linear_forward_kernel[grid](
    x,
    w,
    o,
    b,
    m,
    n,
    K,
    x.stride(0),
    x.stride(1),
    x.stride(2),
    w.stride(0),
    w.stride(1),
    w.stride(2),
    o.stride(0),
    o.stride(1),
    o.stride(2),
)
    return o

@triton.jit
def linear_backward_kernel(
    dy_ptr,
    dx_ptr,
    dw_ptr,
    x_ptr,
    w_ptr,
    b,
    m,
    n,
    K,
    stride_dyb,
    stride_dym,
    stride_dyn,
    stride_dxb,
    stride_dxm,
    stride_dxn,
    stride_dwb,
    stride_dwn,
    stride_dwk,
    stride_xb,
    stride_xm,
    stride_xk,
    stride_wb,
    stride_wk,
    stride_wn,
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

    dy_batch = dy_ptr + pid_b * stride_dyb
    dx_batch = dx_ptr + pid_b * stride_dxb
    dw_batch = dw_ptr + pid_b * stride_dwb
    x_batch = x_ptr + pid_b * stride_xb
    w_batch = w_ptr + pid_b * stride_wb

    acc_dx = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    acc_dw = tl.zeros((BLOCK_N, BLOCK_K), dtype = tl.float32)

    for k in range(K, BLOCK_K):
        k_offs = offs_k * BLOCK_K + k

        x_ptrs = x_batch + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        dy_ptrs = dy_batch + offs_m[:, None] * stride_dym + k_offs[None, :] * stride_dyn
        w_ptrs = w_batch + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn

        dy_tile = tl.load(dy_ptrs, mask = (offs_m[:, None] < m) & (k_offs[None, :] < K), other = 0.0)
        x_tile = tl.load(x_ptrs, mask = (offs_m[:, None] < m) & (k_offs[None, :] < K), other = 0.0)
        w_tile = tl.load(w_ptrs, mask = (k_offs[:, None] < K) & (offs_n[None, :] < n), other = 0.0)

        acc_dx += tl.dot(dy_tile, tl.trans(w_tile))
        acc_dw += tl.dot(tl.trans(x_tile), dy_tile)

    dx_ptrs = dx_batch + offs_m[:, None] * stride_dxm + offs_n[None, :] * stride_dxn
    dx_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(dx_ptrs, acc_dx, mask = dx_mask)

    dw_ptrs = dw_batch + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk
    dw_mask = (offs_n[:, None] < n) & (offs_k[None, :] < K)
    tl.store(dw_ptrs, acc_dw, mask = dw_mask)

def linear_backward(dy, x, w):
    b, m, n = x.shape
    _, n, K = w.shape

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    dx = torch.empty_like(x)
    dw = torch.empty_like(w)

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N), triton.cdiv(K, BLOCK_K))

    linear_backward_kernel[grid](
    dy,
    dx,
    dw,
    x,
    w,
    b,
    m,
    n,
    K,
    dy.stride(0),
    dy.stride(1),
    dy.stride(2),
    dx.stride(0),
    dx.stride(1),
    dx.stride(2),
    dw.stride(0),
    dw.stride(1),
    dw.stride(2),
    x.stride(0),
    x.stride(1),
    x.stride(2),
    w.stride(0),
    w.stride(1),
    w.stride(2),    
    BLOCK_M = BLOCK_M,
    BLOCK_N = BLOCK_N,
    BLOCK_K = BLOCK_K,
)
    return dx, dw

class TritonLinear(torch.autograd.Function):
    def forward(ctx, x, w):
        o = linear_forward(x, w)
        ctx.save_for_backward(x, w)
        return o

    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dx, dw = linear_backward(dy, x, w)
        return dx, dw

def _check(x, w, dy, tag):
    y_ref = x @ w
    dx_ref, dw_ref = torch.autograd.grad(y_ref, (x, w), dy)

    x_t = x.detach().requires_grad_(True)
    w_t = w.detach().requires_grad_(True)
    y_t = TritonLinear.apply(x_t, w_t)
    dx_t, dw_t = torch.autograd.grad(y_t, (x_t, w_t), dy)

    # tl.dot 可能走 TF32，matmul 用稍宽的容差
    rtol, atol = 5e-2, 5e-2
    print(f"[{tag}] fwd allclose: {torch.allclose(y_t, y_ref, rtol=rtol, atol=atol)}  "
          f"max abs err: {(y_t - y_ref).abs().max().item():.3e}")
    print(f"[{tag}] dx  allclose: {torch.allclose(dx_t, dx_ref, rtol=rtol, atol=atol)}  "
          f"max abs err: {(dx_t - dx_ref).abs().max().item():.3e}")
    print(f"[{tag}] dw  allclose: {torch.allclose(dw_t, dw_ref, rtol=rtol, atol=atol)}  "
          f"max abs err: {(dw_t - dw_ref).abs().max().item():.3e}")
    print(f"[{tag}] shape: x{tuple(x.shape)} @ W{tuple(w.shape)}")
    return x_t, w_t, y_t, dy

def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    x = torch.randn(4, 1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(1, 1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    dy = torch.randn(4, 1024, 1024, device="cuda", dtype=torch.float32)
    x_t, w_t, y_t, dy = _check(x, w, dy, "Linear")

    print(f"torch  fwd: {triton.testing.do_bench(lambda: x @ w):.3f} ms")
    print(f"triton fd: {triton.testing.do_bench(lambda: TritonLinear.apply(x_t, w_t)):.3f} ms")

    y_torch = x @ w

    def torch_bwd():
        dx, dw = torch.autograd.grad(y_torch, (x, w), dy, retain_graph=True)
        return dx

    print(f"torch  bwd: {triton.testing.do_bench(torch_bwd):.3f} ms")
    print(
        f"triton bwd: {triton.testing.do_bench(lambda: linear_backward(dy, x.detach(), w.detach())):.3f} ms"
    )

if __name__ == "__main__":
    main()