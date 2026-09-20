import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 8, "BLOCK_P": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_P": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_P": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_P": 16}, num_warps=8),
    ],
    key=["m", "n", "p"],
)
@triton.jit
def relu_kernel(
    x_ptr,
    o_ptr,
    m,
    n,
    p,
    stride_xm,
    stride_xn,
    stride_xp,
    stride_om,
    stride_on,
    stride_op,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    pid_p = tl.program_id(axis = 2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    offs_x = offs_m[:, None, None] * stride_xm + offs_n[None, :, None] * stride_xn + offs_p[None, None, :] * stride_xp
    offs_o = offs_m[:, None, None] * stride_om + offs_n[None, :, None] * stride_on + offs_p[None, None, :] * stride_op

    mask = (offs_m[:, None, None] < m) & (offs_n[None, :, None] < n) & (offs_p[None, None, :] < p)
    x = tl.load(x_ptr + offs_x, mask = mask, other = 0.0)

    tl.store(o_ptr + offs_o, tl.where(x > 0, x, 0.0), mask = mask)

def relu_fwd(x: torch.Tensor) -> torch.Tensor:
    o = torch.empty_like(x)
    m, n, p = x.shape
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
        triton.cdiv(p, meta["BLOCK_P"]),
    )
    relu_kernel[grid](
        x,
        o,
        m,
        n,
        p,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
    )
    return o

@triton.jit
def relu_backward_kernel(
    dy_ptr,
    x_ptr,
    dx_ptr,
    m,
    n,
    p,
    stride_xm,
    stride_xn,
    stride_xp,
    stride_om,
    stride_on,
    stride_op,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    pid_p = tl.program_id(axis = 2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    offs_dy = offs_m[:, None, None] * stride_om + offs_n[None, :, None] * stride_on + offs_p[None, None, :] * stride_op
    offs_dx = offs_m[:, None, None] * stride_xm + offs_n[None, :, None] * stride_xn + offs_p[None, None, :] * stride_xp
    offs_x = offs_m[:, None, None] * stride_xm + offs_n[None, :, None] * stride_xn + offs_p[None, None, :] * stride_xp
    
    mask = (offs_m[:, None, None] < m) & (offs_n[None, :, None] < n) & (offs_p[None, None, :] < p)
    dy = tl.load(dy_ptr + offs_dy, mask = mask, other = 0.0)
    dx = tl.load(dx_ptr + offs_dx, mask = mask, other = 0.0)
    x = tl.load(x_ptr + offs_x, mask = mask, other = 0.0)

    dx = tl.where(x > 0, dy, 0.0)
    tl.store(dx_ptr + offs_dx, dx, mask = mask)


def relu_backward(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dx = torch.empty_like(x)
    m, n, p = x.shape
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
        triton.cdiv(p, meta["BLOCK_P"]),
    )
    relu_backward_kernel[grid](
        dy,
        x,
        dx,
        m,
        n,
        p,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        BLOCK_M=8,
        BLOCK_N=8,
        BLOCK_P=8,
    )
    return dx


class TritonReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return relu_fwd(x)
    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        return relu_backward(dy, x)


def _check(x, dy, tag):
    y_ref = torch.nn.functional.relu(x)
    (dx_ref,) = torch.autograd.grad(y_ref, x, dy)

    x_t = x.detach().requires_grad_(True)
    y_t = TritonReLU.apply(x_t)
    (dx_t,) = torch.autograd.grad(y_t, x_t, dy)

    rtol, atol = 1e-4, 1e-4
    print(
        f"[{tag}] fwd allclose: {torch.allclose(y_t, y_ref, rtol=rtol, atol=atol)}  "
        f"max abs err: {(y_t - y_ref).abs().max().item():.3e}"
    )
    print(
        f"[{tag}] dx  allclose: {torch.allclose(dx_t, dx_ref, rtol=rtol, atol=atol)}  "
        f"max abs err: {(dx_t - dx_ref).abs().max().item():.3e}"
    )
    return x_t, y_t


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    M, N, P = 4, 2048, 2048
    x = torch.randn(M, N, P, device="cuda", dtype=torch.float32, requires_grad=True)
    dy = torch.randn(M, N, P, device="cuda", dtype=torch.float32)

    print(f"torch relu: {triton.testing.do_bench(lambda: torch.nn.functional.relu(x)):.3f} ms")
    print(f"triton relu: {triton.testing.do_bench(lambda: relu_fwd(x)):.3f} ms")

    x_t, y_t = _check(x, dy, "ReLU")

    y_torch = torch.nn.functional.relu(x)

    def torch_bwd():
        torch.autograd.grad(y_torch, x, dy, retain_graph=True)

    print(f"torch bwd: {triton.testing.do_bench(torch_bwd):.3f} ms")
    print(f"triton bwd: {triton.testing.do_bench(lambda: relu_backward(dy, x.detach())):.3f} ms")

if __name__ == "__main__":
    main()