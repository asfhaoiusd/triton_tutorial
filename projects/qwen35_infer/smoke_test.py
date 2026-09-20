"""不下载权重：核数值 + 玩具 greedy 换核后 token 一致。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kernels import flash_attention_gqa, rmsnorm, swiglu_pro
from generate import generate_toy
from toy_model import ToyLM


def _ok(name: str, cond: bool, extra: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond:
        raise SystemExit(1)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    x = torch.randn(2, 16, 64, device="cuda")
    w = torch.randn(64, device="cuda")
    y = rmsnorm(x, w)
    y_ref = F.rms_norm(x, (64,), weight=w, eps=1e-6)
    _ok("rmsnorm", torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4),
        f"max abs {(y - y_ref).abs().max().item():.3e}")

    x3 = torch.randn(2, 8, 64, device="cuda")
    w1 = torch.randn(64, 128, device="cuda")
    w2 = torch.randn(64, 128, device="cuda")
    s = swiglu_pro(x3, w1, w2)
    s_ref = F.silu(x3 @ w1) * (x3 @ w2)
    _ok("swiglu", torch.allclose(s, s_ref, rtol=1e-3, atol=1e-3),
        f"max abs {(s - s_ref).abs().max().item():.3e}")

    q = torch.randn(1, 4, 32, 16, device="cuda")
    k = torch.randn(1, 2, 32, 16, device="cuda")
    v = torch.randn(1, 2, 32, 16, device="cuda")
    o = flash_attention_gqa(q, k, v, q_start=0)
    o_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    _ok("gqa prefill", torch.allclose(o, o_ref, rtol=1e-2, atol=1e-2),
        f"max abs {(o - o_ref).abs().max().item():.3e}")

    q1 = torch.randn(1, 4, 1, 16, device="cuda")
    k_full = torch.randn(1, 2, 32, 16, device="cuda")
    v_full = torch.randn(1, 2, 32, 16, device="cuda")
    o1 = flash_attention_gqa(q1, k_full, v_full, q_start=31)
    g = 4 // 2
    scale = 16**-0.5
    attn = torch.matmul(q1.float(), k_full.repeat_interleave(g, dim=1).transpose(-2, -1).float()) * scale
    q_pos = torch.tensor([31], device="cuda")
    k_pos = torch.arange(32, device="cuda")
    attn = attn.masked_fill(k_pos[None, None, None, :] > q_pos[None, None, :, None], float("-inf"))
    o1_ref = torch.matmul(torch.softmax(attn, dim=-1), v_full.repeat_interleave(g, dim=1).float()).to(q1.dtype)
    _ok("gqa decode", torch.allclose(o1, o1_ref, rtol=1e-2, atol=1e-2),
        f"max abs {(o1 - o1_ref).abs().max().item():.3e}")

    torch.manual_seed(0)
    m0 = ToyLM().cuda()
    m0.set_kernels(False)
    ids = torch.randint(0, m0.cfg.vocab_size, (1, 8), device="cuda")
    t0 = generate_toy(m0, ids, 8)
    torch.manual_seed(0)
    m1 = ToyLM().cuda()
    m1.set_kernels(True)
    t1 = generate_toy(m1, ids, 8)
    _ok("toy greedy match", torch.equal(t0, t1), f"eager={t0[0].tolist()} triton={t1[0].tolist()}")

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        from patch import TritonRMSNorm

        hn = Qwen3_5RMSNorm(64).cuda()
        with torch.no_grad():
            hn.weight.uniform_(-0.05, 0.05)
        xh = torch.randn(2, 8, 64, device="cuda")
        y_hf = hn(xh)
        y_tr = TritonRMSNorm(hn)(xh)
        _ok(
            "qwen35 rmsnorm 1+w",
            torch.allclose(y_hf, y_tr, rtol=1e-4, atol=1e-4),
            f"max abs {(y_hf - y_tr).abs().max().item():.3e}",
        )
    except ImportError:
        print("SKIP  qwen35 rmsnorm 1+w  (pip install transformers)")

    print("smoke_test ok")


if __name__ == "__main__":
    main()
