import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def pt_sdpa_cpu(q, k, v, attn_bias=None):
    assert q.device == torch.device("cpu")
    assert k.device == torch.device("cpu")
    assert v.device == torch.device("cpu")

    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)


def pt_flash(q, k, v, attn_bias=None):
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)


def pt_xformers(q, k, v, attn_bias=None):
    with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)


def pt_math(q, k, v, attn_bias=None):
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)


if __name__ == "__main__":
    b, s, h, d = 2, 24, 4429, 64
    q = torch.randn((b, h, s, d), device="cpu", dtype=torch.float16)
    k = torch.randn((b, h, s, d), device="cpu", dtype=torch.float16)
    v = torch.randn((b, h, s, d), device="cpu", dtype=torch.float16)
    out = F.scaled_dot_product_attention(q, k, v)
