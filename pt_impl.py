import math

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def pt_sdpa_cpu(q, k, v, attn_bias=None):
    assert q.device == torch.device("cpu")
    assert k.device == torch.device("cpu")
    assert v.device == torch.device("cpu")

    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)


def pt_padded(q, k, v, attn_bias=None):
    target_head_dim = 128
    origin_head_dim = q.shape[-1]
    origin_scale = 1 / math.sqrt(origin_head_dim)

    padding_amount = target_head_dim - origin_head_dim
    padding_value = 0

    q = F.pad(q, (0, padding_amount), "constant", padding_value)
    k = F.pad(k, (0, padding_amount), "constant", padding_value)
    v = F.pad(v, (0, padding_amount), "constant", padding_value)

    attn_output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_bias, scale=origin_scale
    )
    attn_output = attn_output[..., :origin_head_dim]
    return attn_output


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
