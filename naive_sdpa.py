# Copied from: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

from xformers.ops import memory_efficient_attention


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


def naive_xformers(q, k, v, p=0.0, attn_bias=None):
    """
    q (B, M, H, K)
    k (B, M, H, K)
    k (B, M, H, K)
    attn_bias(optional) (B, H, M, M)
    output (B, M, H, K)
    """
    
    scale = 1.0 / q.shape[-1] ** 0.5
    q = q * scale

    # (B M H K) -> (B H M K)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn = torch.matmul(q, k.transpose(-2, -1))
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p)
    attn = torch.matmul(attn, v)

    return attn.transpose(1, 2) # (B H M K) -> (B M H K)


def sdpa(query, key, value) -> torch.Tensor:
    # L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) 
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


def custom_check(output_1, output_2):
    TOL = 1e-6
    assert output_1.shape == output_2.shape
    return torch.max(torch.abs(output_1 - output_2)).item() <= TOL


if __name__ == "__main__":
    B, M, H, K = 4, 32, 16, 128
    temperature = 1.0 / K ** 0.5
    generator = torch.Generator(device='cuda')
    input_kwargs = dict(device="cuda", dtype=torch.float16, generator=generator)
    q = torch.randn((B, M, H, K), **input_kwargs)
    k = torch.randn((B, M, H, K), **input_kwargs)
    v = torch.randn((B, M, H, K), **input_kwargs)

    o = F.scaled_dot_product_attention(q, k, v)
    o1 = naive_xformers(q, k, v)
    o2, _ = ScaledDotProductAttention(temperature, 0.0).forward(q, k, v)
    o3 = memory_efficient_attention(q, k, v)
    o4 = sdpa(q, k, v)
    # print(f"{o.shape=}, {o1.shape=}, {o2.shape=}")

    # monkey patch
    # torch.allclose = custom_check

    print(torch.allclose(o, o1))
    try: 
        torch.testing.assert_close(o, o1)
    except AssertionError as e:
        print(e)


    print(torch.allclose(o, o2))
    try: 
        torch.testing.assert_close(o, o2)
    except AssertionError as e:
        print(e)

    print(torch.allclose(o1, o2)) # False
    try: 
        torch.testing.assert_close(o1, o2)
    except AssertionError as e:
        print(e)

    print(torch.allclose(o1, o3))
    # torch.testing.assert_close(o1, o3)
    print(custom_check(o1, o3))

    print(torch.allclose(o, o4))
    print(custom_check(o, o4))

