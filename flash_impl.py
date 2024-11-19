from functools import partial

import torch
from torch.nn import functional as F
from flash_attn import flash_attn_func
from flash_attn.flash_attn_triton_og import attention as flash_attn_triton_og
from flash_attn.flash_attn_triton import flash_attn_func as flash_attn_triton

from flash_attn_triton import MetaData
from flash_attn_triton import attention as pure_triton_attn

if __name__ == "__main__":
    batch = 4
    seq_len_q, seq_len_k = 32, 32
    num_heads = 6
    num_heads_k = num_heads
    d = 128
    input_kwargs = dict(device="cuda", dtype=torch.float16)

    q = torch.randn((batch, seq_len_q, num_heads, d), **input_kwargs)
    k = torch.randn((batch, seq_len_k, num_heads_k, d), **input_kwargs)
    v = torch.randn((batch, seq_len_k, num_heads_k, d), **input_kwargs)

    metadata = MetaData(sm_scale=1.0/(d**0.5))
    metadata.layout = "bshd"
    metadata.max_seqlens_q = seq_len_q
    metadata.max_seqlens_k = seq_len_k

    output = F.scaled_dot_product_attention(q, k, v)
    output_0 = flash_attn_func(q, k, v, return_attn_probs=False)
    output_triton, _ = pure_triton_attn(q, k, v, None, metadata)
    # output_1 = flash_attn_triton(q, k, v)                         # Compile error
    # output_2 = flash_attn_triton_og(q, k, v, 1.0 /(d ** -0.5))    # Compile error

    print(torch.allclose(output, output_triton, atol=1e-3, rtol=1e-2))  # False
    print(torch.allclose(output, output_0, atol=1e-3, rtol=1e-2))       # False

    torch.testing.assert_close(output_0, output_triton, atol=1e-3, rtol=1e-2)   # True

    # print(torch.allclose(output_0, output))
    # print(torch.allclose(output_0, output_1))
    # print(torch.allclose(output_0, output_2))
    # print(torch.allclose(output_1, output_2))
