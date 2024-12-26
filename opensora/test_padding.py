import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import xformers.ops as xops
from torch.nn import functional as F

from opensora.attention import ATTENTION_CONFIGS, prepare_attn_input
from xformers_impl import xformers_padded

# Prepare attn configs
layout = "bshd"
attn_name = "multihead-attn"
attn_config = ATTENTION_CONFIGS[attn_name]
dtype = torch.float16
device = "cuda"

# Prepare inputs
q, k, v, attn_bias = prepare_attn_input(
    attn_config, layout=layout, device=device, dtype=dtype
)
print(f"{q.shape=}")
print(f"{k.shape=}")
print(f"{v.shape=}")
print(f"{attn_bias.shape=}")

out1 = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
out2 = xformers_padded(q, k, v, attn_bias=attn_bias)

print(out1[0, 0, 0])
print(out2[0, 0, 0])

print(torch.max(torch.abs(out1 - out2)))
print(torch.allclose(out1, out2, rtol=0.01, atol=0.001))
