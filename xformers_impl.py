from functools import partial

import torch 
from torch.nn import functional as F
from xformers import ops as xops
from xformers.ops.fmha import MemoryEfficientAttentionCkOp
from xformers.ops.fmha.triton_splitk import (
    FwOp_S1 as TritonFw_S1,
    FwOp_S2 as TritonFw_S2,
    FwOp_S4 as TritonFw_S4,
    FwOp_S8 as TritonFw_S8,
    FwOp_S16 as TritonFw_S16,
    FwOp_S32 as TritonFw_S32,
    FwOp_S64 as TritonFw_S64,
    FwOp_S128 as TritonFw_S128,
)

MemoryEfficientAttentionTritonS1_Op = (TritonFw_S1, )
MemoryEfficientAttentionTritonS2_Op = (TritonFw_S2, )
MemoryEfficientAttentionTritonS4_Op = (TritonFw_S4, )
MemoryEfficientAttentionTritonS8_Op = (TritonFw_S8, )
MemoryEfficientAttentionTritonS16_Op = (TritonFw_S16, )
MemoryEfficientAttentionTritonS32_Op = (TritonFw_S32, )
MemoryEfficientAttentionTritonS64_Op = (TritonFw_S64, )
MemoryEfficientAttentionTritonS128_Op = (TritonFw_S128, )

xformers_attn_ck = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionCkOp)

xformers_attn_triton_s1 = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionTritonS1_Op)
xformers_attn_triton_s2 = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionTritonS2_Op)
xformers_attn_triton_s4 = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionTritonS4_Op)
xformers_attn_triton_s8 = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionTritonS8_Op)
xformers_attn_triton_s16 = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionTritonS16_Op)
xformers_attn_triton_s32 = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionTritonS32_Op)
xformers_attn_triton_s64 = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionTritonS64_Op)
xformers_attn_triton_s128 = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionTritonS128_Op)


# In context of GPU, I have no idea why the need for multiple split-K impls
# Choosing S1 as default
xformers_attn_triton = xformers_attn_triton_s1

if __name__ == "__main__":
    B, M, K = 3, 32, 128
    # input_kwargs = dict(device="cuda", dtype=torch.float16)
    input_kwargs = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn((B, M, 8, K), **input_kwargs)
    k = torch.randn((B, M, 2, K), **input_kwargs)
    v = torch.randn((B, M, 2, K), **input_kwargs)

    q = q.reshape((B, M, 2, 4, K))
    k = k.reshape((B, M, 2, 1, K)).expand((B, M, 2, 4, K))
    v = v.reshape((B, M, 2, 1, K)).expand((B, M, 2, 4, K))

    output_sdpa = F.scaled_dot_product_attention(q, k, v)
    print(f"{output_sdpa.shape=}")

    output = xops.memory_efficient_attention(q, k, v)
    print(f"{output.shape=}")
    output_ck = xformers_attn_ck(q, k, v)
    output_triton = xformers_attn_triton(q, k, v)

    # print(torch.backends.cuda.mem_efficient_sdp_enabled())    # True
    # torch.backends.cuda.enable_mem_efficient_sdp(True)

    # print(torch.allclose(output_ck, output))        # True
    # torch.testing.assert_close(output_ck, output)        # True

    # print(torch.allclose(output, output_triton, rtol=1e-03, atol=1e-05))      # False
    # print(torch.allclose(output, output_triton, rtol=1e-02, atol=1e-03))      
    #
    # print(torch.allclose(output_ck, output_triton, rtol=1e-03, atol=1e-05))   # False
    # print(torch.allclose(output_ck, output_triton, rtol=1e-02, atol=1e-03))  
    # torch.testing.assert_close(output_ck, output_triton) 

    print(torch.allclose(output, output_sdpa, rtol=1e-02, atol=1e-03)) # False
    # torch.testing.assert_close(output, output_sdpa) 
