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
# from xformers.ops.fmha.triton_splitk import (
#     BwOp_S1 as TritonBw_S1,
#     BwOp_S2 as TritonBw_S2,
#     BwOp_S4 as TritonBw_S4,
#     BwOp_S8 as TritonBw_S8,
#     BwOp_S16 as TritonBw_S16,
#     BwOp_S32 as TritonBw_S32,
#     BwOp_S64 as TritonBw_S64,
#     BwOp_S128 as TritonBw_S128,
# )

# MemoryEfficientAttentionTritonS1_Op = (TritonFw_S1, TritonBw_S1)
# MemoryEfficientAttentionTritonS2_Op = (TritonFw_S2, TritonBw_S2)
# MemoryEfficientAttentionTritonS4_Op = (TritonFw_S4, TritonBw_S4)
# MemoryEfficientAttentionTritonS8_Op = (TritonFw_S8, TritonBw_S8)
# MemoryEfficientAttentionTritonS16_Op = (TritonFw_S16, TritonBw_S16)
# MemoryEfficientAttentionTritonS32_Op = (TritonFw_S32, TritonBw_S32)
# MemoryEfficientAttentionTritonS64_Op = (TritonFw_S64, TritonBw_S64)
# MemoryEfficientAttentionTritonS128_Op = (TritonFw_S128, TritonBw_S128)

xformers_attn_ck = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionCkOp)
xformers_attn_triton = partial(xops.memory_efficient_attention, op=(TritonFw_S1, ))

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
