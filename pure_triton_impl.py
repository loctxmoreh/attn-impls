from flash_attn_triton import MetaData
from flash_attn_triton import attention as attn_triton_impl



def pure_triton_attn(q, k, v):
    sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    metadata = MetaData(sm_scale)
    metadata.layout = "bshd"
    metadata.max_seqlens_q = q.shape[1]
    metadata.max_seqlens_k = k.shape[1]
    return attn_triton_impl(q, k, v, None, metadata)