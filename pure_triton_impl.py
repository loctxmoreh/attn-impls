import math

from torch.nn import functional as F

from common import is_cuda, is_rocm


# Padded functions
def apply_padded_triton_attn(q, k, v, attn_func):
    target_head_dim = 128
    origin_head_dim = q.shape[-1]
    origin_scale = 1 / math.sqrt(origin_head_dim)

    padding_amount = target_head_dim - origin_head_dim
    padding_value = 0

    q = F.pad(q, (0, padding_amount), "constant", padding_value)
    k = F.pad(k, (0, padding_amount), "constant", padding_value)
    v = F.pad(v, (0, padding_amount), "constant", padding_value)

    attn_output = attn_func(q, k, v, scale=origin_scale)
    return attn_output[..., :origin_head_dim]


if is_rocm():
    from flash_attn_triton import MetaData
    from flash_attn_triton import attention as attn_triton_impl

    def pure_triton_attn_bshd(q, k, v, scale=None):
        sm_scale = 1.0 / (q.shape[-1] ** 0.5) if scale is None else scale
        metadata = MetaData(sm_scale)
        metadata.layout = "bshd"
        metadata.max_seqlens_q = q.shape[1]
        metadata.max_seqlens_k = k.shape[1]
        out, _ = attn_triton_impl(q, k, v, None, metadata)
        return out

    def pure_triton_attn_bhsd(q, k, v, scale=None):
        sm_scale = 1.0 / (q.shape[-1] ** 0.5) if scale is None else scale
        metadata = MetaData(sm_scale)
        metadata.layout = "bhsd"
        metadata.max_seqlens_q = q.shape[2]
        metadata.max_seqlens_k = k.shape[2]
        out, _ = attn_triton_impl(q, k, v, None, metadata)
        return out

    def pure_triton_bshd_padded(q, k, v):
        return apply_padded_triton_attn(q, k, v, pure_triton_attn_bshd)

    def pure_triton_bhsd_padded(q, k, v):
        return apply_padded_triton_attn(q, k, v, pure_triton_attn_bhsd)

else:
    assert is_cuda()

    from flash_attn_triton_orig import attention as attn_triton_impl

    pure_triton_attn_bshd = None
    pure_triton_bshd_padded = None

    def pure_triton_attn_bhsd(q, k, v, scale=None):
        sm_scale = 1.0 / (q.shape[-1] ** 0.5) if scale is None else scale
        causal = False
        out, *_ = attn_triton_impl(q, k, v, causal, sm_scale)
        return out.half()

    def pure_triton_bhsd_padded(q, k, v):
        return apply_padded_triton_attn(q, k, v, pure_triton_attn_bhsd)
