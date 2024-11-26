from common import is_cuda, is_rocm

if is_rocm():
    from flash_attn_triton import MetaData
    from flash_attn_triton import attention as attn_triton_impl


    def pure_triton_attn_bshd(q, k, v):
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        metadata = MetaData(sm_scale)
        metadata.layout = "bshd"
        metadata.max_seqlens_q = q.shape[1]
        metadata.max_seqlens_k = k.shape[1]
        out, _ = attn_triton_impl(q, k, v, None, metadata)
        return out


    def pure_triton_attn_bhsd(q, k, v):
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        metadata = MetaData(sm_scale)
        metadata.layout = "bhsd"
        metadata.max_seqlens_q = q.shape[2]
        metadata.max_seqlens_k = k.shape[2]
        out, _ = attn_triton_impl(q, k, v, None, metadata)
        return out

else: 
    assert is_cuda()

    from flash_attn_triton_orig import attention as attn_triton_impl

    pure_triton_attn_bshd = None

    def pure_triton_attn_bhsd(q, k, v):
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        causal = False
        out, *_ = attn_triton_impl(q, k, v, causal, sm_scale)
        return out.half()
