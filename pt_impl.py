import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def pt_sdpa_cpu(q, k, v):
    assert q.device == torch.device('cpu')
    assert k.device == torch.device('cpu')
    assert v.device == torch.device('cpu')

    # with sdpa_kernel(backends=[SDPBackend.MATH]):
    return F.scaled_dot_product_attention(q, k, v)


def pt_flash(q, k, v):
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v)


def pt_xformers(q, k, v):
    with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v)
