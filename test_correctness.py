import torch
from torch.nn import functional as F

from common import is_hopper, is_rocm, is_cuda
from flash_attn import flash_attn_func
from flash_impl import flash3_attn
from xformers_impl import xformers_attn_ck, xformers_attn_cutlass, xformers_attn_triton
from pt_impl import pt_flash, pt_xformers, pt_math
from pure_triton_impl import pure_triton_attn_bshd, pure_triton_attn_bhsd

torch.manual_seed(42)


def main():
    # # SD3
    # bs = 2
    # seq_len = 4429
    # num_heads = 24
    # head_dim = 64

    # OpenSora
    bs = 60
    seq_len = 3600
    num_heads = 16
    head_dim = 72

    device = 'cuda'
    # rtol, atol = 1e-3, 1e-5     # too strict
    # rtol, atol = 1e-2, 1e-2     # work
    # rtol, atol = 1e-3, 1e-3     # work
    rtol, atol = 1e-2, 1e-3

    print(f"{rtol=}, {atol=}")

    q = torch.randn(bs, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(bs, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(bs, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    expected = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())

    pt_output = F.scaled_dot_product_attention(q, k, v)
    pt_flash_output = pt_flash(q, k, v)
    pt_xformers_output = pt_xformers(q, k, v)
    pt_math_output = pt_math(q, k, v)

    if is_rocm():
        xformers_ck_output = xformers_attn_ck(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    else: 
        assert is_cuda()
        xformers_cutlass_output = xformers_attn_cutlass(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        
    # xformers_triton_output = xformers_attn_triton(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    flash_output = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    
    if flash3_attn is not None:
        flash3_output = flash3_attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    if pure_triton_attn_bshd is not None:
        triton_bshd_output = pure_triton_attn_bshd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    if pure_triton_attn_bhsd is not None:
        triton_bhsd_output = pure_triton_attn_bhsd(q, k, v)

    print(f"{torch.allclose(pt_output.cpu(), expected, rtol=rtol, atol=atol)=}")
    print(f"{torch.allclose(pt_flash_output.cpu(), expected, rtol=rtol, atol=atol)=}")
    print(f"{torch.allclose(pt_xformers_output.cpu(), expected, rtol=rtol, atol=atol)=}")
    print(f"{torch.allclose(pt_math_output.cpu(), expected, rtol=rtol, atol=atol)=}")

    if is_rocm():
        print(f"{torch.allclose(xformers_ck_output.cpu(), expected, rtol=rtol, atol=atol)=}")
    else:
        assert is_cuda()
        print(f"{torch.allclose(xformers_cutlass_output.cpu(), expected, rtol=rtol, atol=atol)=}")

    # print(f"{torch.allclose(xformers_triton_output.cpu(), expected, rtol=rtol, atol=atol)=}")
    
    print(f"{torch.allclose(flash_output.cpu(), expected, rtol=rtol, atol=atol)=}")

    if flash3_attn is not None:
        print(f"{torch.allclose(flash3_output.cpu(), expected, rtol=rtol, atol=atol)=}")

    if pure_triton_attn_bshd is not None:
        print(f"{torch.allclose(triton_bshd_output.cpu(), expected, rtol=rtol, atol=atol)=}")

    if pure_triton_attn_bhsd is not None:
        print(f"{torch.allclose(triton_bhsd_output.cpu(), expected, rtol=rtol, atol=atol)=}")
    


if __name__ == "__main__":
    main()
