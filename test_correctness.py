import torch
from torch.nn import functional as F

from flash_attn import flash_attn_func
from xformers_impl import xformers_attn_ck, xformers_attn_triton
from pt_impl import pt_flash, pt_xformers

torch.manual_seed(42)


def main():
    bs = 2
    seq_len = 24
    num_heads = 4429
    head_dim = 64
    device = 'cuda'
    # rtol, atol = 1e-3, 1e-5
    rtol, atol = 1e-2, 1e-3

    print(f"{rtol=}, {atol=}")

    q = torch.randn(bs, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(bs, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(bs, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    expected = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())

    pt_output = F.scaled_dot_product_attention(q, k, v)
    pt_flash_output = pt_flash(q, k, v)
    pt_xformers_output = pt_xformers(q, k, v)

    xformers_ck_output = xformers_attn_ck(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    xformers_triton_output = xformers_attn_triton(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    flash_output = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    print(f"{torch.allclose(pt_output.cpu(), expected, rtol=rtol, atol=atol)=}")
    print(f"{torch.allclose(pt_flash_output.cpu(), expected, rtol=rtol, atol=atol)=}")
    print(f"{torch.allclose(pt_xformers_output.cpu(), expected, rtol=rtol, atol=atol)=}")

    print(f"{torch.allclose(xformers_ck_output.cpu(), expected, rtol=rtol, atol=atol)=}")
    print(f"{torch.allclose(xformers_triton_output.cpu(), expected, rtol=rtol, atol=atol)=}")
    
    print(f"{torch.allclose(flash_output.cpu(), expected, rtol=rtol, atol=atol)=}")


if __name__ == "__main__":
    main()
