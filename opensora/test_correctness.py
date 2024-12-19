import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import warnings

import torch
from flash_attn import flash_attn_func
from torch.nn import functional as F

from common import is_cuda, is_rocm
from flash_impl import flash3_attn
from opensora.attention import ATTENTION_CONFIGS, prepare_attn_input
from opensora.utils import get_error_message
from pt_impl import pt_flash, pt_xformers
from pure_triton_impl import pure_triton_attn_bhsd, pure_triton_attn_bshd
from xformers_impl import xformers_attn_ck, xformers_attn_cutlass, xformers_attn_triton

torch.manual_seed(42)
warnings.filterwarnings("ignore", category=UserWarning)  # ignore warnings


def main():
    rtol, atol = 1e-2, 1e-3
    device, dtype = "cuda", torch.float16
    print(f"{device=}, {dtype=}")
    print(f"{rtol=}, {atol=}")

    for attn_name, attn_config in ATTENTION_CONFIGS.items():
        print("\nAttention: {}".format(attn_name))
        q, k, v, attn_bias = prepare_attn_input(
            attn_config, layout="bhsd", device=device, dtype=dtype
        )
        try:
            # can't use CPU impl for attn_bias
            expected = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        except Exception as e:
            print(f"expected: {get_error_message(e)}")

        # Pytorch implementation
        try:
            pt_flash_output = pt_flash(q, k, v, attn_bias=attn_bias)
            print(
                f"pt_flash_output: {torch.allclose(pt_flash_output, expected, rtol=rtol, atol=atol)}"
            )
        except Exception as e:
            print(f"pt_flash_output: {get_error_message(e)}")

        try:
            pt_xformers_output = pt_xformers(q, k, v, attn_bias=attn_bias)
            print(
                f"pt_xformers_output: {torch.allclose(pt_xformers_output, expected, rtol=rtol, atol=atol)}"
            )
        except Exception as e:
            print(f"pt_xformers_output: {get_error_message(e)}")

        # xformers CK/cutlass
        if is_rocm():
            try:
                xformers_ck_output = xformers_attn_ck(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    attn_bias=attn_bias,
                ).transpose(1, 2)
                print(
                    f"xformers_ck_output: {torch.allclose(xformers_ck_output, expected, rtol=rtol, atol=atol)}"
                )
            except Exception as e:
                print(f"xformers_ck_output: {get_error_message(e)}")
        else:
            assert is_cuda()

            try:
                xformers_cutlass_output = xformers_attn_cutlass(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    attn_bias=attn_bias,
                ).transpose(1, 2)
                print(
                    f"xformers_cutlass_output: {torch.allclose(xformers_cutlass_output, expected, rtol=rtol, atol=atol)}"
                )
            except Exception as e:
                print(f"xformers_cutlass_output: {get_error_message(e)}")

        # xformers triton
        try:
            xformers_triton_output = xformers_attn_triton(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_bias=attn_bias,
            ).transpose(1, 2)
            print(
                f"xformers_triton_output: {torch.allclose(xformers_triton_output, expected, rtol=rtol, atol=atol)}"
            )
        except Exception as e:
            print(f"xformers_triton_output: {get_error_message(e)}")

        # flash-attn
        if attn_bias is None:
            try:
                flash_output = flash_attn_func(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                ).transpose(1, 2)
                print(
                    f"flash_output: {torch.allclose(flash_output, expected, rtol=rtol, atol=atol)}"
                )
            except Exception as e:
                print(f"flash_output: {get_error_message(e)}")
        else:
            print(f"flash_output: not support attn_bias")

        if flash3_attn is not None and attn_bias is None:
            try:
                flash3_output = flash3_attn(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                ).transpose(1, 2)
                print(
                    f"flash3_output: {torch.allclose(flash3_output, expected, rtol=rtol, atol=atol)}"
                )
            except Exception as e:
                print(f"flash3_output: {get_error_message(e)}")

        # pure triton
        if pure_triton_attn_bshd is not None and attn_bias is None:
            try:
                triton_bshd_output = pure_triton_attn_bshd(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                ).transpose(1, 2)
                print(
                    f"triton_bshd_output: {torch.allclose(triton_bshd_output, expected, rtol=rtol, atol=atol)}"
                )
            except Exception as e:
                print(f"triton_bshd_output: {get_error_message(e)}")

        if pure_triton_attn_bhsd is not None and attn_bias is None:
            try:
                triton_bhsd_output = pure_triton_attn_bhsd(q, k, v)
                print(
                    f"triton_bhsd_output: {torch.allclose(triton_bhsd_output, expected, rtol=rtol, atol=atol)}"
                )
            except Exception as e:
                print(f"triton_bhsd_output: {get_error_message(e)}")


if __name__ == "__main__":
    main()
