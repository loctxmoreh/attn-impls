import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import warnings

import torch
from flash_attn import flash_attn_func
from torch.nn import functional as F
from xformers import ops as xops

from common import is_cuda, is_rocm
from flash_impl import flash3_attn, flash3_attn_padded, flash_attn_padded
from opensora.attention import ATTENTION_CONFIGS, benchmark_attn
from pt_impl import (
    pt_flash,
    pt_flash_padded,
    pt_padded,
    pt_xformers,
    pt_xformers_padded,
)
from pure_triton_impl import (
    pure_triton_attn_bhsd,
    pure_triton_attn_bshd,
    pure_triton_bhsd_padded,
    pure_triton_bshd_padded,
)
from xformers_impl import xformers_attn_ck, xformers_attn_cutlass
from xformers_impl import xformers_attn_triton_s1 as xformers_attn_triton
from xformers_impl import (
    xformers_ck_padded,
    xformers_cutlass_padded,
    xformers_padded,
    xformers_triton_padded,
)

warnings.filterwarnings("ignore", category=UserWarning)  # ignore warnings


if __name__ == "__main__":
    device, dtype = "cuda", torch.float16
    print(f"{device=}, {dtype=}")

    for attn_name, attn_config in ATTENTION_CONFIGS.items():
        print("\nAttention: {}".format(attn_name))

        # pytorch default impl
        benchmark_attn(
            "pytorch-default",
            F.scaled_dot_product_attention,
            attn_config,
            layout="bhsd",
            device=device,
            dtype=dtype,
        )
        benchmark_attn(
            "pytorch-default-padded",
            pt_padded,
            attn_config,
            layout="bhsd",
            device=device,
            dtype=dtype,
        )

        # Pytorch flash impl
        benchmark_attn(
            "pytorch-flash",
            pt_flash,
            attn_config,
            layout="bhsd",
            device=device,
            dtype=dtype,
        )
        benchmark_attn(
            "pytorch-flash-padded",
            pt_flash_padded,
            attn_config,
            layout="bhsd",
            device=device,
            dtype=dtype,
        )

        # Pytorch xformers impl
        benchmark_attn(
            "pytorch-xformers",
            pt_xformers,
            attn_config,
            layout="bhsd",
            device=device,
            dtype=dtype,
        )
        benchmark_attn(
            "pytorch-xformers-padded",
            pt_xformers_padded,
            attn_config,
            layout="bhsd",
            device=device,
            dtype=dtype,
        )

        # flash-attn
        benchmark_attn(
            "flash-attn-ck",
            flash_attn_func,
            attn_config,
            layout="bshd",
            device=device,
            dtype=dtype,
        )
        benchmark_attn(
            "flash-attn-ck-padded",
            flash_attn_padded,
            attn_config,
            layout="bshd",
            device=device,
            dtype=dtype,
        )

        if flash3_attn is not None:
            benchmark_attn(
                "flash-attn3",
                flash3_attn,
                attn_config,
                layout="bshd",
                device=device,
                dtype=dtype,
            )

            benchmark_attn(
                "flash-attn3-padded",
                flash3_attn_padded,
                attn_config,
                layout="bshd",
                device=device,
                dtype=dtype,
            )

        # xformers
        benchmark_attn(
            "xformers-default",
            xops.memory_efficient_attention,
            attn_config,
            layout="bshd",
            device=device,
            dtype=dtype,
        )
        benchmark_attn(
            "xformers-default-padded",
            xformers_padded,
            attn_config,
            layout="bshd",
            device=device,
            dtype=dtype,
        )

        if is_rocm():
            benchmark_attn(
                "xformers-ck",
                xformers_attn_ck,
                attn_config,
                layout="bshd",
                device=device,
                dtype=dtype,
            )
            benchmark_attn(
                "xformers-ck-padded",
                xformers_ck_padded,
                attn_config,
                layout="bshd",
                device=device,
                dtype=dtype,
            )
        if is_cuda():
            benchmark_attn(
                "xformers-cutlass",
                xformers_attn_cutlass,
                attn_config,
                layout="bshd",
                device=device,
                dtype=dtype,
            )
            benchmark_attn(
                "xformers-cutlass-padded",
                xformers_cutlass_padded,
                attn_config,
                layout="bshd",
                device=device,
                dtype=dtype,
            )
        benchmark_attn(
            "xformers-triton",
            xformers_attn_triton,
            attn_config,
            layout="bshd",
            device=device,
            dtype=dtype,
        )
        benchmark_attn(
            "xformers-triton-padded",
            xformers_triton_padded,
            attn_config,
            layout="bshd",
            device=device,
            dtype=dtype,
        )

        # pure triton
        if pure_triton_attn_bshd is not None:
            benchmark_attn(
                "triton-bshd",
                pure_triton_attn_bshd,
                attn_config,
                layout="bshd",
                device=device,
                dtype=dtype,
            )

            benchmark_attn(
                "triton-bshd-padded",
                pure_triton_bshd_padded,
                attn_config,
                layout="bshd",
                device=device,
                dtype=dtype,
            )

        if pure_triton_attn_bhsd is not None:
            benchmark_attn(
                "triton-bhsd",
                pure_triton_attn_bhsd,
                attn_config,
                layout="bhsd",
                device=device,
                dtype=dtype,
            )

            benchmark_attn(
                "triton-bhsd-padded",
                pure_triton_bhsd_padded,
                attn_config,
                layout="bhsd",
                device=device,
                dtype=dtype,
            )
