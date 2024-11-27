from functools import partial

import torch
from torch.nn import functional as F
from xformers import ops as xops
from flash_attn import flash_attn_func
from flash_attn.flash_attn_triton_og import attention as flash_attn_triton_func
import triton

from common import is_cuda, is_rocm, is_hopper
from xformers_impl import xformers_attn_ck, xformers_attn_cutlass
from xformers_impl import (
    xformers_attn_triton_s1 as xformers_attn_triton,
    # xformers_attn_triton_s2,
    # xformers_attn_triton_s4,
    # xformers_attn_triton_s8,
    # xformers_attn_triton_s16,
    # xformers_attn_triton_s32,
    # xformers_attn_triton_s64,
)
from pt_impl import pt_sdpa_cpu, pt_flash, pt_xformers, pt_math
from flash_impl import flash3_attn
from pure_triton_impl import pure_triton_attn_bshd, pure_triton_attn_bhsd


# wrap flash_attn_triton to pass sm_scale
def flash_attn_triton(q, k, v):
    sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return flash_attn_triton_func(q, k, v, sm_scale)


def calculate_tflops(latency, batch_size, sequence_length, num_heads, head_dim):
    # FLOPs for self-attention = 4 * batch_size * sequence_length^2 * num_heads * head_dim
    flops = 4 * batch_size * (sequence_length ** 2) * num_heads * head_dim
    tflops = flops / (latency * 10**12)  # Convert to TFLOPs
    return tflops


def prepare_input(bs, seq_len, num_heads, head_dim, layout="bshd", device="cuda"):
    if layout == "bshd":
        q = torch.randn(bs, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(bs, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(bs, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    else: 
        assert layout == "bhsd"
        q = torch.randn(bs, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(bs, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(bs, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    return q, k, v


def benchmark(name, func, batch_size, sequence_length, num_heads, head_dim, layout="bshd", device="cuda"):
    WARM_UP = 25
    REP = 100
    query, key, value = prepare_input(batch_size, sequence_length, num_heads,
                                      head_dim, layout=layout, device=device)

    ms = triton.testing.do_bench(partial(func, query, key, value),
                                 warmup=WARM_UP, rep=REP, return_mode="mean")
    tflops = calculate_tflops(ms * 1e-3, batch_size, sequence_length, num_heads, head_dim)
    print(f"{name:<20} \tMilisec={ms:.6f} \tTFLOPS={tflops:.6f}")


if __name__ == "__main__":
    # # SD3 input shape
    # batch_size = 2
    # sequence_length = 4429
    # num_heads = 24
    # head_dim = 64

    # OpenSora input shape
    batch_size = 60
    sequence_length = 3600
    num_heads = 16
    head_dim = 72

    benchmark("cpu-impl", pt_sdpa_cpu, batch_size, sequence_length, num_heads, head_dim, device="cpu")

    benchmark("flash_attn", flash_attn_func, batch_size, sequence_length, num_heads, head_dim, layout="bshd")
    # benchmark("flash_attn-triton", flash_attn_triton, batch_size, sequence_length, num_heads, head_dim) # Compile error
    if flash3_attn is not None:
        benchmark("flash_attn3", flash3_attn, batch_size, sequence_length, num_heads, head_dim, layout="bshd")

    benchmark("xformers-default", xops.memory_efficient_attention, batch_size, sequence_length, num_heads, head_dim, layout="bshd")
    if is_rocm():
        benchmark("xformers-ck", xformers_attn_ck, batch_size, sequence_length, num_heads, head_dim, layout="bshd")
    if is_cuda():
        benchmark("xformers-cutlass", xformers_attn_cutlass, batch_size, sequence_length, num_heads, head_dim, layout="bshd")

    # benchmark("xformers-triton", xformers_attn_triton, batch_size, sequence_length, num_heads, head_dim, layout="bshd")

    # These ones perform so bad, commented out
    # benchmark("xformers-triton-s2", xformers_attn_triton_s2, batch_size, sequence_length, num_heads, head_dim, layout="bshd")
    # benchmark("xformers-triton-s4", xformers_attn_triton_s4, batch_size, sequence_length, num_heads, head_dim, layout="bshd")
    # benchmark("xformers-triton-s8", xformers_attn_triton_s8, batch_size, sequence_length, num_heads, head_dim, layout="bshd")
    # benchmark("xformers-triton-s16", xformers_attn_triton_s16, batch_size, sequence_length, num_heads, head_dim, layout="bshd")
    # benchmark("xformers-triton-s32", xformers_attn_triton_s32, batch_size, sequence_length, num_heads, head_dim, layout="bshd")
    # benchmark("xformers-triton-s64", xformers_attn_triton_s64, batch_size, sequence_length, num_heads, head_dim, layout="bshd")

    if pure_triton_attn_bshd is not None:
        benchmark("triton-bshd", pure_triton_attn_bshd, batch_size, sequence_length, num_heads, head_dim, layout="bshd")

    if pure_triton_attn_bhsd is not None:
        benchmark("triton-bhsd", pure_triton_attn_bhsd, batch_size, sequence_length, num_heads, head_dim, layout="bhsd")

    benchmark("pytorch-default", F.scaled_dot_product_attention, batch_size, sequence_length, num_heads, head_dim, layout="bhsd")
    benchmark("pytorch-flash", pt_flash, batch_size, sequence_length, num_heads, head_dim, layout="bhsd")
    benchmark("pytorch-xformers", pt_xformers, batch_size, sequence_length, num_heads, head_dim, layout="bhsd")
    # benchmark("pytorch-math", pt_math, batch_size, sequence_length, num_heads, head_dim, layout="bhsd")


