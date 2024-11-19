from functools import partial
import time

import torch

# https://github.com/ROCm/xformers
import xformers.ops as xops
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

from pt_impl import pt_sdpa_cpu, pt_flash, pt_xformers

# https://github.com/ROCm/flash-attention
from flash_attn import flash_attn_func
from flash_attn.flash_attn_triton_og import attention as flash_attn_triton_func

# get from: https://github.com/ROCm/triton/blob/db2ca015159c6592c30a6bfcd77b9cc540063a8e/python/perf-kernels/flash-attention.py
from flash_attn_triton import MetaData
from flash_attn_triton import attention as attn_triton_impl


def calculate_tflops(latency, batch_size, sequence_length, num_heads, head_dim):
    # FLOPs for self-attention = 4 * batch_size * sequence_length^2 * num_heads * head_dim
    flops = 4 * batch_size * (sequence_length ** 2) * num_heads * head_dim
    tflops = flops / (latency * 10**12)  # Convert to TFLOPs
    return tflops


def prepare_input(bs, seq_len, num_heads, head_dim, device="cuda"):
    q = torch.randn(bs, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(bs, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(bs, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    return q, k, v


def benchmark(name, func, shapes, device='cuda'):
    query, key, value = prepare_input(*shapes, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = func(query, key, value)

    # Measure latency
    latencies = []
    tflopses = []
    for _ in range(30):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _ = flash_attn_func(query, key, value)
        torch.cuda.synchronize()
        latency = time.time() - start_time
        latencies.append(latency)
        tflopses.append(calculate_tflops(latency, batch_size, sequence_length, num_heads, head_dim))

    avg = lambda x: sum(x) / len(x)
    avg_latency = avg(latencies)
    avg_tflops = avg(tflopses)
    print(f"{name}: Avg latency: {avg_latency:.6f} s, Avg TFLOPs: {avg_tflops:.6f}")


if __name__ == "__main__":
    # SD3 input shape
    batch_size = 2
    sequence_length = 16384
    num_heads = 24
    head_dim = 64
    shapes = (batch_size, sequence_length, num_heads, head_dim)

    # Set up func
    xformers_attn_ck = partial(xops.memory_efficient_attention, op=MemoryEfficientAttentionCkOp)
    xformers_attn_triton = partial(xops.memory_efficient_attention, op=(TritonFw_S64, )) # SplitK seems not affect perf much...

    # wrap flash_attn_triton to pass sm_scale
    def flash_attn_triton(q, k, v):
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        return flash_attn_triton_func(q, k, v, sm_scale)

    # wrap pure triton impl
    def pure_triton_attn(q, k, v):
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        metadata = MetaData(sm_scale)
        metadata.layout = "bshd"
        metadata.max_seqlens_q = q.shape[1]
        metadata.max_seqlens_k = k.shape[1]
        return attn_triton_impl(q, k, v, None, metadata)


    # benchmark("cpu-impl", pt_sdpa_cpu, shapes, device="cpu")
    benchmark("flash_attn-ck", flash_attn_func, shapes)
    # benchmark("flash_attn-triton", flash_attn_triton, shapes) # Compile error
    benchmark("xformers-ck", xformers_attn_ck, shapes)
    benchmark("xformers-triton", xformers_attn_triton, shapes)
    benchmark("pure-triton", pure_triton_attn, shapes)
    benchmark("pytorch-default", F.scaled_dot_product_attention, shapes)
    benchmark("pytorch-flash", pt_flash, shapes)
    benchmark("pytorch-xformers", pt_xformers, shapes)


