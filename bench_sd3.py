import time

import torch
from torch.nn import functional as F

from xformers import ops as xops

from xformers_impl import xformers_attn_ck
from xformers_impl import xformers_attn_triton_s64 as xformers_attn_triton

# https://github.com/ROCm/flash-attention
from flash_attn import flash_attn_func
from flash_attn.flash_attn_triton_og import attention as flash_attn_triton_func

from flash_attn_triton import MetaData
from flash_attn_triton import attention as attn_triton_impl

from pt_impl import pt_flash, pt_xformers


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
            _ = func(query, key, value)
        torch.cuda.synchronize()
        latency = time.time() - start_time
        latencies.append(latency)
        tflopses.append(calculate_tflops(latency, batch_size, sequence_length, num_heads, head_dim))

    avg = lambda x: sum(x) / len(x)
    avg_latency = avg(latencies)
    avg_tflops = avg(tflopses)
    print(f"{name:<20}:\tAvg latency: {avg_latency:.6f} s,\tAvg TFLOPs: {avg_tflops:.6f}")


if __name__ == "__main__":
    # SD3 input shape
    batch_size = 2
    sequence_length = 16384
    num_heads = 24
    head_dim = 64
    shapes = (batch_size, sequence_length, num_heads, head_dim)

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


    # benchmark("cpu-impl", pt_sdpa_cpu, shapes, device="cpu")  # Dispatch error
    benchmark("flash_attn-ck", flash_attn_func, shapes)
    # benchmark("flash_attn-triton", flash_attn_triton, shapes) # Compile error
    benchmark("xformers-default", xops.memory_efficient_attention, shapes)
    benchmark("xformers-ck", xformers_attn_ck, shapes)
    benchmark("xformers-triton", xformers_attn_triton, shapes)
    benchmark("pure-triton", pure_triton_attn, shapes)
    benchmark("pytorch-default", F.scaled_dot_product_attention, shapes)
    benchmark("pytorch-flash", pt_flash, shapes)
    benchmark("pytorch-xformers", pt_xformers, shapes)


