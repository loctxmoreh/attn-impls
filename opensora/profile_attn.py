import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functools import partial
from pathlib import Path

import torch
import triton
import xformers.ops as xops
from torch.profiler import ProfilerActivity, profile

from opensora.attention import ATTENTION_CONFIGS, prepare_attn_input
from opensora.utils import trace_handler_wrapper
from xformers_impl import xformers_padded

torch.manual_seed(42)

# Prepare attn configs
layout = "bshd"
origin_func = xops.memory_efficient_attention
padded_func = xformers_padded
attn_name = "multihead-attn"
func_name = "xformers-default"

attn_config = ATTENTION_CONFIGS[attn_name]
num_runs = 100
warm_up = 25
dtype = torch.float16
device = "cuda"
save_dir = Path("profiling_data")

# Prepare inputs
q, k, v, attn_bias = prepare_attn_input(
    attn_config, layout=layout, device=device, dtype=dtype
)
if attn_bias is None:
    args = (q, k, v)
else:
    args = (q, k, v, attn_bias)

# Re-benchmark
print("Re-benchmarking...")
func = partial(origin_func, *args)
ms = triton.testing.do_bench(func, warmup=warm_up, rep=num_runs, return_mode="mean")
print("Original Latency: {:.2f}ms".format(ms))

func = partial(padded_func, *args)
ms = triton.testing.do_bench(func, warmup=warm_up, rep=num_runs, return_mode="mean")
print("Padded Latency: {:.2f}ms".format(ms))

print("Profiling...")
# Normal attention
for i in range(warm_up):
    out = origin_func(*args)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    on_trace_ready=trace_handler_wrapper(f"{attn_name}.{func_name}.origin", save_dir),
):
    for i in range(num_runs):
        out = origin_func(*args)


# Padding attention
for i in range(warm_up):
    out = padded_func(*args)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    on_trace_ready=trace_handler_wrapper(f"{attn_name}.{func_name}.padded", save_dir),
):
    for i in range(num_runs):
        out = padded_func(*args)

print("Done!")
