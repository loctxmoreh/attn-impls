import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import torch
import xformers.ops as xops
from torch.profiler import ProfilerActivity, profile, schedule

from opensora.attention import ATTENTION_CONFIGS, prepare_attn_input
from opensora.utils import trace_handler_wrapper
from xformers_impl import xformers_padded

torch.manual_seed(42)

# Prepare attn configs
attn_name = "multihead-attn"
attn_config = ATTENTION_CONFIGS[attn_name]
num_runs = 30
layout = "bshd"
dtype = torch.float16
device = "cuda"
save_dir = Path("profiling_data")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
        wait=25,  # Number of steps to skip
        warmup=0,  # Number of steps to include in the warm-up phase
        active=5,  # Number of steps to include in the active phase (profiling)
        repeat=1,  # Number of times to repeat the above schedule
    ),
    record_shapes=True,
    with_stack=True,
    on_trace_ready=trace_handler_wrapper(
        f"{attn_name}.xformers-default.origin", save_dir
    ),
) as prof:
    for i in range(num_runs):
        q, k, v, attn_bias = prepare_attn_input(
            attn_config, layout=layout, device=device, dtype=dtype
        )
        out = xops.memory_efficient_attention(q, k, v)
        prof.step()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
        wait=25,  # Number of steps to skip
        warmup=0,  # Number of steps to include in the warm-up phase
        active=5,  # Number of steps to include in the active phase (profiling)
        repeat=1,  # Number of times to repeat the above schedule
    ),
    record_shapes=True,
    with_stack=True,
    on_trace_ready=trace_handler_wrapper(
        f"{attn_name}.xformers-default.padded", save_dir
    ),
) as prof:
    for i in range(num_runs):
        q, k, v, attn_bias = prepare_attn_input(
            attn_config, layout=layout, device=device, dtype=dtype
        )
        out = xformers_padded(q, k, v)
        prof.step()

print("Done!")
