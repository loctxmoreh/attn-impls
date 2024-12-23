from functools import partial

import torch
import triton

from opensora.utils import calculate_tflops, get_error_message

# All Attention configs in Open-Sora
ATTENTION_CONFIGS = {
    "self-attn.spatial": {
        "qkv": {
            "batch_size": 60,
            "seq_len": 3600,
            "num_heads": 16,
            "head_dim": 72,
        }
    },
    "self-attn.temporal": {
        "qkv": {
            "batch_size": 7200,
            "seq_len": 30,
            "num_heads": 16,
            "head_dim": 72,
        }
    },
    "multihead-attn": {
        "q": {
            "batch_size": 1,
            "seq_len": 216000,
            "num_heads": 16,
            "head_dim": 72,
        },
        "kv": {
            "batch_size": 1,
            "seq_len": 600,
            "num_heads": 16,
            "head_dim": 72,
        },
        "attn_bias": True,
    },
}


def prepare_attn_input(attn_config, layout="bhsd", device="cuda", dtype=torch.float16):
    """Prepare q, k, v and attention bias in terms of attention config.

    Returns:
        q, k, v: Shape [batch_size, heads, seqlen, dim]
        attn_bias: Shape [q_seqlen, kv_seqlen]"""
    q, k, v = None, None, None
    attn_bias = None

    # Prepare qkv
    for name, config in attn_config.items():
        if "q" in name:
            if layout == "bhsd":
                q = torch.randn(
                    config["batch_size"],
                    config["num_heads"],
                    config["seq_len"],
                    config["head_dim"],
                    device=device,
                    dtype=dtype,
                )
            else:  # bshd
                q = torch.randn(
                    config["batch_size"],
                    config["seq_len"],
                    config["num_heads"],
                    config["head_dim"],
                    device=device,
                    dtype=dtype,
                )
            q_seqlen = config["seq_len"]
            batch_size = config["batch_size"]
            num_heads = config["num_heads"]

        if "kv" in name:
            if layout == "bhsd":
                k = torch.randn(
                    config["batch_size"],
                    config["num_heads"],
                    config["seq_len"],
                    config["head_dim"],
                    device=device,
                    dtype=dtype,
                )
                v = torch.randn(
                    config["batch_size"],
                    config["num_heads"],
                    config["seq_len"],
                    config["head_dim"],
                    device=device,
                    dtype=dtype,
                )
            else:  # bshd
                k = torch.randn(
                    config["batch_size"],
                    config["seq_len"],
                    config["num_heads"],
                    config["head_dim"],
                    device=device,
                    dtype=dtype,
                )
                v = torch.randn(
                    config["batch_size"],
                    config["seq_len"],
                    config["num_heads"],
                    config["head_dim"],
                    device=device,
                    dtype=dtype,
                )
            kv_seqlen = config["seq_len"]

    if "attn_bias" in attn_config and attn_config["attn_bias"]:
        random_mask = torch.randint(
            0,
            2,
            (batch_size, num_heads, q_seqlen, kv_seqlen),
            dtype=torch.float32,
            device=device,
        )
        random_mask[random_mask == 1] = float("-inf")
        attn_bias = random_mask.to(dtype)

    return q, k, v, attn_bias


def benchmark_attn(
    name, func, attn_config, layout="bshd", device="cuda", dtype=torch.float16
):
    WARM_UP = 25
    REP = 100
    query, key, value, attn_bias = prepare_attn_input(
        attn_config, layout, device, dtype
    )

    if attn_bias is None:
        func = partial(func, query, key, value)
    else:
        func = partial(func, query, key, value, attn_bias)

    try:
        ms = triton.testing.do_bench(func, warmup=WARM_UP, rep=REP, return_mode="mean")

        tflops = calculate_tflops(
            ms * 1e-3,
            query.shape[0],
            query.shape[1] if layout == "bshd" else query.shape[2],
            key.shape[1] if layout == "bshd" else key.shape[2],
            query.shape[2] if layout == "bshd" else query.shape[1],
            query.shape[3],
        )
        print(f"{name:<30} \tMilisec={ms:.6f} \tTFLOPS={tflops:.6f}")

    except Exception as e:
        print(f"{name:<30} \tException: {get_error_message(e)}")
