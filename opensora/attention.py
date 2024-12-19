import torch

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


def prepare_attn_input(attn_config, device="cuda", dtype=torch.float16):
    """Prepare q, k, v and attention bias in terms of attention config.

    Returns:
        q, k, v: Shape [batch_size, heads, seqlen, dim]
        attn_bias: Shape [q_seqlen, kv_seqlen]"""
    q, k, v = None, None, None
    attn_bias = None

    # Prepare qkv
    for name, config in attn_config.items():
        if "q" in name:
            q = torch.randn(
                config["batch_size"],
                config["num_heads"],
                config["seq_len"],
                config["head_dim"],
                device=device,
                dtype=dtype,
            )
            q_seqlen = config["seq_len"]
            batch_size = config["batch_size"]
            num_heads = config["num_heads"]

        if "kv" in name:
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
