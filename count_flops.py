import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count


def flops_formulate(batch_size, sequence_length, num_heads, head_dim):
    # FLOPs for self-attention = 4 * batch_size * sequence_length^2 * num_heads * head_dim
    return 4 * batch_size * (sequence_length ** 2) * num_heads * head_dim


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 24
    num_heads = 4429
    head_dim = 64
    shapes = (batch_size, sequence_length, num_heads, head_dim)
    device = "cuda"

    # # b s h d 
    # q = torch.randn(batch_size, sequence_length, num_heads, head_dim, device=device, dtype=torch.float16)
    # k = torch.randn(batch_size, sequence_length, num_heads, head_dim, device=device, dtype=torch.float16)
    # v = torch.randn(batch_size, sequence_length, num_heads, head_dim, device=device, dtype=torch.float16)

    # b h s d
    q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device, dtype=torch.float16)

    print(flops_formulate(batch_size, sequence_length, num_heads, head_dim))
    print(sdpa_flop_count(q.shape, k.shape, v.shape))

    with FlopCounterMode(display=False) as counter:
        _ = F.scaled_dot_product_attention(q, k, v)
        # print(counter.get_flop_counts())
        print(counter.get_total_flops())
    
