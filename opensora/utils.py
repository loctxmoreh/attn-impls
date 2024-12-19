def get_error_message(exception):
    exception_str = str(exception).strip()
    if len(exception_str):
        return exception_str.split("\n")[-1].strip()
    else:
        return "Empty error message."


def calculate_tflops(latency, batch_size, q_seqlen, kv_seqlen, num_heads, head_dim):
    # FLOPs for attention = 4 * batch_size * q_seqlen * kv_seqlen * num_heads * head_dim
    flops = 4 * batch_size * (q_seqlen * kv_seqlen) * num_heads * head_dim
    tflops = flops / (latency * 10**12)  # Convert to TFLOPs
    return tflops
