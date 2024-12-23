from pathlib import Path


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


def trace_handler_wrapper(target, save_dir=Path("./"), row_limit=10000):
    """Function to create a custom trace handler."""

    def trace_handler(prof):
        with open(save_dir / f"{target}.profile", "w") as f:
            table = prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=row_limit
            )
            f.write(str(table))
        prof.export_chrome_trace(str(save_dir / f"{target}.json"))

    return trace_handler
