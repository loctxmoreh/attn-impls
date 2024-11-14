from functools import wraps

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import AttnProcessor2_0


def attn_pre_forward_hook(*args, **kwargs):
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            continue
        print(arg.shape)


def add_pre_call_hook(call, hook):
    @wraps(call)
    def wrapper(*args, **kwargs):
        hook(*args, **kwargs)
        return call(*args, **kwargs)

    return wrapper


AttnProcessor2_0.__call__ = add_pre_call_hook(AttnProcessor2_0.__call__, attn_pre_forward_hook)
# input shape [1, 512, 128, 128]

prompt = "Astronaut in a jungle"
model_name_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"
# model_name_or_path = "/home/share-mv/diffusion-models/stabilityai/stable-diffusion-3-medium"
pipe = StableDiffusion3Pipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to("cuda")
_ = pipe(prompt, negative_prompt="", num_inference_steps=10, guidance_scale=7.0, height=1024, width=1024)
