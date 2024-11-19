# Attn impls


## Setup on MI250

```bash
conda create -n attn python=3.10 -y && conda activate attn 
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/ROCm/flash-attention.git


# Some extra dependecies
pip install transformers diffusers accelerate sentencepiece protobuf tabulate
pip install pytest matplotlib pandas
```

## Setup on A100

```bash 
conda create -n attn-cuda python=3.10 -y && conda activate attn-cuda 
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 triton xformers --index-url https://download.pytorch.org/whl/cu121

pip install --no-build-isolation git+https://github.com//flash-attention.git

pip install transformers diffusers accelerate sentencepiece protobuf
pip install pytest matplotlib pandas
```
