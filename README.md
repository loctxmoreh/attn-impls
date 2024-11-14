# Attn impls


## Setup 

```bash
conda create -n attn python=3.10 -y && conda activate attn 
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/ROCm/flash-attention.git


# Some extra dependecies
pip install transformers diffusers accelerate sentencepiece protobuf
pip install pytest matplotlib pandas
```
