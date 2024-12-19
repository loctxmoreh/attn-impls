# Attn impls


## Setup on MI250/MI300X

```bash
conda create -n attn-rocm python=3.10 -y && conda activate attn-rocm 
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/ROCm/flash-attention.git


# Some extra dependecies
pip install transformers diffusers accelerate sentencepiece protobuf tabulate
pip install pytest matplotlib pandas
```

## Setup on A100/H100

```bash 
conda create -n attn-cuda python=3.10 -y && conda activate attn-cuda 
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 triton xformers --index-url https://download.pytorch.org/whl/cu124 # cu121 for A100

# install flash-attn
pip install --no-build-isolation git+https://github.com/Dao-AILab/flash-attention.git

# istall flash-attn 3 (H100 only)
pip clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper 
pip setup.py install

pip install transformers diffusers accelerate sentencepiece protobuf
pip install pytest matplotlib pandas
```
