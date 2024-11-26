import torch


def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda


def is_rocm():
    return torch.cuda.is_available() and torch.version.hip


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability() == (9, 0)
