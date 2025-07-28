import platform
import torch
import os

def get_system_info(device, cpu_count=None):
    return {
        "device": str(device),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "cpu_count": cpu_count if cpu_count is not None else os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
