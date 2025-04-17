import platform
import torch
import os

def get_system_info(device):
    return {
        "device": str(device),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
