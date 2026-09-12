import torch
from dataclasses import dataclass
from typing import Optional

def get_torch_device() -> torch.device:
    """
    Returns the fastest available hardware compute device in order:
    CUDA (NVIDIA / AMD ROCm on Linux) -> DirectML (AMD/Intel on Windows) -> MPS (Apple Silicon) -> XPU (Intel) -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

def get_torch_dtype(device: Optional[torch.device] = None) -> torch.dtype:
    """Returns optimal precision: float16 for accelerators, float32 for CPU."""
    dev = device or get_torch_device()
    return torch.float32 if dev.type == "cpu" else torch.float16

@dataclass
class DeviceInfo:
    cuda_available: bool
    name: str
    vram_gb: float
    cuda_index: int
    torch_version: str
    cuda_version: str

def get_device_info() -> DeviceInfo:
    dev = get_torch_device()
    torch_version = torch.__version__
    
    if dev.type == "cuda":
        cuda_index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(cuda_index)
        try:
            vram_bytes = torch.cuda.get_device_properties(cuda_index).total_memory
            vram_gb = vram_bytes / (1024**3)
        except Exception:
            vram_gb = 0.0
        cuda_version = torch.version.cuda or "ROCm/CUDA"
        cuda_available = True
    elif dev.type == "mps":
        cuda_index = 0
        name = "Apple Silicon (MPS)"
        vram_gb = 0.0
        cuda_version = "MPS"
        cuda_available = False
    elif dev.type == "xpu":
        cuda_index = 0
        name = getattr(torch.xpu, "get_device_name", lambda _: "Intel XPU")(0)
        vram_gb = 0.0
        cuda_version = "XPU"
        cuda_available = False
    else:
        try:
            import torch_directml
            if dev == torch_directml.device():
                name = f"AMD/DirectML ({torch_directml.device_name(0)})"
                return DeviceInfo(
                    cuda_available=False,
                    name=name,
                    vram_gb=0.0,
                    cuda_index=0,
                    torch_version=torch_version,
                    cuda_version="DirectML"
                )
        except Exception:
            pass
        cuda_index = -1
        name = "CPU"
        vram_gb = 0.0
        cuda_version = "N/A"
        cuda_available = False

    return DeviceInfo(
        cuda_available=cuda_available,
        name=name,
        vram_gb=vram_gb,
        cuda_index=cuda_index,
        torch_version=torch_version,
        cuda_version=cuda_version
    )