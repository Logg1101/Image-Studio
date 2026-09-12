import gc
import torch

def clear_vram():
    """Aggressively purges GPU cache and uncollected Python objects across hardware."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
    gc.collect()

def get_vram_gb() -> float:
    """Returns total VRAM usage in GB if supported by hardware."""
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info(0)
            return (total - free) / (1024 ** 3)
        except Exception:
            pass
    return 0.0
