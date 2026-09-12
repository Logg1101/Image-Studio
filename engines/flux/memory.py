import torch
import gc

def optimize_memory_for_12gb(pipeline) -> None:
    """
    Applies memory optimizations specific to 12GB VRAM cards (RTX 5070).
    Uses CPU offloading to stream model weights efficiently.
    """
    try:
        # This is the optimal offloading strategy for 12GB VRAM
        pipeline.enable_model_cpu_offload()
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        elif hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
    except AttributeError as e:
        print(f"Warning: Memory optimization unsupported by this pipeline version. {e}")

def clear_vram() -> None:
    """Forces garbage collection and clears the CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()