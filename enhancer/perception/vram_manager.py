import gc
import torch
from contextlib import contextmanager
from typing import Generator, Dict, Any


class VRAMManager:
    """
    Subsystem-isolated VRAM Manager for perception models.
    Enforces staged, sequential GPU inference with immediate CPU offloading
    and cache release between model passes.
    """

    @staticmethod
    def get_cuda_memory_mb() -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2),
        }

    @staticmethod
    def clear_cache() -> None:
        """Purges PyTorch CUDA allocator cache and runs Python garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    @contextmanager
    def staged_execution(self, stage_name: str = "perception_stage") -> Generator[str, None, None]:
        """
        Context manager ensuring VRAM is cleanly reclaimed before and after a model inference stage.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clear_cache()
        try:
            yield device
        finally:
            self.clear_cache()


vram_manager = VRAMManager()
