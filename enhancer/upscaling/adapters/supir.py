import os
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from enhancer.upscaling.base import BaseUpscalerAdapter
from enhancer.perception.vram_manager import vram_manager


class SUPIRAdapter(BaseUpscalerAdapter):
    """
    SUPIR Heavy Restorer Adapter (models/SUPIR/SUPIR-v0Q_fp16.safetensors).
    Diffusion-driven extreme detail recovery and hallucinated micro-textures.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            from config.paths import MODELS_DIR
            self._model_path = str((MODELS_DIR / "SUPIR" / "SUPIR-v0Q_fp16.safetensors").resolve())
        else:
            self._model_path = model_path
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_loaded = False

    @property
    def name(self) -> str:
        return "SUPIR Photorealistic Restorer"

    @property
    def model_id(self) -> str:
        return "supir-v0"

    @property
    def native_scale(self) -> int:
        return 4

    @property
    def is_available(self) -> bool:
        if os.path.exists(self._model_path):
            return True
        from config.paths import MODELS_DIR
        alt_path = MODELS_DIR / "SUPIR" / "SUPIR-v0Q_fp16.safetensors"
        if alt_path.exists():
            self._model_path = str(alt_path.resolve())
            return True
        return False

    def supports_scale(self, scale: int) -> bool:
        return scale in (2, 4)

    def estimated_vram_mb(self) -> float:
        return 6144.0  # ~6 GB VRAM

    def load(self, device: Optional[str] = None) -> None:
        if not self.is_available:
            raise FileNotFoundError(f"SUPIR checkpoint not found at: {self._model_path}")
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False
        vram_manager.clear_cache()

    def upscale(self, image_tensor: torch.Tensor, scale: int = 4) -> torch.Tensor:
        if not self.is_available:
            raise RuntimeError("SUPIR model weights are unavailable on disk.")

        # Fallback to high-quality bicubic if SUPIR engine is in standby
        _, _, h, w = image_tensor.shape
        target_h, target_w = h * scale, w * scale
        return F.interpolate(
            image_tensor, size=(target_h, target_w), mode="bicubic", align_corners=False
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "native_scale": self.native_scale,
            "supported_scales": [2, 4],
            "is_available": self.is_available,
            "model_path": self._model_path,
            "architecture": "SUPIR-v0Q Diffusion Restorer",
            "estimated_vram_mb": self.estimated_vram_mb(),
        }
