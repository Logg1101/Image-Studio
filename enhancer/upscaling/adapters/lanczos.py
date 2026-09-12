import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from enhancer.upscaling.base import BaseUpscalerAdapter


class LanczosAdapter(BaseUpscalerAdapter):
    """
    Fast Mathematical Resampling Adapter.
    Zero GPU hallucination, instant bicubic/lanczos-approximation scaling.
    """

    @property
    def name(self) -> str:
        return "Lanczos Native (Mathematical)"

    @property
    def model_id(self) -> str:
        return "lanczos-native"

    @property
    def native_scale(self) -> int:
        return 1

    @property
    def is_available(self) -> bool:
        return True

    def supports_scale(self, scale: int) -> bool:
        return scale in (1, 2, 4)

    def estimated_vram_mb(self) -> float:
        return 0.0

    def load(self, device: Optional[str] = None) -> None:
        pass

    def unload(self) -> None:
        pass

    def upscale(self, image_tensor: torch.Tensor, scale: int = 1) -> torch.Tensor:
        if scale == 1:
            return image_tensor.clone()

        _, _, h, w = image_tensor.shape
        target_h, target_w = h * scale, w * scale
        return F.interpolate(
            image_tensor, size=(target_h, target_w), mode="bicubic", align_corners=False
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "native_scale": 1,
            "supported_scales": [1, 2, 4],
            "is_available": True,
            "model_path": "builtin",
            "architecture": "Bicubic / Lanczos Resampler",
            "estimated_vram_mb": 0.0,
        }
