import os
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from enhancer.upscaling.base import BaseUpscalerAdapter
from enhancer.upscaling.rrdbnet import RRDBNet
from enhancer.perception.vram_manager import vram_manager


class NMKDSuperscaleAdapter(BaseUpscalerAdapter):
    """
    Neural 4x NMKD Superscale RRDBNet Upscaler Adapter.
    Specialized for smooth art, realistic textures, and noise artifact reduction.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            from config.paths import UPSCALERS_DIR
            self._model_path = str((UPSCALERS_DIR / "4x_NMKD-Superscale-SP_178000_G.pth").resolve())
        else:
            self._model_path = model_path
        self._model: Optional[RRDBNet] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def name(self) -> str:
        return "4x-NMKD Superscale (Neural ESRGAN)"

    @property
    def model_id(self) -> str:
        return "4x-nmkd-superscale"

    @property
    def native_scale(self) -> int:
        return 4

    @property
    def is_available(self) -> bool:
        if os.path.exists(self._model_path):
            return True
        from config.paths import UPSCALERS_DIR
        alt_path = UPSCALERS_DIR / "4x_NMKD-Superscale-SP_178000_G.pth"
        if alt_path.exists():
            self._model_path = str(alt_path.resolve())
            return True
        return False

    def supports_scale(self, scale: int) -> bool:
        return scale in (1, 2, 4)

    def estimated_vram_mb(self) -> float:
        return 512.0

    def load(self, device: Optional[str] = None) -> None:
        if self._model is not None:
            return
        if not self.is_available:
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

        target_device = device or self._device
        state = torch.load(self._model_path, map_location="cpu", weights_only=False)
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
        model.load_esrgan_state(state)
        model.to(target_device).eval()
        self._model = model
        self._device = target_device

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            vram_manager.clear_cache()

    def upscale(self, image_tensor: torch.Tensor, scale: int = 4) -> torch.Tensor:
        if self._model is None:
            self.load()

        orig_device = image_tensor.device
        img_dev = image_tensor.to(self._device)

        with torch.no_grad():
            out_4x = self._model(img_dev)

        out_4x = torch.clamp(out_4x, 0.0, 1.0)

        _, _, h, w = image_tensor.shape
        target_h, target_w = h * scale, w * scale

        if scale != 4:
            out_scaled = F.interpolate(
                out_4x, size=(target_h, target_w), mode="bicubic", align_corners=False
            )
            out_scaled = torch.clamp(out_scaled, 0.0, 1.0)
            return out_scaled.to(orig_device)

        return out_4x.to(orig_device)

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "native_scale": self.native_scale,
            "supported_scales": [1, 2, 4],
            "is_available": self.is_available,
            "model_path": self._model_path,
            "architecture": "RRDBNet (64nf, 23nb, 32gc)",
            "estimated_vram_mb": self.estimated_vram_mb(),
        }
