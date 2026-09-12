import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

from enhancer.upscaling.base import BaseUpscalerAdapter
from enhancer.perception.vram_manager import vram_manager


class RealCUGANAdapter(BaseUpscalerAdapter):
    """
    Neural 4x Real-CUGAN (Cascaded U-Net) Anime & Digital Art Upscaler Adapter.
    Specialized for pristine vector-sharp lineart, clean fills, and artifact-free super-resolution.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            from config.paths import UPSCALERS_DIR
            self._model_path = str((UPSCALERS_DIR / "up4x-latest-no-denoise.pth").resolve())
        else:
            self._model_path = model_path
        self._scaler = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def name(self) -> str:
        return "4x Real-CUGAN (Anime & Illustration)"

    @property
    def model_id(self) -> str:
        return "4x-realcugan"

    @property
    def native_scale(self) -> int:
        return 4

    @property
    def is_available(self) -> bool:
        if os.path.exists(self._model_path):
            return True
        from config.paths import UPSCALERS_DIR
        alt = UPSCALERS_DIR / "up4x-latest-no-denoise.pth"
        if alt.exists():
            self._model_path = str(alt.resolve())
            return True
        return False

    def supports_scale(self, scale: int) -> bool:
        return scale in (1, 2, 4)

    def estimated_vram_mb(self) -> float:
        return 768.0

    def load(self, device: Optional[str] = None) -> None:
        if self._scaler is not None:
            return
        if not self.is_available:
            raise FileNotFoundError(f"Real-CUGAN weights not found: {self._model_path}")

        target_device = device or self._device
        from enhancer.upscaling.upcunet_v3 import RealWaifuUpScaler
        use_half = (target_device == "cuda")
        self._scaler = RealWaifuUpScaler(
            scale=4,
            weight_path=self._model_path,
            half=use_half,
            device=target_device,
        )
        self._device = target_device

    def unload(self) -> None:
        if self._scaler is not None:
            del self._scaler
            self._scaler = None
            vram_manager.clear_cache()

    def upscale(self, image_tensor: torch.Tensor, scale: int = 4) -> torch.Tensor:
        """
        Upscales input tensor [1, 3, H, W] in [0.0, 1.0].
        """
        if self._scaler is None:
            self.load()

        orig_device = image_tensor.device
        # Convert tensor [1, 3, H, W] to uint8 numpy array [H, W, 3]
        np_img = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Dynamic safe tile_mode: images smaller than 256px on shortest side run tile_mode=0
        safe_tile_mode = 0 if min(np_img.shape[0], np_img.shape[1]) < 256 else 2

        # Run Real-CUGAN
        with torch.no_grad():
            out_np = self._scaler(np_img, tile_mode=safe_tile_mode, cache_mode=0, alpha=1)

        # Convert back to torch tensor [1, 3, H*4, W*4]
        out_tensor = torch.from_numpy(out_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

        _, _, h, w = image_tensor.shape
        target_h, target_w = h * scale, w * scale

        if scale != 4:
            out_tensor = F.interpolate(
                out_tensor, size=(target_h, target_w), mode="bicubic", align_corners=False
            )

        return torch.clamp(out_tensor, 0.0, 1.0).to(orig_device)

    def upscale_pil(self, image: Image.Image, scale: int = 4) -> Image.Image:
        """
        Direct high-speed PIL upscaling with automatic scale interpolation.
        """
        if self._scaler is None:
            self.load()

        img_rgb = image.convert("RGB")
        np_img = np.array(img_rgb, dtype=np.uint8)

        safe_tile_mode = 0 if min(np_img.shape[0], np_img.shape[1]) < 256 else 2

        with torch.no_grad():
            out_np = self._scaler(np_img, tile_mode=safe_tile_mode, cache_mode=0, alpha=1)

        res_pil = Image.fromarray(out_np)
        target_w = int(image.width * scale)
        target_h = int(image.height * scale)

        if (res_pil.width, res_pil.height) != (target_w, target_h):
            res_pil = res_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)

        return res_pil

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "native_scale": self.native_scale,
            "supported_scales": [1, 2, 4],
            "is_available": self.is_available,
            "model_path": self._model_path,
            "architecture": "Real-CUGAN (UpCunet4x Cascaded U-Net)",
            "estimated_vram_mb": self.estimated_vram_mb(),
        }
