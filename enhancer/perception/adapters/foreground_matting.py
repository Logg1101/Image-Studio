import time
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any

from enhancer.perception.adapters.base_adapter import BasePerceptionAdapter
from enhancer.perception.vram_manager import vram_manager


class ForegroundMattingAdapter(BasePerceptionAdapter):
    """
    Neural Foreground Matting Adapter powered by BiRefNet.
    Generates high-resolution, sub-pixel soft alpha masks separating Subject from Background.
    """

    def __init__(self, model_id: str = "ZhengPeng7/BiRefNet"):
        self.model_id = model_id
        self._is_available = True
        self._is_neural = True

    @property
    def adapter_name(self) -> str:
        return f"BiRefNet ({self.model_id})"

    @property
    def is_available(self) -> bool:
        return self._is_available

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        vram_before = vram_manager.get_cuda_memory_mb()
        orig_w, orig_h = image.size
        img_rgb = image.convert("RGB")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            with vram_manager.staged_execution("neural_foreground_matting"):
                from transformers import AutoModelForImageSegmentation

                # Load model to device in float32
                model = AutoModelForImageSegmentation.from_pretrained(
                    self.model_id, trust_remote_code=True
                ).to(device=device, dtype=torch.float32).eval()

                transform = T.Compose([
                    T.Resize((1024, 1024)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                input_tensor = transform(img_rgb).unsqueeze(0).to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    preds = model(input_tensor)[-1].sigmoid().cpu()

                # Interpolate to original image resolution
                alpha_tensor = F.interpolate(
                    preds.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False
                ).squeeze()

                alpha_mask = np.clip(alpha_tensor.numpy(), 0.0, 1.0).astype(np.float32)

                # Explicitly unload
                del model, input_tensor, preds, alpha_tensor

            inference_ms = (time.time() - start_time) * 1000
            vram_after = vram_manager.get_cuda_memory_mb()

            # Calculate bounding box
            coords = np.argwhere(alpha_mask > 0.3)
            if len(coords) > 0:
                ymin, xmin = coords.min(axis=0).tolist()
                ymax, xmax = coords.max(axis=0).tolist()
                bbox = [int(ymin), int(xmin), int(ymax), int(xmax)]
            else:
                bbox = [0, 0, orig_h, orig_w]

            coverage_pct = float(np.sum(alpha_mask > 0.3) / (orig_w * orig_h) * 100)
            confidence = float(np.mean(alpha_mask[alpha_mask > 0.3]) if np.any(alpha_mask > 0.3) else 0.95)

            return {
                "subject_alpha": alpha_mask,
                "bounding_box": bbox,
                "coverage_pct": coverage_pct,
                "confidence": confidence,
                "status": "available",
                "is_neural": True,
                "model_name": self.adapter_name,
                "device": device,
                "inference_time_ms": inference_ms,
                "vram_before": vram_before,
                "vram_after": vram_after,
            }

        except Exception as err:
            print(f"[ForegroundMattingAdapter] Neural inference failed: {err}. Reporting error state.")
            return {
                "subject_alpha": np.zeros((orig_h, orig_w), dtype=np.float32),
                "bounding_box": [0, 0, orig_h, orig_w],
                "coverage_pct": 0.0,
                "confidence": 0.0,
                "status": "unavailable",
                "is_neural": False,
                "error": str(err),
            }
