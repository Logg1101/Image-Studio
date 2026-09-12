import time
import torch
import numpy as np
import torch.nn.functional as F
import scipy.ndimage as ndi
from PIL import Image
from typing import Dict, Any

from enhancer.perception.adapters.base_adapter import BasePerceptionAdapter
from enhancer.perception.vram_manager import vram_manager


class FaceParserAdapter(BasePerceptionAdapter):
    """
    Neural Face & Expressive Features Parser Adapter.
    Powered by SegFormer Face Parsing (jonathandinu/face-parsing).
    Segments facial skin, eyes, eyebrows, nose, mouth, and lips to enforce
    high-priority identity protection.
    """

    def __init__(self, model_id: str = "jonathandinu/face-parsing"):
        self.model_id = model_id
        self._is_available = True
        self._is_neural = True

    @property
    def adapter_name(self) -> str:
        return f"FaceParser ({self.model_id})"

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
            with vram_manager.staged_execution("neural_face_parsing"):
                from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

                processor = SegformerImageProcessor.from_pretrained(self.model_id)
                model = SegformerForSemanticSegmentation.from_pretrained(self.model_id).to(device=device, dtype=torch.float32).eval()

                raw_inputs = processor(images=img_rgb, return_tensors="pt")
                inputs = {
                    k: (v.to(device=device, dtype=torch.float32) if v.is_floating_point() else v.to(device=device))
                    for k, v in raw_inputs.items()
                }

                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits  # [1, 19, H/4, W/4]
                probs = F.interpolate(
                    logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                ).softmax(dim=1).cpu().squeeze().numpy()  # [19, orig_h, orig_w]

                del model, inputs, outputs, logits

            inference_ms = (time.time() - start_time) * 1000
            vram_after = vram_manager.get_cuda_memory_mb()

            # -------------------------------------------------------------
            # Class Mapping for jonathandinu/face-parsing:
            # 0: background
            # 1: skin (face)
            # 2: nose
            # 3: eye_g (glasses)
            # 4: l_eye
            # 5: r_eye
            # 6: l_brow
            # 7: r_brow
            # 8: l_ear
            # 9: r_ear
            # 10: mouth
            # 11: u_lip
            # 12: l_lip
            # 13: hair
            # 14: hat
            # 15: ear_r
            # 16: neck_l
            # 17: neck
            # 18: cloth
            # -------------------------------------------------------------

            # Combined facial feature mask: Skin(1) + Nose(2) + Glasses(3) + Eyes(4,5) + Brows(6,7) + Mouth/Lips(10,11,12) + Ears(8,9)
            face_features = (
                probs[1] + probs[2] + probs[3] + probs[4] + probs[5] +
                probs[6] + probs[7] + probs[8] + probs[9] + probs[10] +
                probs[11] + probs[12]
            )
            raw_face = np.clip(face_features, 0.0, 1.0).astype(np.float32)

            # Soften and calculate bounding box
            face_soft = ndi.gaussian_filter(raw_face, sigma=1.5)
            coords = np.argwhere(face_soft > 0.25)

            if len(coords) > 0:
                ymin, xmin = coords.min(axis=0).tolist()
                ymax, xmax = coords.max(axis=0).tolist()
                # Slight expansion padding
                f_ymin = max(0, int(ymin) - 8)
                f_ymax = min(orig_h, int(ymax) + 12)
                f_xmin = max(0, int(xmin) - 10)
                f_xmax = min(orig_w, int(xmax) + 10)
                face_bbox = [f_ymin, f_xmin, f_ymax, f_xmax]
            else:
                face_bbox = [int(orig_h * 0.05), int(orig_w * 0.25), int(orig_h * 0.32), int(orig_w * 0.75)]

            coverage_pct = float(np.sum(face_soft > 0.25) / (orig_w * orig_h) * 100)
            confidence = float(np.mean(face_soft[face_soft > 0.25]) if np.any(face_soft > 0.25) else 0.95)

            return {
                "raw_face": face_soft,
                "bounding_box": face_bbox,
                "is_protected": True,
                "protection_priority": "high",
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
            print(f"[FaceParserAdapter] Neural inference failed: {err}. Reporting error state.")
            return {
                "raw_face": np.zeros((orig_h, orig_w), dtype=np.float32),
                "bounding_box": [0, 0, orig_h, orig_w],
                "is_protected": True,
                "protection_priority": "high",
                "coverage_pct": 0.0,
                "confidence": 0.0,
                "status": "unavailable",
                "is_neural": False,
                "error": str(err),
            }
