import time
import torch
import numpy as np
import torch.nn.functional as F
import scipy.ndimage as ndi
from PIL import Image
from typing import Dict, Any

from enhancer.perception.adapters.base_adapter import BasePerceptionAdapter
from enhancer.perception.vram_manager import vram_manager


class HumanParserAdapter(BasePerceptionAdapter):
    """
    Neural Human Body & Apparel Semantic Parser Adapter.
    Powered by SegFormer B2 Clothes (mattmdjaga/segformer_b2_clothes).
    Decodes multi-class probability maps for skin (limbs/torso), hair, upper clothes,
    skirt, dress, pants, accessories, and sub-textiles.
    """

    def __init__(self, model_id: str = "mattmdjaga/segformer_b2_clothes"):
        self.model_id = model_id
        self._is_available = True
        self._is_neural = True

    @property
    def adapter_name(self) -> str:
        return f"SegFormer-Clothes ({self.model_id})"

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
            with vram_manager.staged_execution("neural_human_parsing"):
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

                logits = outputs.logits  # [1, 18, H/4, W/4]
                probs = F.interpolate(
                    logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                ).softmax(dim=1).cpu().squeeze().numpy()  # [18, orig_h, orig_w]

                # Explicitly unload
                del model, inputs, outputs, logits

            inference_ms = (time.time() - start_time) * 1000
            vram_after = vram_manager.get_cuda_memory_mb()

            # -------------------------------------------------------------
            # Class Mapping for mattmdjaga/segformer_b2_clothes:
            # 0: Background
            # 1: Hat
            # 2: Hair
            # 3: Sunglasses
            # 4: Upper-clothes
            # 5: Skirt
            # 6: Pants
            # 7: Dress
            # 8: Belt
            # 9: Left-shoe
            # 10: Right-shoe
            # 11: Face (Skin)
            # 12: Left-leg
            # 13: Right-leg
            # 14: Left-arm
            # 15: Right-arm
            # 16: Bag
            # 17: Scarf
            # -------------------------------------------------------------

            # 1. Skin & Anatomy (Face skin + Limbs)
            raw_skin = probs[11] + probs[12] + probs[13] + probs[14] + probs[15]
            raw_skin = np.clip(raw_skin, 0.0, 1.0).astype(np.float32)

            # 2. Hair Mass
            raw_hair = np.clip(probs[2], 0.0, 1.0).astype(np.float32)

            # 3. Clothing & Apparel (Upper clothes + Skirt + Pants + Dress + Belt)
            raw_clothing = probs[4] + probs[5] + probs[6] + probs[7] + probs[8]
            raw_clothing = np.clip(raw_clothing, 0.0, 1.0).astype(np.float32)

            # 4. Accessories (Hat + Sunglasses + Bag + Scarf)
            raw_accessories = probs[1] + probs[3] + probs[16] + probs[17]
            raw_accessories = np.clip(raw_accessories, 0.0, 1.0).astype(np.float32)

            # Sub-layers: Sheer wet shirt & Stocking lace
            # Sheer layer: overlap of upper-clothes (4) and skin probability
            raw_sheer = np.clip(probs[4] * (raw_skin + 0.3), 0.0, 1.0).astype(np.float32)

            # Stocking lace: top section of leg class (12, 13)
            raw_lace = np.clip((probs[12] + probs[13]) * 0.8, 0.0, 1.0).astype(np.float32)

            # Surface Modifiers: Specular highlights
            np_rgb = np.array(img_rgb, dtype=np.float32) / 255.0
            v_chan = np.max(np_rgb, axis=-1)
            grad_v = ndi.gaussian_gradient_magnitude(v_chan, sigma=1.0)
            droplets = (v_chan > 0.92) & (grad_v > 0.15)
            raw_droplets = ndi.gaussian_filter(droplets.astype(np.float32), sigma=0.6)

            return {
                "raw_skin": raw_skin,
                "raw_hair": raw_hair,
                "raw_clothing": raw_clothing,
                "raw_accessories": raw_accessories,
                "raw_sheer": raw_sheer,
                "raw_lace": raw_lace,
                "raw_droplets": raw_droplets,
                "status": "available",
                "is_neural": True,
                "model_name": self.adapter_name,
                "device": device,
                "inference_time_ms": inference_ms,
                "vram_before": vram_before,
                "vram_after": vram_after,
            }

        except Exception as err:
            print(f"[HumanParserAdapter] Neural inference failed: {err}. Reporting error state.")
            return {
                "raw_skin": np.zeros((orig_h, orig_w), dtype=np.float32),
                "raw_hair": np.zeros((orig_h, orig_w), dtype=np.float32),
                "raw_clothing": np.zeros((orig_h, orig_w), dtype=np.float32),
                "raw_accessories": np.zeros((orig_h, orig_w), dtype=np.float32),
                "raw_sheer": np.zeros((orig_h, orig_w), dtype=np.float32),
                "raw_lace": np.zeros((orig_h, orig_w), dtype=np.float32),
                "raw_droplets": np.zeros((orig_h, orig_w), dtype=np.float32),
                "status": "unavailable",
                "is_neural": False,
                "error": str(err),
            }
