import os
import io
import numpy as np
from PIL import Image
from typing import Dict, Any


class DebugArtifactExporter:
    """
    Developer / Debug Artifact Exporter.
    Saves isolated float32 semantic masks, boundary transition maps,
    and composite overlays as PNG files to a specified output directory.
    """

    @staticmethod
    def export_masks_to_directory(
        fused_masks: Dict[str, np.ndarray],
        boundaries: Dict[str, Dict[str, Any]],
        confidence_map: np.ndarray,
        composite_rgb: np.ndarray,
        output_dir: str = "outputs/enhancer_debug",
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}

        def _save_mask(mask_arr: np.ndarray, filename: str) -> str:
            uint8_data = np.clip(mask_arr * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(uint8_data, mode="L")
            full_path = os.path.join(output_dir, filename)
            img.save(full_path, format="PNG")
            return full_path

        # Core semantic masks
        saved_paths["subject_mask"] = _save_mask(fused_masks["subject"], "subject_mask.png")
        saved_paths["background_mask"] = _save_mask(fused_masks["background"], "background_mask.png")
        saved_paths["skin_mask"] = _save_mask(fused_masks["skin"], "skin_mask.png")
        saved_paths["clothing_mask"] = _save_mask(fused_masks["clothing"], "clothing_mask.png")
        saved_paths["hair_mask"] = _save_mask(fused_masks["hair"], "hair_mask.png")
        saved_paths["face_mask"] = _save_mask(fused_masks["face"], "face_mask.png")
        saved_paths["accessory_mask"] = _save_mask(fused_masks["accessories"], "accessory_mask.png")

        # Confidence & Boundaries
        saved_paths["confidence_map"] = _save_mask(confidence_map, "confidence_map.png")
        if "skin_clothing" in boundaries:
            saved_paths["boundary_map"] = _save_mask(boundaries["skin_clothing"]["mask"], "boundary_map.png")

        # Multi-color composite
        comp_img = Image.fromarray(np.clip(composite_rgb, 0, 255).astype(np.uint8), mode="RGB")
        comp_path = os.path.join(output_dir, "composite_overlay.png")
        comp_img.save(comp_path, format="PNG")
        saved_paths["composite_overlay"] = comp_path

        return saved_paths


debug_exporter = DebugArtifactExporter()
