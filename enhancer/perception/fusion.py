import numpy as np
from typing import Dict, Any


class MaskFusionEngine:
    """
    Mask Fusion & Semantic Priority Resolution Engine.
    Combines explicit model predictions into a coherent soft-layered scene representation,
    resolving overlaps probabilistically without reducing masks to crude binary rasters.
    """

    def fuse(
        self,
        matting_out: Dict[str, Any],
        human_out: Dict[str, Any],
        face_out: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Executes hierarchical soft fusion across adapter outputs.

        Returns:
            Dict[str, np.ndarray]: Cleaned, prioritized float32 masks in [0.0, 1.0].
        """
        subject_alpha = np.clip(matting_out.get("subject_alpha", np.zeros((1, 1), dtype=np.float32)), 0.0, 1.0)
        bg_alpha = np.clip(1.0 - subject_alpha, 0.0, 1.0)

        raw_skin = np.clip(human_out.get("raw_skin", np.zeros_like(subject_alpha)), 0.0, 1.0)
        raw_hair = np.clip(human_out.get("raw_hair", np.zeros_like(subject_alpha)), 0.0, 1.0)
        raw_clothing = np.clip(human_out.get("raw_clothing", np.zeros_like(subject_alpha)), 0.0, 1.0)
        raw_accessories = np.clip(human_out.get("raw_accessories", np.zeros_like(subject_alpha)), 0.0, 1.0)
        raw_droplets = np.clip(human_out.get("raw_droplets", np.zeros_like(subject_alpha)), 0.0, 1.0)
        raw_face = np.clip(face_out.get("raw_face", np.zeros_like(subject_alpha)), 0.0, 1.0)

        # -------------------------------------------------------------
        # 1. Subject Containment Gating
        # -------------------------------------------------------------
        # All anatomical and apparel features must belong strictly to the subject silhouette
        skin_fused = np.clip(raw_skin * subject_alpha, 0.0, 1.0)
        face_fused = np.clip(raw_face * subject_alpha, 0.0, 1.0)

        # 2. Hair Priority: Hair attenuates where skin is confident, but hair over face/skin is preserved as soft strands
        hair_fused = np.clip(raw_hair * subject_alpha * (1.0 - skin_fused * 0.4), 0.0, 1.0)

        # 3. Clothing Priority: Clothing respects exposed skin
        clothing_fused = np.clip(raw_clothing * subject_alpha * (1.0 - skin_fused * 0.75), 0.0, 1.0)

        # 4. Accessories Priority: Bow and garter hardware
        accessories_fused = np.clip(raw_accessories * subject_alpha, 0.0, 1.0)

        # 5. Surface Modifiers: Water droplets
        droplets_fused = np.clip(raw_droplets * subject_alpha, 0.0, 1.0)

        return {
            "subject": subject_alpha,
            "background": bg_alpha,
            "skin": skin_fused,
            "face": face_fused,
            "hair": hair_fused,
            "clothing": clothing_fused,
            "accessories": accessories_fused,
            "droplets": droplets_fused,
            "sheer_layer": human_out.get("raw_sheer", np.zeros_like(subject_alpha)),
            "lace_layer": human_out.get("raw_lace", np.zeros_like(subject_alpha)),
        }


mask_fusion_engine = MaskFusionEngine()
