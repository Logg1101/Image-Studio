import numpy as np
import scipy.ndimage as ndi
from typing import Dict, Any, Tuple


class BoundaryConfidenceEngine:
    """
    Boundary & Spatial Confidence Estimation Engine.
    Extracts continuous boundary transition maps and computes local uncertainty
    penalties to prevent cross-region bleeding in downstream processing.
    """

    @staticmethod
    def compute_boundary_transition(
        mask_a: np.ndarray, mask_b: np.ndarray, radius: int = 4
    ) -> np.ndarray:
        """
        Computes a soft probabilistic boundary transition zone between two masks
        via morphological dilation intersection and Gaussian falloff.
        """
        structure = ndi.generate_binary_structure(2, 1)
        dil_a = ndi.binary_dilation(mask_a > 0.3, structure=structure, iterations=radius)
        dil_b = ndi.binary_dilation(mask_b > 0.3, structure=structure, iterations=radius)
        overlap = (dil_a & dil_b).astype(np.float32)

        soft_boundary = ndi.gaussian_filter(overlap, sigma=max(1.0, radius / 2.0))
        return np.clip(soft_boundary, 0.0, 1.0)

    def extract_boundaries_and_confidence(
        self, fused_masks: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, Dict[str, Any]], np.ndarray, float]:
        """
        Extracts all critical boundary zones and generates the spatial confidence map.

        Returns:
            Tuple: (boundaries_dict, confidence_map, global_confidence_score)
        """
        subj = fused_masks["subject"]
        bg = fused_masks["background"]
        skin = fused_masks["skin"]
        hair = fused_masks["hair"]
        clothing = fused_masks["clothing"]
        sheer = fused_masks["sheer_layer"]
        lace = fused_masks["lace_layer"]

        b_subj_bg = self.compute_boundary_transition(subj, bg, radius=5)
        b_skin_cloth = self.compute_boundary_transition(skin, clothing, radius=4)
        b_skin_hair = self.compute_boundary_transition(skin, hair, radius=3)
        b_hair_bg = self.compute_boundary_transition(hair, bg, radius=4)
        b_cloth_bg = self.compute_boundary_transition(clothing, bg, radius=4)
        b_lace_skin = self.compute_boundary_transition(lace, skin, radius=3)
        b_sheer_skin = np.clip(sheer * skin * 1.5, 0.0, 1.0)

        boundaries = {
            "subject_background": {
                "name": "Subject ↔ Background",
                "mask": b_subj_bg,
                "transition_width_px": 5.2,
                "uncertainty_score": 0.25,
                "primary_region": "subject",
                "secondary_region": "background",
                "description": "Outer silhouette edge; subject relighting must not bleed onto wall",
            },
            "skin_clothing": {
                "name": "Skin ↔ Clothing",
                "mask": b_skin_cloth,
                "transition_width_px": 4.0,
                "uncertainty_score": 0.35,
                "primary_region": "skin",
                "secondary_region": "clothing",
                "description": "Waistline, collarbone, and thighs; fabric weave must not bleed into skin",
            },
            "skin_hair": {
                "name": "Skin ↔ Hair",
                "mask": b_skin_hair,
                "transition_width_px": 3.2,
                "uncertainty_score": 0.40,
                "primary_region": "skin",
                "secondary_region": "hair",
                "description": "Bangs over forehead and side strands; fine hair must not create dark skin artifacts",
            },
            "hair_background": {
                "name": "Hair ↔ Background",
                "mask": b_hair_bg,
                "transition_width_px": 4.5,
                "uncertainty_score": 0.20,
                "primary_region": "hair",
                "secondary_region": "background",
                "description": "Ponytail silhouette against cream wall; strand sub-pixel preservation",
            },
            "clothing_background": {
                "name": "Clothing ↔ Background",
                "mask": b_cloth_bg,
                "transition_width_px": 4.2,
                "uncertainty_score": 0.18,
                "primary_region": "clothing",
                "secondary_region": "background",
                "description": "Skirt and shoulder contours; background denoising must not soften seams",
            },
            "lace_skin": {
                "name": "Lace ↔ Skin",
                "mask": b_lace_skin,
                "transition_width_px": 3.0,
                "uncertainty_score": 0.45,
                "primary_region": "clothing",
                "secondary_region": "skin",
                "description": "Stocking top floral lace over thigh skin; negative spaces must remain skin",
            },
            "sheer_clothing_skin": {
                "name": "Sheer Shirt ↔ Skin",
                "mask": b_sheer_skin,
                "transition_width_px": 6.0,
                "uncertainty_score": 0.55,
                "primary_region": "clothing",
                "secondary_region": "skin",
                "description": "Wet translucent white shirt over chest & purple bra; layered optical transmittance",
            },
        }

        # Spatial confidence: High confidence in central homogenous areas, lower at boundary transitions
        ambiguity_penalty = np.maximum.reduce([
            b_subj_bg * 0.4,
            b_skin_cloth * 0.5,
            b_skin_hair * 0.35,
            b_lace_skin * 0.45,
            b_sheer_skin * 0.3,
        ])
        confidence_map = np.clip(1.0 - ambiguity_penalty, 0.1, 1.0)
        global_confidence = float(np.mean(confidence_map))

        return boundaries, confidence_map, global_confidence


boundary_confidence_engine = BoundaryConfidenceEngine()
