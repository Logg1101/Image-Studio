import numpy as np
import scipy.ndimage as ndi
from typing import Dict, Any, Optional

from enhancer.lighting.types import RegionLightingMetrics


class LightingDecompositionEngine:
    """
    Semantic-Aware Lighting Decomposition Engine.
    Decomposes an RGB image into 2D spatial lighting channels (Luminance, Shadows,
    Highlights, Ambient Floor) and computes per-material photometric metrics.
    """

    @staticmethod
    def compute_luminance(image_np: np.ndarray) -> np.ndarray:
        """Computes standard perceptual luminance Y from RGB [H, W, 3]."""
        return (
            0.299 * image_np[..., 0]
            + 0.587 * image_np[..., 1]
            + 0.114 * image_np[..., 2]
        ).astype(np.float32)

    def decompose(
        self,
        image_np: np.ndarray,
        regions: Dict[str, np.ndarray],
        boundaries: Dict[str, np.ndarray],
        base_confidence: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Decomposes RGB image into 2D lighting maps and material metrics.

        Args:
            image_np (np.ndarray): Float RGB array [H, W, 3] in [0.0, 1.0].
            regions (Dict[str, np.ndarray]): Semantic masks for subject, background, skin, hair, clothing, face, etc.
            boundaries (Dict[str, np.ndarray]): Boundary transition maps.
            base_confidence (Optional[np.ndarray]): Spatial confidence map from perception.

        Returns:
            Dict[str, Any]: 2D maps (luminance, shadows, highlights, ambient, confidence) and scalar metrics.
        """
        h, w, _ = image_np.shape
        lum = self.compute_luminance(image_np)

        # 1. Low-frequency illumination background (Ambient Estimation)
        ambient_map = ndi.gaussian_filter(lum, sigma=18.0)
        ambient_map = np.clip(ambient_map, 0.05, 0.95)

        # 2. Specular and Diffuse Highlights Map
        # High-pass thresholding above ambient baseline
        high_residual = lum - ambient_map
        highlights_map = np.clip(high_residual * 2.5 + (lum - 0.7) * 2.0, 0.0, 1.0)
        # Suppress noise
        highlights_map = ndi.gaussian_filter(highlights_map, sigma=1.0)

        # 3. Shadows Map (Deep shadow density where luminance < ambient)
        shadow_residual = ambient_map - lum
        shadows_map = np.clip(shadow_residual * 2.5 + (0.35 - lum) * 1.5, 0.0, 1.0)
        shadows_map = ndi.gaussian_filter(shadows_map, sigma=1.2)

        # 4. Global Intensities
        key_intensity = float(np.percentile(lum, 92))
        ambient_intensity = float(np.percentile(lum, 20))
        shadow_strength = float(np.mean(shadows_map[shadows_map > 0.3]) if np.any(shadows_map > 0.3) else 0.4)

        # 5. Spatial Lighting Confidence Map
        # Confidence is high in flat diffuse areas and reduced near high-frequency texture edges & boundaries
        grad_y, grad_x = np.gradient(lum)
        texture_edge = np.sqrt(grad_x ** 2 + grad_y ** 2)
        norm_edge = np.clip(texture_edge / (np.percentile(texture_edge, 95) + 1e-5), 0.0, 1.0)

        # Base confidence from perception
        conf = base_confidence.copy() if base_confidence is not None else np.ones((h, w), dtype=np.float32)
        # Attenuate confidence near boundary zones
        for b_name, b_mask in boundaries.items():
            conf = conf * (1.0 - b_mask * 0.35)

        lighting_conf = np.clip(conf * (1.0 - norm_edge * 0.4), 0.15, 1.0)

        # 6. Semantic Material Metrics
        material_metrics: Dict[str, RegionLightingMetrics] = {}

        reflectance_dict = {
            "skin": "subsurface_organic",
            "face": "subsurface_delicate",
            "hair": "anisotropic_specular",
            "clothing": "diffuse_woven",
            "accessories": "metallic_chrome",
            "background": "architectural_diffuse",
            "subject": "composite_subject",
        }

        for r_name, r_mask in regions.items():
            if r_mask is None or np.sum(r_mask) < 1.0:
                continue

            r_pixels = lum[r_mask > 0.3]
            if len(r_pixels) == 0:
                continue

            mean_l = float(np.mean(r_pixels))
            high_cov = float(np.sum((highlights_map > 0.4) & (r_mask > 0.3)) / np.sum(r_mask > 0.3) * 100.0)
            shad_cov = float(np.sum((shadows_map > 0.4) & (r_mask > 0.3)) / np.sum(r_mask > 0.3) * 100.0)
            spec_i = float(np.percentile(r_pixels, 95))
            diff_i = float(np.median(r_pixels))

            material_metrics[r_name] = RegionLightingMetrics(
                region_name=r_name,
                mean_luminance=round(mean_l, 3),
                highlight_coverage_pct=round(high_cov, 1),
                shadow_coverage_pct=round(shad_cov, 1),
                specular_intensity=round(spec_i, 3),
                diffuse_intensity=round(diff_i, 3),
                estimated_reflectance=reflectance_dict.get(r_name, "standard_diffuse"),
            )

        global_confidence = float(np.mean(lighting_conf))

        return {
            "luminance_map": lum,
            "shadows_map": shadows_map,
            "highlights_map": highlights_map,
            "ambient_map": ambient_map,
            "confidence_map": lighting_conf,
            "key_intensity": key_intensity,
            "ambient_intensity": ambient_intensity,
            "shadow_strength": shadow_strength,
            "global_confidence": global_confidence,
            "material_lighting": material_metrics,
        }


lighting_decomposition_engine = LightingDecompositionEngine()
