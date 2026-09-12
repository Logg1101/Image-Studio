import time
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple

from enhancer.types import SemanticAnalysisResult
from enhancer.analyzer.semantic_analyzer import SemanticAnalyzer
from enhancer.lighting.types import (
    LightingProfile,
    LightingMaps,
    LightingAnalysisResult,
    RegionLightingMetrics,
)
from enhancer.lighting.direction import light_direction_estimator
from enhancer.lighting.decomposition import lighting_decomposition_engine
from enhancer.lighting.temperature import color_temperature_estimator
from enhancer.lighting.debug_exporter import lighting_debug_exporter


class LightingAnalyzer:
    """
    Primary Phase 4 Semantic-Aware Lighting Analysis Engine.
    Estimates 3D light vectors, canvas source position, color temperature,
    material reflectance metrics, and 2D spatial lighting decomposition channels.
    """

    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.direction_estimator = light_direction_estimator
        self.decomposition_engine = lighting_decomposition_engine
        self.temperature_estimator = color_temperature_estimator
        self.debug_exporter = lighting_debug_exporter

    def _extract_mask_arrays(
        self, semantic_res: SemanticAnalysisResult, h: int, w: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        """Extracts float [0.0, 1.0] numpy masks from SemanticAnalysisResult."""
        import base64, io

        def _decode(b64_str: str) -> np.ndarray:
            if "base64," in b64_str:
                b64_str = b64_str.split("base64,")[1]
            raw = base64.b64decode(b64_str)
            p = Image.open(io.BytesIO(raw)).convert("L")
            if p.size != (w, h):
                p = p.resize((w, h), Image.Resampling.BILINEAR)
            return np.array(p, dtype=np.float32) / 255.0

        region_arrays = {
            name: _decode(reg.mask_b64) for name, reg in semantic_res.regions.items()
        }
        boundary_arrays = {
            name: _decode(b.mask_b64) for name, b in semantic_res.boundaries.items()
        }
        conf_arr = _decode(semantic_res.confidence_map_b64)

        return region_arrays, boundary_arrays, conf_arr

    def analyze(
        self,
        image: Image.Image,
        semantic_result: Optional[SemanticAnalysisResult] = None,
        export_debug: bool = True,
        output_dir: str = "outputs/enhancer_debug/lighting",
    ) -> LightingAnalysisResult:
        """
        Performs full semantic lighting analysis on input image.

        Args:
            image (Image.Image): Input PIL RGB image.
            semantic_result (Optional[SemanticAnalysisResult]): Pre-computed semantic analysis or None.
            export_debug (bool): If True, saves debug PNGs to disk.
            output_dir (str): Destination directory for debug exports.

        Returns:
            LightingAnalysisResult: LightingProfile and 2D spatial LightingMaps.
        """
        start_time = time.time()
        img_rgb = image.convert("RGB")
        w, h = img_rgb.size
        img_np = np.array(img_rgb, dtype=np.float32) / 255.0

        # 1. Semantic Analysis
        if semantic_result is None:
            semantic_result = self.semantic_analyzer.analyze(
                img_rgb, export_debug=False
            )

        region_masks, boundary_masks, base_conf = self._extract_mask_arrays(
            semantic_result, h, w
        )

        # 2. Lighting Decomposition & Material Photometrics
        decomp = self.decomposition_engine.decompose(
            image_np=img_np,
            regions=region_masks,
            boundaries=boundary_masks,
            base_confidence=base_conf,
        )

        lum = decomp["luminance_map"]
        shadows = decomp["shadows_map"]
        highlights = decomp["highlights_map"]
        ambient = decomp["ambient_map"]
        lighting_conf = decomp["confidence_map"]

        # 3. 3D Light Direction & Source Position Estimation
        dir_res = self.direction_estimator.estimate_direction(
            luminance=lum,
            subject_mask=region_masks.get("subject"),
            face_mask=region_masks.get("face"),
        )

        # 4. Color Temperature & Chromaticity Estimation
        neutral_mask = (
            region_masks.get("background", np.ones((h, w), dtype=np.float32)) * 0.7
            + region_masks.get("skin", np.zeros((h, w), dtype=np.float32)) * 0.3
        )
        temp_res = self.temperature_estimator.estimate_temperature(
            image_np=img_np, neutral_mask=neutral_mask
        )

        # 5. Assemble LightingProfile
        profile = LightingProfile(
            direction_angle_deg=dir_res["direction_angle_deg"],
            direction_vector=dir_res["direction_vector"],
            light_source_position=dir_res["light_source_position"],
            key_intensity=decomp["key_intensity"],
            ambient_intensity=decomp["ambient_intensity"],
            shadow_strength=decomp["shadow_strength"],
            shadow_direction_deg=dir_res["shadow_direction_deg"],
            color_temperature_k=temp_res["color_temperature_k"],
            color_bias=temp_res["color_bias"],
            color_tint_rgb=temp_res["color_tint_rgb"],
            global_confidence=decomp["global_confidence"],
            material_lighting=decomp["material_lighting"],
        )

        # 6. Export Debug Maps & Generate Base64
        maps = self.debug_exporter.export_and_encode(
            base_rgb=img_np,
            luminance=lum,
            shadows=shadows,
            highlights=highlights,
            ambient=ambient,
            confidence=lighting_conf,
            profile=profile,
            output_dir=output_dir,
            save_disk=export_debug,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return LightingAnalysisResult(
            profile=profile,
            maps=maps,
            analysis_time_ms=elapsed_ms,
            version="4.0.0",
        )


lighting_analyzer = LightingAnalyzer()
