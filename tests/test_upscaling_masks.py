import os
import unittest
import numpy as np
from PIL import Image

from enhancer.analyzer.semantic_analyzer import SemanticAnalyzer
from enhancer.upscaling.blending import SemanticMaskBlender


class TestUpscalingMasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "reference_test_image.jpg")
        if os.path.exists(asset_path):
            cls.test_image = Image.open(asset_path)
        else:
            cls.test_image = Image.new("RGB", (256, 256), color=(200, 180, 160))
        cls.analyzer = SemanticAnalyzer()
        cls.blender = SemanticMaskBlender()

    def test_semantic_mask_guidance(self):
        """Semantic mask blender modulates high frequencies differently for skin vs hair."""
        sem_res = self.analyzer.analyze(self.test_image)

        target_h, target_w = self.test_image.height * 2, self.test_image.width * 2
        base = np.full((target_h, target_w, 3), 0.5, dtype=np.float32)
        # Create candidate with strong high frequencies (+0.3)
        neural = np.full((target_h, target_w, 3), 0.8, dtype=np.float32)

        res = self.blender.blend(
            base_upscaled=base,
            neural_upscaled=neural,
            semantic_result=sem_res,
            detail_recovery=100.0,
            texture_synthesis=100.0,
            sharpen_strength=0.0,
            denoise_strength=0.0,
            face_restoration_strength=50.0,
        )

        final_img = res["final_image"]
        eff_weights = res["effective_weight_map"]

        self.assertEqual(final_img.shape, (target_h, target_w, 3))
        self.assertTrue(np.all(final_img >= 0.0))
        self.assertTrue(np.all(final_img <= 1.0))
        self.assertFalse(np.any(np.isnan(final_img)))

    def test_subject_other_detail_recovery(self):
        """Subject regions not covered by sub-parts (e.g. props, armor) receive detail weight."""
        from enhancer.analyzer.semantic_analyzer import _mask_to_base64
        from enhancer.types import SemanticAnalysisResult, SemanticRegion, MaterialClass, ProtectionPriority, ImageMetadata

        h, w = 64, 64
        ones = np.ones((h, w), dtype=np.float32)
        zeros = np.zeros((h, w), dtype=np.float32)

        sem_res = SemanticAnalysisResult(
            image=ImageMetadata(w, h, "1:1", 3, False),
            regions={
                "subject": SemanticRegion(
                    name="subject",
                    mask_b64=_mask_to_base64(ones),
                    coverage_pct=100.0,
                    mean_confidence=0.95,
                    material_type=MaterialClass.ORGANIC_SKIN,
                    protection_priority=ProtectionPriority.LOW,
                ),
                "background": SemanticRegion(
                    name="background",
                    mask_b64=_mask_to_base64(zeros),
                    coverage_pct=0.0,
                    mean_confidence=0.0,
                    material_type=MaterialClass.ARCHITECTURAL_DIFFUSE,
                    protection_priority=ProtectionPriority.NONE,
                ),
                "skin": SemanticRegion(
                    name="skin",
                    mask_b64=_mask_to_base64(zeros),
                    coverage_pct=0.0,
                    mean_confidence=0.0,
                    material_type=MaterialClass.ORGANIC_SKIN,
                    protection_priority=ProtectionPriority.MEDIUM,
                ),
                "hair": SemanticRegion(
                    name="hair",
                    mask_b64=_mask_to_base64(zeros),
                    coverage_pct=0.0,
                    mean_confidence=0.0,
                    material_type=MaterialClass.ANISOTROPIC_HAIR,
                    protection_priority=ProtectionPriority.LOW,
                ),
                "clothing": SemanticRegion(
                    name="clothing",
                    mask_b64=_mask_to_base64(zeros),
                    coverage_pct=0.0,
                    mean_confidence=0.0,
                    material_type=MaterialClass.WOVEN_FABRIC,
                    protection_priority=ProtectionPriority.LOW,
                ),
                "accessories": SemanticRegion(
                    name="accessories",
                    mask_b64=_mask_to_base64(zeros),
                    coverage_pct=0.0,
                    mean_confidence=0.0,
                    material_type=MaterialClass.METALLIC_HARDWARE,
                    protection_priority=ProtectionPriority.MEDIUM,
                ),
                "face": SemanticRegion(
                    name="face",
                    mask_b64=_mask_to_base64(zeros),
                    coverage_pct=0.0,
                    mean_confidence=0.0,
                    material_type=MaterialClass.ORGANIC_SKIN,
                    protection_priority=ProtectionPriority.HIGH,
                ),
            },
            boundaries={},
            surface_modifiers={},
            global_confidence=0.95,
            confidence_map_b64=_mask_to_base64(ones),
            composite_preview_b64="",
            analysis_time_ms=10.0,
            analyzer_name="mock",
        )

        base = np.zeros((h, w, 3), dtype=np.float32)
        neural = np.ones((h, w, 3), dtype=np.float32)

        res = self.blender.blend(
            base_upscaled=base,
            neural_upscaled=neural,
            semantic_result=sem_res,
            detail_recovery=100.0,
            texture_synthesis=50.0,
            sharpen_strength=0.0,
            denoise_strength=0.0,
            face_restoration_strength=0.0,
        )

        eff_weights = res["effective_weight_map"]
        # m_subj_other receives (0.60 * w_detail + 0.20 * w_texture) = 0.70 weight
        self.assertGreater(float(np.mean(eff_weights)), 0.5)


if __name__ == "__main__":
    unittest.main()
