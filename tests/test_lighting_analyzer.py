import os
import unittest
import numpy as np
from PIL import Image

from enhancer.lighting.analyzer import lighting_analyzer
from enhancer.lighting.direction import light_direction_estimator
from enhancer.lighting.temperature import color_temperature_estimator
from enhancer.lighting.decomposition import lighting_decomposition_engine


class TestLightingAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(
            os.path.dirname(__file__), "assets", "reference_test_image.jpg"
        )
        if os.path.exists(asset_path):
            cls.test_image = Image.open(asset_path)
        else:
            cls.test_image = Image.new("RGB", (256, 256), color=(220, 190, 170))

    def test_lighting_profile_validity_and_normalization(self):
        """Lighting profile unit direction vector has length 1.0 and valid bounds."""
        res = lighting_analyzer.analyze(self.test_image, export_debug=False)
        prof = res.profile

        # 1. Angle in [0, 360)
        self.assertGreaterEqual(prof.direction_angle_deg, 0.0)
        self.assertLess(prof.direction_angle_deg, 360.0)

        # 2. Shadow angle in [0, 360)
        self.assertGreaterEqual(prof.shadow_direction_deg, 0.0)
        self.assertLess(prof.shadow_direction_deg, 360.0)

        # 3. Vector normalization
        vx, vy, vz = prof.direction_vector
        norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        self.assertAlmostEqual(norm, 1.0, places=3)

        # 4. Intensities within [0.0, 1.0]
        self.assertTrue(0.0 <= prof.key_intensity <= 1.0)
        self.assertTrue(0.0 <= prof.ambient_intensity <= 1.0)
        self.assertTrue(0.0 <= prof.shadow_strength <= 1.0)
        self.assertTrue(0.0 <= prof.global_confidence <= 1.0)

        # 5. Color temperature in [2200, 10000] Kelvin
        self.assertTrue(2200.0 <= prof.color_temperature_k <= 10000.0)
        self.assertIn(prof.color_bias, ["warm", "cool", "neutral"])

    def test_material_aware_lighting_metrics(self):
        """Material photometrics are computed for skin, face, hair, clothing, background."""
        res = lighting_analyzer.analyze(self.test_image, export_debug=False)
        mats = res.profile.material_lighting

        self.assertIn("subject", mats)
        self.assertIn("background", mats)
        self.assertIn("skin", mats)
        self.assertIn("hair", mats)
        self.assertIn("clothing", mats)
        self.assertIn("face", mats)

        # Face mean luminance and reflectance
        self.assertEqual(mats["face"].estimated_reflectance, "subsurface_delicate")
        self.assertTrue(0.0 <= mats["face"].mean_luminance <= 1.0)

        # Hair specular
        self.assertEqual(mats["hair"].estimated_reflectance, "anisotropic_specular")

    def test_deterministic_analysis(self):
        """Repeated analysis on the same image produces identical results."""
        res1 = lighting_analyzer.analyze(self.test_image, export_debug=False)
        res2 = lighting_analyzer.analyze(self.test_image, export_debug=False)

        self.assertAlmostEqual(
            res1.profile.direction_angle_deg, res2.profile.direction_angle_deg, places=2
        )
        self.assertAlmostEqual(
            res1.profile.color_temperature_k, res2.profile.color_temperature_k, places=1
        )
        self.assertAlmostEqual(
            res1.profile.key_intensity, res2.profile.key_intensity, places=3
        )

    def test_no_nan_or_inf_in_lighting_maps(self):
        """Lighting decomposition maps have no NaNs or Infs."""
        img_np = np.array(self.test_image, dtype=np.float32) / 255.0
        decomp = lighting_decomposition_engine.decompose(
            image_np=img_np,
            regions={"subject": np.ones((self.test_image.height, self.test_image.width))},
            boundaries={},
        )

        for map_name in ["luminance_map", "shadows_map", "highlights_map", "ambient_map", "confidence_map"]:
            arr = decomp[map_name]
            self.assertFalse(np.any(np.isnan(arr)))
            self.assertFalse(np.any(np.isinf(arr)))
            self.assertTrue(np.all(arr >= 0.0))
            self.assertTrue(np.all(arr <= 1.0))


if __name__ == "__main__":
    unittest.main()
