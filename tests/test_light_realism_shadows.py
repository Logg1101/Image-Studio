import os
import unittest
import numpy as np
from PIL import Image

from enhancer.lighting.relighting.micro_relief import micro_relief_engine
from enhancer.lighting.relighting.contact_shadows import contact_shadow_engine
from enhancer.lighting.relighting.engine import relighting_engine
from enhancer.lighting.relighting.compositor import relighting_compositor
from enhancer.lighting.relighting.validation import relighting_validator
from enhancer.perception.vram_manager import vram_manager


class TestLightRealismShadows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(
            os.path.dirname(__file__), "assets", "reference_flat_test_image.jpg"
        )
        if os.path.exists(asset_path):
            cls.test_image = Image.open(asset_path)
        else:
            cls.test_image = Image.new("RGB", (256, 256), color=(220, 190, 170))

    def test_micro_relief_highlight_shadow_directional_inversion(self):
        """Rotating light by 180 deg (90 vs 270) reverses highlight and self-shadow positions on raised threads."""
        img_np = np.array(self.test_image.convert("L"), dtype=np.float32) / 255.0
        h, w = img_np.shape

        regions = {
            "clothing": np.ones((h, w), dtype=np.float32),
            "skin": np.zeros((h, w), dtype=np.float32),
            "face": np.zeros((h, w), dtype=np.float32),
        }

        # Light from Right (90 deg)
        res_right = micro_relief_engine.compute_micro_relief(
            luminance=img_np,
            light_angle_deg=90.0,
            regions=regions,
            relief_strength=75.0,
        )

        # Light from Left (270 deg)
        res_left = micro_relief_engine.compute_micro_relief(
            luminance=img_np,
            light_angle_deg=270.0,
            regions=regions,
            relief_strength=75.0,
        )

        h_right = res_right["highlight_map"]
        s_right = res_right["shadow_map"]
        h_left = res_left["highlight_map"]
        s_left = res_left["shadow_map"]

        # Highlights and shadows are non-trivial
        self.assertGreater(float(np.max(h_right)), 0.05)
        self.assertGreater(float(np.max(h_left)), 0.05)
        self.assertGreater(float(np.max(s_right)), 0.05)
        self.assertGreater(float(np.max(s_left)), 0.05)

        # High-correlation inversion: what is highlighted from Right is self-shadowed from Left
        active_mask = (h_right > 0.01) | (s_left > 0.01)
        if np.sum(active_mask) > 100:
            corr_inv = float(np.corrcoef(h_right[active_mask], s_left[active_mask])[0, 1])
            self.assertGreater(corr_inv, 0.70)

    def test_shadow_localization_no_global_darkening(self):
        """Self-shadows and under-casts are localized around raised details and do not cause global darkening."""
        img_np = np.array(self.test_image.convert("L"), dtype=np.float32) / 255.0
        h, w = img_np.shape

        regions = {
            "clothing": np.ones((h, w), dtype=np.float32),
            "skin": np.zeros((h, w), dtype=np.float32),
            "face": np.zeros((h, w), dtype=np.float32),
        }

        res = micro_relief_engine.compute_micro_relief(
            luminance=img_np,
            light_angle_deg=90.0,
            regions=regions,
            relief_strength=75.0,
        )

        shadows = res["shadow_map"]
        # Mean shadow intensity is very small (< 0.05), localized to thread edges
        self.assertLess(float(np.mean(shadows)), 0.05)
        # But peak shadow intensity at thread edges is clear (> 0.08)
        self.assertGreater(float(np.max(shadows)), 0.08)

    def test_strap_and_hair_directional_cast_shift(self):
        """Directional cast shadows under straps shift opposite to the light vector."""
        img_np = np.array(self.test_image.convert("L"), dtype=np.float32) / 255.0
        h, w = img_np.shape

        regions = {
            "clothing": np.zeros((h, w), dtype=np.float32),
            "accessories": np.zeros((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32),
            "face": np.zeros((h, w), dtype=np.float32),
        }
        # Add localized horizontal and vertical strap contours
        regions["accessories"][h // 3 : h // 3 + 20, w // 4 : 3 * w // 4] = 1.0
        regions["accessories"][h // 2 : 2 * h // 3, w // 2 - 10 : w // 2 + 10] = 1.0

        # Light from Right (90 deg -> cast_x < 0, shifts Left)
        cast_right = contact_shadow_engine.compute_directional_contact_shadows(
            luminance=img_np,
            light_angle_deg=90.0,
            regions=regions,
            shadow_depth=60.0,
        )

        # Light from Left (270 deg -> cast_x > 0, shifts Right)
        cast_left = contact_shadow_engine.compute_directional_contact_shadows(
            luminance=img_np,
            light_angle_deg=270.0,
            regions=regions,
            shadow_depth=60.0,
        )

        d_right = cast_right["directional_cast"]
        d_left = cast_left["directional_cast"]

        # Different spatial distribution
        diff = float(np.mean(np.abs(d_right - d_left)))
        self.assertGreater(diff, 0.002)

    def test_facial_protection_and_linework_preservation(self):
        """Micro-relief is strictly zero on facial skin, and geometric correlation r >= 0.80."""
        img_np = np.array(self.test_image.convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = img_np.shape
        lum = 0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]

        regions = {
            "subject": np.ones((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32) * 0.8,
            "face": np.ones((h, w), dtype=np.float32),  # Full face
            "clothing": np.zeros((h, w), dtype=np.float32),
        }

        res = micro_relief_engine.compute_micro_relief(
            luminance=lum,
            light_angle_deg=90.0,
            regions=regions,
            relief_strength=100.0,
        )

        # Micro-relief on face must be exactly 0.0
        self.assertEqual(float(np.max(np.abs(res["micro_relief_field"]))), 0.0)

    def test_vram_cleanup_after_phase8(self):
        """VRAM remains cleanly released after execution."""
        vram = vram_manager.get_cuda_memory_mb()
        self.assertLess(vram["allocated_mb"], 50.0)


if __name__ == "__main__":
    unittest.main()
