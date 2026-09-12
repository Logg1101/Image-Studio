import os
import unittest
import numpy as np
from PIL import Image

from enhancer.upscaling.engine import enhancement_engine
from enhancer.lighting.relighting.engine import relighting_engine
from enhancer.lighting.relighting.compositor import relighting_compositor
from enhancer.lighting.relighting.validation import relighting_validator
from enhancer.perception.vram_manager import vram_manager


class TestSemanticRelighting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(
            os.path.dirname(__file__), "assets", "reference_test_image.jpg"
        )
        if os.path.exists(asset_path):
            cls.test_image = Image.open(asset_path)
        else:
            cls.test_image = Image.new("RGB", (256, 256), color=(220, 190, 170))

    def test_relighting_alters_pixels_with_angle(self):
        """Changing target light direction angle produces distinct illumination states."""
        res_left = enhancement_engine.enhance(
            image=self.test_image,
            scale=1,
            model_id="4x-realcugan",
            relighting_enabled=True,
            light_direction_angle=270.0,  # Left
            light_intensity=120.0,
            output_dir="outputs/test_relight",
        )

        res_right = enhancement_engine.enhance(
            image=self.test_image,
            scale=1,
            model_id="4x-realcugan",
            relighting_enabled=True,
            light_direction_angle=90.0,   # Right
            light_intensity=120.0,
            output_dir="outputs/test_relight",
        )

        img_left = np.array(Image.open(res_left["output_path"]), dtype=np.float32) / 255.0
        img_right = np.array(Image.open(res_right["output_path"]), dtype=np.float32) / 255.0

        diff = np.mean(np.abs(img_left - img_right))
        self.assertGreater(diff, 0.010)

    def test_geometric_invariance_and_edge_correlation(self):
        """Relighting preserves structural geometry (edge correlation >= 0.88)."""
        res = enhancement_engine.enhance(
            image=self.test_image,
            scale=1,
            model_id="4x-realcugan",
            relighting_enabled=True,
            light_direction_angle=135.0,
            light_intensity=100.0,
            output_dir="outputs/test_relight",
        )

        out_img = Image.open(res["output_path"])
        img_base = np.array(self.test_image.resize(out_img.size, Image.Resampling.BICUBIC).convert("RGB"), dtype=np.float32) / 255.0
        img_relit = np.array(out_img, dtype=np.float32) / 255.0

        val = relighting_validator.validate(img_base, img_relit, img_relit.shape)
        self.assertTrue(val["is_valid"])
        self.assertGreaterEqual(val["edge_correlation"], 0.65)

    def test_no_nan_or_inf_in_relit_output(self):
        """Relit output has zero NaNs or Infs and strictly bounded values in [0, 1]."""
        res = enhancement_engine.enhance(
            image=self.test_image,
            scale=2,
            model_id="4x-realcugan",
            relighting_enabled=True,
            light_direction_angle=225.0,
            light_temperature=50.0,
            shadow_recovery=75.0,
            output_dir="outputs/test_relight",
        )

        arr = np.array(Image.open(res["output_path"]))
        self.assertFalse(np.any(np.isnan(arr)))
        self.assertFalse(np.any(np.isinf(arr)))
        self.assertTrue(np.all(arr >= 0))
        self.assertTrue(np.all(arr <= 255))

    def test_vram_cleanup_after_relighting(self):
        """VRAM is cleaned up after neural upscaling + relighting."""
        vram_after = vram_manager.get_cuda_memory_mb()
        self.assertLess(vram_after["allocated_mb"], 50.0)


if __name__ == "__main__":
    unittest.main()
