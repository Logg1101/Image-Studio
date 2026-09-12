import os
import unittest
import numpy as np
from PIL import Image

from enhancer.upscaling.engine import enhancement_engine
from enhancer.upscaling.registry import upscaler_registry


class TestUpscalingEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "reference_test_image.jpg")
        if os.path.exists(asset_path):
            cls.test_image = Image.open(asset_path)
        else:
            cls.test_image = Image.new("RGB", (256, 256), color=(200, 180, 160))

    def test_1x_restoration_mode(self):
        """1x mode preserves original dimensions while performing neural restoration."""
        res = enhancement_engine.enhance(
            image=self.test_image,
            scale=1,
            model_id="4x-realcugan",
            detail_recovery=50.0,
            texture_synthesis=30.0,
            sharpen=20.0,
            denoise=10.0,
            output_dir="outputs/test_upscale",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["output_resolution"], [self.test_image.width, self.test_image.height])
        self.assertTrue(os.path.exists(res["output_path"]))

    def test_2x_upscaling_mode(self):
        """2x mode produces exact 2x dimensions."""
        res = enhancement_engine.enhance(
            image=self.test_image,
            scale=2,
            model_id="4x-realcugan",
            output_dir="outputs/test_upscale",
        )
        self.assertTrue(res["success"])
        self.assertEqual(
            res["output_resolution"],
            [self.test_image.width * 2, self.test_image.height * 2],
        )

    def test_4x_upscaling_mode(self):
        """4x mode produces exact 4x dimensions."""
        res = enhancement_engine.enhance(
            image=self.test_image,
            scale=4,
            model_id="4x-realcugan",
            output_dir="outputs/test_upscale",
        )
        self.assertTrue(res["success"])
        self.assertEqual(
            res["output_resolution"],
            [self.test_image.width * 4, self.test_image.height * 4],
        )

    def test_missing_model_error(self):
        """Requesting an invalid model produces a controlled error."""
        with self.assertRaises(FileNotFoundError):
            enhancement_engine.enhance(
                image=self.test_image,
                scale=2,
                model_id="nonexistent-model-xyz",
            )


if __name__ == "__main__":
    unittest.main()
