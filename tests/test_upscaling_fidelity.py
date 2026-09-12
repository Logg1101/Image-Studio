import os
import unittest
import numpy as np
from PIL import Image

from enhancer.upscaling.engine import enhancement_engine


class TestUpscalingFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "reference_test_image.jpg")
        if os.path.exists(asset_path):
            cls.test_image = Image.open(asset_path)
        else:
            cls.test_image = Image.new("RGB", (256, 256), color=(200, 180, 160))

    def test_original_base_anchoring(self):
        """Zero detail recovery produces base upscaled image."""
        res_zero_detail = enhancement_engine.enhance(
            image=self.test_image,
            scale=2,
            model_id="4x-realcugan",
            detail_recovery=0.0,
            texture_synthesis=0.0,
            sharpen=0.0,
            denoise=0.0,
            output_dir="outputs/test_upscale",
        )

        img_out = Image.open(res_zero_detail["output_path"])
        arr_out = np.array(img_out, dtype=np.float32) / 255.0

        # Base bicubic
        base_pil = self.test_image.resize((self.test_image.width * 2, self.test_image.height * 2), Image.Resampling.BICUBIC)
        arr_base = np.array(base_pil, dtype=np.float32) / 255.0

        # Mean absolute error should be virtually 0 (< 0.01)
        mae = np.mean(np.abs(arr_out - arr_base))
        self.assertLess(mae, 0.01)

    def test_no_nan_or_inf(self):
        """Output contains no NaNs or Infs."""
        res = enhancement_engine.enhance(
            image=self.test_image,
            scale=2,
            model_id="4x-realcugan",
            detail_recovery=75.0,
            texture_synthesis=50.0,
            output_dir="outputs/test_upscale",
        )
        img_out = Image.open(res["output_path"])
        arr = np.array(img_out)
        self.assertFalse(np.any(np.isnan(arr)))
        self.assertFalse(np.any(np.isinf(arr)))


if __name__ == "__main__":
    unittest.main()
