import unittest
import torch
from PIL import Image

from enhancer.upscaling.engine import enhancement_engine
from enhancer.perception.vram_manager import vram_manager


class TestUpscalingVRAM(unittest.TestCase):
    def test_vram_release_after_enhancement(self):
        """VRAM allocated is cleaned up after model execution."""
        vram_before = vram_manager.get_cuda_memory_mb()

        # Run 2x enhancement
        test_img = Image.new("RGB", (128, 128), color=(180, 150, 120))
        res = enhancement_engine.enhance(
            image=test_img,
            scale=2,
            model_id="4x-realcugan",
            output_dir="outputs/test_upscale",
        )

        self.assertTrue(res["success"])
        vram_after = vram_manager.get_cuda_memory_mb()

        # If on CUDA, verify memory is returned to base level (< 20 MB residual cache)
        if torch.cuda.is_available():
            self.assertLess(vram_after["allocated_mb"], 50.0)


if __name__ == "__main__":
    unittest.main()
