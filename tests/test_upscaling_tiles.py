import unittest
import torch
import numpy as np

from enhancer.upscaling.tiled import TiledUpscaler


class TestUpscalingTiles(unittest.TestCase):
    def setUp(self):
        self.tiler = TiledUpscaler(tile_size=128, tile_overlap=32)

    def test_tiled_dimension_accuracy(self):
        """Tiled upscaling produces exact scaled dimensions."""
        h, w = 300, 450
        scale = 2
        input_tensor = torch.rand((1, 3, h, w), dtype=torch.float32)

        # Mock upscale callback (bilinear resize)
        def _mock_upscale(tile: torch.Tensor, sc: int) -> torch.Tensor:
            _, _, th, tw = tile.shape
            return torch.nn.functional.interpolate(
                tile, size=(th * sc, tw * sc), mode="bilinear", align_corners=False
            )

        output_tensor = self.tiler.upscale_tiled(
            image_tensor=input_tensor, upscale_fn=_mock_upscale, scale=scale
        )

        self.assertEqual(output_tensor.shape, (1, 3, h * scale, w * scale))
        self.assertTrue(torch.all(output_tensor >= 0.0))
        self.assertTrue(torch.all(output_tensor <= 1.0))
        self.assertFalse(torch.any(torch.isnan(output_tensor)))

    def test_tile_seamless_continuity(self):
        """Uniform input tensor remains uniform after tiled overlap blending."""
        h, w = 256, 256
        scale = 2
        uniform_val = 0.65
        input_tensor = torch.full((1, 3, h, w), uniform_val, dtype=torch.float32)

        def _identity_upscale(tile: torch.Tensor, sc: int) -> torch.Tensor:
            _, _, th, tw = tile.shape
            return torch.full((1, 3, th * sc, tw * sc), uniform_val, dtype=torch.float32)

        output_tensor = self.tiler.upscale_tiled(
            image_tensor=input_tensor, upscale_fn=_identity_upscale, scale=scale
        )

        # Check that no edge seams exist (max difference should be < 1e-4)
        diff = torch.abs(output_tensor - uniform_val)
        self.assertLess(torch.max(diff).item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
