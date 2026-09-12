import unittest
import numpy as np
from PIL import Image
from enhancer.upscaling.material_renderer import MaterialRenderer, material_renderer


class TestMaterialRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = MaterialRenderer(
            crest_strength=0.35,
            trough_depth=0.40,
            sss_strength=0.30,
        )

    def test_dimensions_and_channels_preserved(self):
        # Create a synthetic image with fabric-like stripes and gradients
        w, h = 256, 256
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        # Synthetic weave pattern: high-frequency sin waves
        weave = np.sin(xx * 50) * np.cos(yy * 50) * 0.2 + 0.5
        rgb = np.stack([weave, weave * 0.9, weave * 0.8], axis=-1)
        pil_in = Image.fromarray((rgb * 255).astype(np.uint8))

        pil_out = self.renderer.process_pil(pil_in)

        self.assertEqual(pil_out.size, (w, h))
        self.assertEqual(pil_out.mode, "RGB")
        out_arr = np.array(pil_out)
        self.assertFalse(np.isnan(out_arr).any())
        self.assertFalse(np.isinf(out_arr).any())

    def test_crest_and_trough_contrast_expansion(self):
        # Base mid-gray background
        arr = np.full((100, 100, 3), 0.5, dtype=np.float32)
        # Ridge (crest) at x=50, valley (trough) at x=40
        arr[:, 50, :] = 0.65  # crest
        arr[:, 40, :] = 0.35  # trough

        enhanced = self.renderer.apply_crest_trough_depth(arr, crest_strength=0.5, trough_depth=0.5)

        # The crest should be brighter than original, trough should be darker
        self.assertGreater(enhanced[:, 50, :].mean(), arr[:, 50, :].mean())
        self.assertLess(enhanced[:, 40, :].mean(), arr[:, 40, :].mean())

    def test_subsurface_scattering_warmth_on_skin(self):
        # Create a 100x100 skin-toned patch with a gradient shadow
        skin_base = np.array([210, 160, 130], dtype=np.float32) / 255.0  # Warm skin tone
        grad = np.linspace(0.2, 0.9, 100)[:, None]  # 100x1
        grad_2d = np.repeat(grad, 100, axis=1)[:, :, None]  # 100x100x1
        skin_img = skin_base[None, None, :] * grad_2d  # 100x100x3

        sss_out = self.renderer.apply_subsurface_scattering(skin_img, sss_strength=0.5)

        # Terminator should exhibit red/amber channel elevation relative to blue
        r_diff = sss_out[:, :, 0] - skin_img[:, :, 0]
        b_diff = sss_out[:, :, 2] - skin_img[:, :, 2]
        self.assertGreater(r_diff.mean(), b_diff.mean())


if __name__ == "__main__":
    unittest.main()
