import os
import unittest
import base64
import io
import numpy as np
from PIL import Image

from core.preprocessors import preprocessor_engine
from pipelines.composition_pipeline import composition_coordinator


class TestCompositionStudioPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create synthetic test reference images
        cls.face_img = Image.new("RGB", (256, 256), color=(230, 190, 170))
        cls.pose_img = Image.new("RGB", (256, 256), color=(100, 120, 180))

    def test_preprocessors_canny(self):
        """Canny preprocessor extracts valid 3-channel binary edge map."""
        canny = preprocessor_engine.extract_canny(self.pose_img, 100, 200)
        self.assertIsInstance(canny, Image.Image)
        self.assertEqual(canny.size, (256, 256))
        arr = np.array(canny)
        self.assertEqual(arr.ndim, 3)
        self.assertEqual(arr.shape[2], 3)

    def test_preprocessors_depth(self):
        """Depth preprocessor produces continuous normalized depth representation in grayscale RGB."""
        depth = preprocessor_engine.extract_depth(self.pose_img)
        self.assertIsInstance(depth, Image.Image)
        self.assertEqual(depth.size, (256, 256))
        arr = np.array(depth)
        self.assertEqual(arr.shape, (256, 256, 3))
        self.assertFalse(np.any(np.isnan(arr)))
        # Grayscale depth map must have identical R, G, B values across all pixels
        np.testing.assert_array_equal(arr[..., 0], arr[..., 1])
        np.testing.assert_array_equal(arr[..., 1], arr[..., 2])

    def test_preprocessors_openpose(self):
        """OpenPose preprocessor produces 18-keypoint skeleton map."""
        pose = preprocessor_engine.extract_openpose(self.pose_img)
        self.assertIsInstance(pose, Image.Image)
        self.assertEqual(pose.size, (256, 256))
        arr = np.array(pose)
        self.assertEqual(arr.shape, (256, 256, 3))

    def test_preprocessors_lineart(self):
        """LineArt preprocessor produces high-contrast clean line map."""
        lineart = preprocessor_engine.extract_lineart(self.pose_img)
        self.assertIsInstance(lineart, Image.Image)
        self.assertEqual(lineart.size, (256, 256))

    def test_adapter_inventory_listing(self):
        """Scanner reports installed ControlNet, IP-Adapter, and PuLID models."""
        adapters = composition_coordinator.list_available_adapters()
        self.assertIn("controlnet", adapters)
        self.assertIn("ip_adapter", adapters)
        self.assertIn("pulid", adapters)
        self.assertIn("supported_architectures", adapters)

        cnet_ids = [c["id"] for c in adapters["controlnet"]]
        self.assertIn("openpose", cnet_ids)
        self.assertIn("depth", cnet_ids)
        self.assertIn("canny", cnet_ids)

    def test_composition_generation_sdxl_and_flux(self):
        """Multi-adapter composition generation pipeline handles SDXL generation and missing weights correctly."""
        res_sdxl = composition_coordinator.generate(
            prompt="1girl, solo, wearing kimono in bamboo forest",
            architecture="sdxl",
            identity_image=self.face_img,
            identity_strength=0.8,
            composition_image=self.pose_img,
            controlnet_type="openpose",
            controlnet_strength=0.75,
            width=256,
            height=256,
            steps=1,
            seed=42,
        )
        self.assertTrue(res_sdxl["success"])
        self.assertIn("data:image/png;base64,", res_sdxl["image_url"])
        self.assertEqual(res_sdxl["architecture"], "sdxl")

        # Flux checkpoint files are not present in this test environment;
        # it should report failure cleanly rather than silently producing a black frame.
        res_flux = composition_coordinator.generate(
            prompt="1girl, cyber armor in neon city",
            architecture="flux",
            identity_image=self.face_img,
            identity_strength=0.8,
            composition_image=self.pose_img,
            controlnet_type="depth",
            width=256,
            height=256,
            steps=1,
            seed=42,
        )
        self.assertFalse(res_flux["success"])
        self.assertIn("error", res_flux)
        self.assertNotIn("image_url", res_flux)
        self.assertEqual(res_flux["architecture"], "flux")

    def test_composition_generation_failure_does_not_save_corrupted_output(self):
        """Failure during diffusion run does not save dark frame to disk or HistoryDB."""
        from unittest.mock import patch
        import config.paths as paths

        images_before = set(paths.IMAGES_DIR.glob("*.png")) if paths.IMAGES_DIR.exists() else set()

        with patch.object(composition_coordinator, "_generate_sdxl_composition", side_effect=RuntimeError("Simulated CUDA OOM")):
            result = composition_coordinator.generate(
                prompt="test failure handling",
                architecture="sdxl",
                width=256,
                height=256,
                steps=1,
            )

        self.assertFalse(result["success"])
        self.assertIn("Simulated CUDA OOM", result["error"])
        self.assertNotIn("file_path", result)

        images_after = set(paths.IMAGES_DIR.glob("*.png")) if paths.IMAGES_DIR.exists() else set()
        self.assertEqual(images_before, images_after, "Failed generation should not write image to disk")


if __name__ == "__main__":
    unittest.main()
