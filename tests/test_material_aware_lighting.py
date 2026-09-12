import os
import unittest
import numpy as np
from PIL import Image

from enhancer.upscaling.engine import enhancement_engine
from enhancer.lighting.relighting.micro_relief import micro_relief_engine
from enhancer.lighting.relighting.contact_shadows import contact_shadow_engine
from enhancer.lighting.relighting.materials import material_shader_engine
from enhancer.lighting.relighting.validation import relighting_validator
from enhancer.perception.vram_manager import vram_manager


class TestMaterialAwareLighting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(
            os.path.dirname(__file__), "assets", "reference_test_image.jpg"
        )
        if os.path.exists(asset_path):
            cls.test_image = Image.open(asset_path)
        else:
            cls.test_image = Image.new("RGB", (256, 256), color=(220, 190, 170))

    def test_micro_relief_directional_modulation(self):
        """Micro-relief engine produces directional shading on clothing/lace while protecting face."""
        img_np = np.array(self.test_image.convert("L"), dtype=np.float32) / 255.0
        h, w = img_np.shape

        regions = {
            "clothing": np.ones((h, w), dtype=np.float32) * 0.8,
            "face": np.zeros((h, w), dtype=np.float32),
            "skin": np.zeros((h, w), dtype=np.float32),
        }
        # Mark center as face
        regions["face"][h // 4 : h // 2, w // 4 : w // 2] = 1.0

        res = micro_relief_engine.compute_micro_relief(
            luminance=img_np,
            light_angle_deg=315.0,
            regions=regions,
            relief_strength=75.0,
        )

        relief = res["micro_relief_field"]
        # Micro relief field has non-zero response on clothing
        self.assertGreater(np.max(np.abs(relief)), 0.01)
        # Face area is strictly zero
        face_relief = relief[h // 4 : h // 2, w // 4 : w // 2]
        self.assertAlmostEqual(float(np.mean(np.abs(face_relief))), 0.0, places=3)

    def test_contact_shadows_synthesis(self):
        """Contact shadow engine detects crevices and directional cast offsets."""
        img_np = np.array(self.test_image.convert("L"), dtype=np.float32) / 255.0
        h, w = img_np.shape

        regions = {
            "clothing": np.ones((h, w), dtype=np.float32) * 0.5,
            "skin": np.ones((h, w), dtype=np.float32) * 0.5,
            "face": np.zeros((h, w), dtype=np.float32),
        }

        res = contact_shadow_engine.compute_directional_contact_shadows(
            luminance=img_np,
            light_angle_deg=45.0,
            regions=regions,
            shadow_depth=60.0,
        )

        shadow_field = res["contact_shadow_field"]
        self.assertTrue(np.all(shadow_field >= 0.0))
        self.assertTrue(np.all(shadow_field <= 1.0))
        self.assertGreater(float(np.max(shadow_field)), 0.05)

    def test_phase5_enhancement_pipeline_execution(self):
        """Complete Phase 5 enhancement with micro-relief and physical lighting executes cleanly."""
        res = enhancement_engine.enhance(
            image=self.test_image,
            scale=1,
            model_id="4x-ultrasharp",
            relighting_enabled=True,
            light_direction_angle=315.0,
            light_intensity=120.0,
            micro_relief_strength=60.0,
            shadow_depth=45.0,
            specular_strength=55.0,
            material_response=110.0,
            output_dir="outputs/test_phase5",
        )

        self.assertTrue(res["success"])
        out_img = Image.open(res["output_path"])
        arr = np.array(out_img)

        # No NaNs or Infs
        self.assertFalse(np.any(np.isnan(arr)))
        self.assertFalse(np.any(np.isinf(arr)))

        # Structural edge correlation
        img_base = np.array(self.test_image.resize(out_img.size, Image.Resampling.BICUBIC).convert("RGB"), dtype=np.float32) / 255.0
        img_relit = np.array(out_img, dtype=np.float32) / 255.0
        val = relighting_validator.validate(img_base, img_relit, img_relit.shape)
        self.assertTrue(val["is_valid"])
        self.assertGreaterEqual(val["edge_correlation"], 0.60)

    def test_vram_cleanup_phase5(self):
        """VRAM is cleanly released after Phase 5 execution."""
        vram = vram_manager.get_cuda_memory_mb()
        self.assertLess(vram["allocated_mb"], 50.0)


if __name__ == "__main__":
    unittest.main()
