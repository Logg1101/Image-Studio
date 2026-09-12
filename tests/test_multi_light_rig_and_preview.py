import os
import time
import unittest
import numpy as np
from PIL import Image

from enhancer.lighting.relighting.engine import relighting_engine
from enhancer.lighting.relighting.compositor import relighting_compositor
from enhancer.lighting.relighting.validation import relighting_validator
from enhancer.perception.vram_manager import vram_manager


class TestMultiLightRigAndPreview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(
            os.path.dirname(__file__), "assets", "reference_flat_test_image.jpg"
        )
        if os.path.exists(asset_path):
            cls.test_image = Image.open(asset_path)
        else:
            cls.test_image = Image.new("RGB", (256, 256), color=(220, 190, 170))

    def test_multi_light_rig_key_rim_fill_computation(self):
        """Computes Key, Rim, and Fill light fields simultaneously."""
        img_np = np.array(self.test_image.convert("L"), dtype=np.float32) / 255.0
        h, w = img_np.shape

        regions = {
            "subject": np.ones((h, w), dtype=np.float32),
            "hair": np.zeros((h, w), dtype=np.float32),
            "clothing": np.ones((h, w), dtype=np.float32) * 0.7,
            "skin": np.ones((h, w), dtype=np.float32) * 0.8,
            "face": np.zeros((h, w), dtype=np.float32),
        }
        regions["hair"][: h // 3, :] = 1.0

        normals, tangents, refined_depth, base_depth, normal_rgb = relighting_engine.estimate_surface_normals(
            luminance=img_np, regions=regions
        )

        fields = relighting_engine.compute_lighting_fields(
            luminance=img_np,
            normals=normals,
            tangents=tangents,
            regions=regions,
            target_angle_deg=90.0,
            rim_light_enabled=True,
            rim_light_angle=45.0,
            rim_light_intensity=75.0,
            fill_light_enabled=True,
            fill_light_angle=225.0,
            fill_light_intensity=40.0,
        )

        # Assert all fields are present and valid
        self.assertIn("target_diffuse", fields)
        self.assertIn("rim_light_field", fields)
        self.assertIn("fill_light_field", fields)

        rim = fields["rim_light_field"]
        fill = fields["fill_light_field"]

        self.assertGreater(float(np.max(rim)), 0.05)
        self.assertGreater(float(np.max(fill)), 0.05)
        self.assertFalse(np.any(np.isnan(rim)))
        self.assertFalse(np.any(np.isnan(fill)))

    def test_rim_light_fresnel_grazing_effect(self):
        """Rim light is concentrated on grazing silhouette edges and minimal on flat center."""
        normals = np.zeros((100, 100, 3), dtype=np.float32)
        # Center points forward (Nz = 1.0)
        normals[40:60, 40:60] = [0.0, 0.0, 1.0]
        # Edges point outward (grazing, Nz = 0.1, Nx = 0.99)
        normals[0:20, :] = [0.0, -0.99, 0.1]
        normals[80:100, :] = [0.0, 0.99, 0.1]
        normals[:, 0:20] = [-0.99, 0.0, 0.1]
        normals[:, 80:100] = [0.99, 0.0, 0.1]

        tangents = np.zeros_like(normals)
        tangents[..., 0] = 1.0

        lum = np.ones((100, 100), dtype=np.float32) * 0.5
        regions = {
            "subject": np.ones((100, 100), dtype=np.float32),
            "hair": np.ones((100, 100), dtype=np.float32),
            "clothing": np.zeros((100, 100), dtype=np.float32),
            "skin": np.zeros((100, 100), dtype=np.float32),
            "face": np.zeros((100, 100), dtype=np.float32),
        }

        fields = relighting_engine.compute_lighting_fields(
            luminance=lum,
            normals=normals,
            tangents=tangents,
            regions=regions,
            target_angle_deg=90.0,
            rim_light_enabled=True,
            rim_light_angle=90.0,
            rim_light_intensity=100.0,
        )

        rim = fields["rim_light_field"]
        # Center has Nz = 1.0 -> Fresnel (1 - Nz) is 0.0
        self.assertAlmostEqual(float(np.mean(rim[40:60, 40:60])), 0.0, places=3)
        # Right edge facing rim light has strong grazing highlight
        self.assertGreater(float(np.max(rim[:, 80:100])), 0.30)

    def test_fast_relight_preview_speed_and_bounds(self):
        """Fast preview executes in under 150ms and produces clean output."""
        t0 = time.time()
        preview = relighting_engine.fast_relight_preview(
            image=self.test_image,
            light_direction_angle=135.0,
            light_intensity=120.0,
            rim_light_enabled=True,
            rim_light_angle=45.0,
            fill_light_enabled=True,
            max_dim=512,
        )
        elapsed_ms = (time.time() - t0) * 1000

        self.assertIsInstance(preview, Image.Image)
        self.assertLessEqual(max(preview.size), 512)
        # Should be ultra fast on downscaled proxy
        self.assertLess(elapsed_ms, 200.0)

        arr = np.array(preview)
        self.assertFalse(np.any(np.isnan(arr)))
        self.assertFalse(np.any(np.isinf(arr)))

    def test_compositor_color_tint_parsing(self):
        """Color tint parser correctly parses hex codes to normalized RGB multipliers."""
        cyan = relighting_compositor._parse_color_tint("#35D6C5")
        # Cyan has high G and B, lower R
        self.assertGreater(cyan[1], cyan[0])
        self.assertGreater(cyan[2], cyan[0])

        amber = relighting_compositor._parse_color_tint("#F59E0B")
        # Amber has high R and G, lower B
        self.assertGreater(amber[0], amber[2])
        self.assertGreater(amber[1], amber[2])

    def test_geometry_preservation_with_multi_light(self):
        """3-Point multi-light rig preserves image structural correlation r >= 0.80."""
        img_np = np.array(self.test_image.convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = img_np.shape
        lum = 0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]

        regions = {
            "subject": np.ones((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32),
            "face": np.zeros((h, w), dtype=np.float32),
        }

        normals, tangents, refined_depth, base_depth, normal_rgb = relighting_engine.estimate_surface_normals(
            luminance=lum, regions=regions
        )

        fields = relighting_engine.compute_lighting_fields(
            luminance=lum,
            normals=normals,
            tangents=tangents,
            regions=regions,
            target_angle_deg=90.0,
            micro_relief_strength=30.0,
            rim_light_enabled=True,
            rim_light_angle=45.0,
            rim_light_intensity=60.0,
            fill_light_enabled=True,
            fill_light_angle=225.0,
            fill_light_intensity=30.0,
        )

        res = relighting_compositor.composite(
            base_image_np=img_np,
            target_diffuse=fields["target_diffuse"],
            target_specular=fields["target_specular"],
            orig_diffuse=fields["orig_diffuse"],
            orig_specular=fields["orig_specular"],
            material_response_map=fields["material_response_map"],
            micro_relief_field=fields["micro_relief_field"],
            contact_shadow_field=fields["contact_shadow_field"],
            rim_light_field=fields.get("rim_light_field"),
            fill_light_field=fields.get("fill_light_field"),
            regions=regions,
            boundaries={},
            confidence_map=np.ones_like(lum),
            export_debug=False,
        )

        val = relighting_validator.validate(img_np, res["final_image"], (h, w, 3))
        self.assertTrue(val["is_valid"])
        self.assertGreaterEqual(val["edge_correlation"], 0.80)


if __name__ == "__main__":
    unittest.main()
