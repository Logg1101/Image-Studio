import os
import unittest
import numpy as np
from PIL import Image

from enhancer.upscaling.engine import enhancement_engine
from enhancer.lighting.relighting.depth_reconstruction import surface_depth_reconstructor
from enhancer.lighting.relighting.engine import relighting_engine
from enhancer.lighting.relighting.validation import relighting_validator


class TestSurfaceDepthAndNormals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(
            os.path.dirname(__file__), "assets", "reference_flat_test_image.jpg"
        )
        if os.path.exists(asset_path):
            cls.test_image = Image.open(asset_path)
        else:
            cls.test_image = Image.new("RGB", (256, 256), color=(220, 190, 170))

    def test_depth_has_meaningful_spatial_variation(self):
        """Depth map has non-zero spatial variance and continuous curvature across body."""
        img_np = np.array(self.test_image.convert("L"), dtype=np.float32) / 255.0
        h, w = img_np.shape

        regions = {
            "subject": np.ones((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32) * 0.8,
            "face": np.zeros((h, w), dtype=np.float32),
            "clothing": np.zeros((h, w), dtype=np.float32),
        }
        regions["face"][h // 6 : h // 3, w // 3 : 2 * w // 3] = 1.0

        base_d, refined_d = surface_depth_reconstructor.reconstruct_depth_map(
            luminance=img_np,
            regions=regions,
        )

        self.assertGreater(float(np.std(refined_d)), 0.04)
        self.assertTrue(np.all(refined_d >= 0.0))
        self.assertTrue(np.all(refined_d <= 1.0))

    def test_normals_are_unit_length_and_bounded(self):
        """Reconstructed surface normal vectors are unit length (||N|| = 1.0) and bounded [-1, 1]."""
        img_np = np.array(self.test_image.convert("L"), dtype=np.float32) / 255.0
        h, w = img_np.shape

        regions = {
            "subject": np.ones((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32) * 0.8,
        }

        base_d, refined_d = surface_depth_reconstructor.reconstruct_depth_map(
            luminance=img_np,
            regions=regions,
        )

        normals, tangents, normal_rgb = surface_depth_reconstructor.compute_surface_normals_from_depth(
            depth_map=refined_d,
            luminance=img_np,
        )

        # Unit length
        norm_mags = np.linalg.norm(normals, axis=-1)
        np.testing.assert_allclose(norm_mags, 1.0, atol=1e-3)

        # Bounded [-1, 1]
        self.assertTrue(np.all(normals >= -1.0 - 1e-4))
        self.assertTrue(np.all(normals <= 1.0 + 1e-4))

    def test_normals_vary_across_curved_body_regions(self):
        """Nx and Ny exhibit opposing signs on left vs right side of curved body volumes."""
        img_np = np.array(self.test_image.convert("L"), dtype=np.float32) / 255.0
        h, w = img_np.shape

        regions = {
            "subject": np.ones((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32),
        }

        base_d, refined_d = surface_depth_reconstructor.reconstruct_depth_map(
            luminance=img_np,
            regions=regions,
        )

        normals, _, _ = surface_depth_reconstructor.compute_surface_normals_from_depth(
            depth_map=refined_d,
            luminance=img_np,
        )

        # Left hemisphere of subject volume
        left_nx = normals[:, : w // 3, 0]
        # Right hemisphere of subject volume
        right_nx = normals[:, 2 * w // 3 :, 0]

        # Left side points left (Nx < 0) and right side points right (Nx > 0)
        self.assertLess(float(np.mean(left_nx)), 0.0)
        self.assertGreater(float(np.mean(right_nx)), 0.0)

    def test_directional_ndotl_changes_with_light_angle(self):
        """N·L illumination state shifts spatially and significantly between Right and Left lighting."""
        img_np = np.array(self.test_image.convert("L"), dtype=np.float32) / 255.0
        h, w = img_np.shape

        regions = {
            "subject": np.ones((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32),
        }

        normals, tangents, refined_depth, base_depth, normal_rgb = relighting_engine.estimate_surface_normals(
            luminance=img_np,
            regions=regions,
        )

        f_right = relighting_engine.compute_lighting_fields(
            luminance=img_np,
            normals=normals,
            tangents=tangents,
            regions=regions,
            target_angle_deg=90.0,
        )

        f_left = relighting_engine.compute_lighting_fields(
            luminance=img_np,
            normals=normals,
            tangents=tangents,
            regions=regions,
            target_angle_deg=270.0,
        )

        diff = float(np.mean(np.abs(f_right["target_ndotl_raw"] - f_left["target_ndotl_raw"])))
        # Substantial spatial variation of surface normal alignment (diff > 0.12)
        self.assertGreater(diff, 0.12)

    def test_full_phase7_compositor_and_geometry_preservation(self):
        """Complete Phase 7 compositor preserves structural geometry without geometric distortion."""
        img_np = np.array(self.test_image.convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = img_np.shape
        lum = 0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]

        regions = {
            "subject": np.ones((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32) * 0.8,
            "face": np.zeros((h, w), dtype=np.float32),
            "clothing": np.zeros((h, w), dtype=np.float32),
        }

        normals, tangents, refined_depth, base_depth, normal_rgb = relighting_engine.estimate_surface_normals(
            luminance=lum,
            regions=regions,
        )

        fields = relighting_engine.compute_lighting_fields(
            luminance=lum,
            normals=normals,
            tangents=tangents,
            regions=regions,
            target_angle_deg=90.0,
        )

        from enhancer.lighting.relighting.compositor import relighting_compositor

        comp_res = relighting_compositor.composite(
            base_image_np=img_np,
            target_diffuse=fields["target_diffuse"],
            target_specular=fields["target_specular"],
            orig_diffuse=fields["orig_diffuse"],
            orig_specular=fields["orig_specular"],
            material_response_map=fields["material_response_map"],
            micro_relief_field=fields["micro_relief_field"],
            contact_shadow_field=fields["contact_shadow_field"],
            base_depth=base_depth,
            refined_depth=refined_depth,
            normal_rgb=normal_rgb,
            target_ndotl_raw=fields["target_ndotl_raw"],
            regions=regions,
            boundaries={},
            confidence_map=np.ones((h, w), dtype=np.float32),
            light_intensity=100.0,
            export_debug=True,
            output_dir="outputs/test_phase7_comp",
        )

        val = relighting_validator.validate(img_np, comp_res["final_image"], img_np.shape)
        self.assertTrue(val["is_valid"])
        self.assertGreaterEqual(val["edge_correlation"], 0.80)

    def test_face_geometry_reconstruction(self):
        """Facial mask produces distinct convex volume compared to unmasked geometry."""
        img_np = np.zeros((128, 128), dtype=np.float32)
        h, w = img_np.shape

        regions_without_face = {
            "subject": np.ones((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32),
        }
        _, depth_no_face = surface_depth_reconstructor.reconstruct_depth_map(
            luminance=img_np, regions=regions_without_face
        )

        regions_with_face = {
            "subject": np.ones((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32),
            "face": np.zeros((h, w), dtype=np.float32),
        }
        # Place face in upper center
        regions_with_face["face"][20:60, 44:84] = 1.0
        _, depth_with_face = surface_depth_reconstructor.reconstruct_depth_map(
            luminance=img_np, regions=regions_with_face
        )

        # Center of the face should have higher depth due to ellipsoidal facial dome
        face_center_no = depth_no_face[40, 64]
        face_center_with = depth_with_face[40, 64]
        self.assertGreater(face_center_with, face_center_no)

    def test_background_mask_isolation(self):
        """Segmented background mask correctly isolates background to low depth values."""
        img_np = np.zeros((128, 128), dtype=np.float32)
        h, w = img_np.shape

        # Subject in left half, background in right half
        regions = {
            "subject": np.zeros((h, w), dtype=np.float32),
            "background": np.zeros((h, w), dtype=np.float32),
        }
        regions["subject"][:, :64] = 1.0
        regions["background"][:, 64:] = 1.0

        _, refined_depth = surface_depth_reconstructor.reconstruct_depth_map(
            luminance=img_np, regions=regions
        )

        # Background region should be suppressed near 0.05
        bg_mean_depth = float(np.mean(refined_depth[:, 70:]))
        self.assertLess(bg_mean_depth, 0.10)


if __name__ == "__main__":
    unittest.main()
