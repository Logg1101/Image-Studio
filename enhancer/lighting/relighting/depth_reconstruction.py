import numpy as np
import scipy.ndimage as ndi
from typing import Dict, Any, Tuple, Optional


class SurfaceDepthReconstructor:
    """
    True Hierarchical Surface Depth & Normal Reconstruction Engine (Phase 7).
    Constructs continuous, physical 3D depth volumes for anatomical and material parts,
    then derives physically accurate surface normals N(x, y) via image-space spatial gradients.
    """

    def reconstruct_depth_map(
        self,
        luminance: np.ndarray,
        regions: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstructs macro depth and refined multi-layer material depth.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (base_depth, refined_depth)
        """
        h, w = luminance.shape

        if "subject" in regions:
            m_subj = regions["subject"]
        elif "background" in regions and np.any(regions["background"] > 0.1):
            m_subj = np.clip(1.0 - regions["background"], 0.0, 1.0)
        else:
            m_subj = np.ones((h, w), dtype=np.float32)

        m_skin = regions.get("skin", np.zeros((h, w), dtype=np.float32))
        m_face = regions.get("face", np.zeros((h, w), dtype=np.float32))
        m_hair = regions.get("hair", np.zeros((h, w), dtype=np.float32))
        m_cloth = regions.get("clothing", np.zeros((h, w), dtype=np.float32))
        m_acc = regions.get("accessories", np.zeros((h, w), dtype=np.float32))
        m_bg = regions.get("background", np.zeros((h, w), dtype=np.float32))

        # 1. Continuous Cylindrical & Ellipsoidal Anatomical Volumetric Depth
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)
        x_center = float(w) / 2.0
        half_w = float(w) / 2.2

        # Normalized horizontal coordinate u in [-1.0, 1.0] across subject width
        u_body = np.clip((x_indices - x_center) / half_w, -1.0, 1.0)
        z_cylinder = np.sqrt(np.clip(1.0 - u_body ** 2, 0.0, 1.0))

        # 2. Dual Chest & Limb Sub-Volume Domes (Left and Right Convexities)
        x_left_chest = x_center - 0.22 * w
        x_right_chest = x_center + 0.22 * w
        r_chest = 0.24 * w

        u_left_chest = np.clip((x_indices - x_left_chest) / r_chest, -1.0, 1.0)
        u_right_chest = np.clip((x_indices - x_right_chest) / r_chest, -1.0, 1.0)

        # Chest vertical span
        y_chest = 0.40 * h
        v_chest = np.clip((y_indices - y_chest) / (0.18 * h), -1.0, 1.0)

        dome_left = np.sqrt(np.clip(1.0 - (u_left_chest ** 2 + v_chest ** 2), 0.0, 1.0))
        dome_right = np.sqrt(np.clip(1.0 - (u_right_chest ** 2 + v_chest ** 2), 0.0, 1.0))

        # 3. Facial Ellipsoidal Volumetric Dome
        if np.any(m_face > 0.2):
            y_face_idx, x_face_idx = np.where(m_face > 0.2)
            y_face_c = float(np.mean(y_face_idx))
            x_face_c = float(np.mean(x_face_idx))
            r_face_x = max(float(np.max(x_face_idx) - np.min(x_face_idx)) / 2.0, 0.05 * w)
            r_face_y = max(float(np.max(y_face_idx) - np.min(y_face_idx)) / 2.0, 0.05 * h)
        else:
            y_face_c = 0.25 * h
            x_face_c = x_center
            r_face_x = 0.18 * w
            r_face_y = 0.18 * h

        u_face = np.clip((x_indices - x_face_c) / (r_face_x + 1e-6), -1.0, 1.0)
        v_face = np.clip((y_indices - y_face_c) / (r_face_y + 1e-6), -1.0, 1.0)
        dome_face = np.sqrt(np.clip(1.0 - (u_face ** 2 + v_face ** 2), 0.0, 1.0))

        # 4. Base Silhouette Distance Depth
        if np.max(m_subj) > 0.1:
            bin_mask = m_subj > 0.20
            dist = ndi.distance_transform_edt(bin_mask).astype(np.float32)
            max_d = np.max(dist) + 1e-6
            z_dist = np.sqrt(dist / max_d)
        else:
            z_dist = z_cylinder

        # Base macro depth
        base_depth = z_cylinder * 0.50 + z_dist * 0.50

        # 5. Fabric Folds & Micro-Relief High-Pass Layer
        lum_smooth = ndi.gaussian_filter(luminance, sigma=5.0)
        lum_folds = (luminance - lum_smooth) * (
            m_cloth * 0.25 + m_hair * 0.20 + m_face * 0.15
        )

        fine_texture = luminance - ndi.gaussian_filter(luminance, sigma=1.0)
        lace_micro_relief = fine_texture * m_cloth * 0.30

        # 6. Composite Refined Multi-Layer Depth
        z_face = dome_face * 0.30 * m_face
        z_subject = (
            base_depth * 0.45
            + (dome_left * 0.25 + dome_right * 0.25) * np.clip(m_skin + m_cloth, 0.0, 1.0)
            + z_face
            + lum_folds
            + lace_micro_relief
            + m_acc * 0.05
        )

        bg_weight = np.clip(np.maximum(m_bg, 1.0 - m_subj), 0.0, 1.0)
        subj_weight = np.clip(m_subj * (1.0 - m_bg), 0.0, 1.0)
        z_bg = bg_weight * 0.05
        refined_depth = z_subject * subj_weight + z_bg
        refined_depth = ndi.gaussian_filter(refined_depth, sigma=1.0)
        refined_depth = np.clip(refined_depth, 0.0, 1.0)

        return base_depth, refined_depth

    def compute_surface_normals_from_depth(
        self,
        depth_map: np.ndarray,
        luminance: np.ndarray,
        gradient_scale: float = 4.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Derives continuous surface normal map N [H, W, 3] from reconstructed depth field.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (unit_normals, unit_tangents, normal_rgb)
        """
        h, w = depth_map.shape

        # Image-space spatial gradients of depth
        dz_dy, dz_dx = np.gradient(depth_map)

        # High-frequency luminance gradient perturbation for micro-folds and lace
        lum_fine = luminance - ndi.gaussian_filter(luminance, sigma=1.2)
        dl_dy, dl_dx = np.gradient(lum_fine)

        # Surface normal components
        # Negative sign ensures facing left -> Nx < 0, facing right -> Nx > 0
        nx = -(dz_dx * gradient_scale * 12.0 + dl_dx * 2.5)
        ny = -(dz_dy * gradient_scale * 12.0 + dl_dy * 2.5)

        xy_sq = np.clip(nx ** 2 + ny ** 2, 0.0, 0.88)
        nz = np.sqrt(1.0 - xy_sq)

        normals = np.stack([nx, ny, nz], axis=-1)
        norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
        unit_normals = normals / norm_mag  # Guaranteed unit length and bounded [-1, 1]

        # Compute In-Plane Tangents T
        tx = -unit_normals[..., 1]
        ty = unit_normals[..., 0]
        tz = np.zeros_like(tx)
        tangents = np.stack([tx, ty, tz], axis=-1)
        t_mag = np.linalg.norm(tangents, axis=-1, keepdims=True) + 1e-6
        unit_tangents = tangents / t_mag

        # Normal RGB visualization: R=(Nx+1)/2, G=(Ny+1)/2, B=(Nz+1)/2
        normal_rgb = np.clip((unit_normals + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)

        return unit_normals, unit_tangents, normal_rgb


surface_depth_reconstructor = SurfaceDepthReconstructor()
