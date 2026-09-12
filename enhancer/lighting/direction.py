import numpy as np
import scipy.ndimage as ndi
from typing import Dict, Any, Tuple, Optional


class LightDirectionEstimator:
    """
    Multi-Cue Light Direction & Source Position Estimator.
    Combines 2D luminance gradient vector fields, highlight/shadow centroid disparities,
    and semantic mask weighting to resolve the 3D key-light vector and 2D canvas origin.
    """

    @staticmethod
    def estimate_direction(
        luminance: np.ndarray,
        subject_mask: Optional[np.ndarray] = None,
        face_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Estimates dominant light angle (degrees), 3D direction vector, and canvas origin.

        Args:
            luminance (np.ndarray): 2D float array [H, W] in [0.0, 1.0].
            subject_mask (Optional[np.ndarray]): Spatial mask for the subject [H, W].
            face_mask (Optional[np.ndarray]): Spatial mask for face region [H, W].

        Returns:
            Dict[str, Any]: Light angle, 3D unit vector, shadow angle, and source position.
        """
        h, w = luminance.shape

        # Smooth luminance to filter micro-texture noise
        smooth_lum = ndi.gaussian_filter(luminance, sigma=3.0)

        # 1. 2D Gradient Vector Field (gx = dI/dx, gy = dI/dy)
        gy, gx = np.gradient(smooth_lum)

        # Weight gradients by subject region if available
        if subject_mask is not None:
            weight_map = subject_mask.copy()
            if face_mask is not None:
                # Add higher weight to face highlights/shading
                weight_map += face_mask * 1.5
        else:
            weight_map = np.ones((h, w), dtype=np.float32)

        # Focus on prominent gradient magnitudes
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        thresh = np.percentile(grad_mag, 65)
        grad_mask = (grad_mag > thresh).astype(np.float32) * weight_map

        total_weight = np.sum(grad_mask) + 1e-6
        mean_gx = float(np.sum(gx * grad_mask) / total_weight)
        mean_gy = float(np.sum(gy * grad_mask) / total_weight)

        # 2. Highlight and Shadow Centroid Analysis
        high_thresh = np.percentile(luminance, 85)
        shadow_thresh = np.percentile(luminance, 25)

        high_mask = (luminance > high_thresh).astype(np.float32) * weight_map
        shadow_mask = (luminance < shadow_thresh).astype(np.float32) * weight_map

        y_coords, x_coords = np.mgrid[0:h, 0:w]

        sum_high = np.sum(high_mask) + 1e-6
        high_cy = float(np.sum(y_coords * high_mask) / sum_high)
        high_cx = float(np.sum(x_coords * high_mask) / sum_high)

        sum_shadow = np.sum(shadow_mask) + 1e-6
        shadow_cy = float(np.sum(y_coords * shadow_mask) / sum_shadow)
        shadow_cx = float(np.sum(x_coords * shadow_mask) / sum_shadow)

        # Centroid disparity vector (pointing from shadow to highlight)
        disp_x = (high_cx - shadow_cx) / max(w, 1)
        disp_y = (high_cy - shadow_cy) / max(h, 1)

        # Combine gradient cue with centroid cue
        combined_dx = mean_gx * 0.4 + disp_x * 0.6
        combined_dy = mean_gy * 0.4 + disp_y * 0.6

        # Calculate angle: 0 deg = North (Top), 90 deg = East (Right), 180 deg = South (Bottom), 270 deg = West (Left)
        # Note: In image coordinates, Y increases downward. Light from top means dy < 0.
        # Vector points toward light: light_vec = (-combined_dx, -combined_dy)
        light_dx = combined_dx
        light_dy = -combined_dy  # Invert for visual Cartesian system (up is positive)

        rad = np.arctan2(light_dx, light_dy)  # Angle from North
        deg = float((np.degrees(rad)) % 360.0)

        # Shadow direction is opposite to light direction
        shadow_deg = float((deg + 180.0) % 360.0)

        # 3. 3D Unit Vector Construction [vx, vy, vz] (vz > 0 front-facing)
        xy_mag = np.sqrt(light_dx ** 2 + light_dy ** 2) + 1e-6
        elevation_angle_rad = np.radians(35.0)  # Standard frontal key-light elevation
        vx = float((light_dx / xy_mag) * np.cos(elevation_angle_rad))
        vy = float((light_dy / xy_mag) * np.cos(elevation_angle_rad))
        vz = float(np.sin(elevation_angle_rad))
        
        # Normalize unit vector
        norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        direction_vector = [vx / norm, vy / norm, vz / norm]

        # 4. Light source canvas position [x, y] in [0.0, 1.0]
        # Position is projected along the angle from image center (0.5, 0.5)
        angle_rad = np.radians(deg)
        source_x = float(np.clip(0.5 + 0.45 * np.sin(angle_rad), 0.05, 0.95))
        source_y = float(np.clip(0.5 - 0.45 * np.cos(angle_rad), 0.05, 0.95))

        return {
            "direction_angle_deg": deg,
            "direction_vector": direction_vector,
            "shadow_direction_deg": shadow_deg,
            "light_source_position": [source_x, source_y],
        }


light_direction_estimator = LightDirectionEstimator()
