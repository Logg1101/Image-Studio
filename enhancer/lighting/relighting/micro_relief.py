import numpy as np
import scipy.ndimage as ndi
from typing import Dict, Any, Tuple


class MicroReliefEngine:
    """
    Physical Micro-Relief & Raised-Detail Shading Engine (Phase 8).
    Synthesizes localized directional highlight crests on illuminated sides of
    embroidery, lace threads, seams, buttons, and hair strands, while casting
    soft self-shadows and localized under-cast shadows on the trailing opposite sides.
    """

    @staticmethod
    def extract_high_frequency_bands(
        luminance: np.ndarray,
        sigma_fine: float = 0.8,
        sigma_medium: float = 2.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts multi-scale bandpass frequency layers isolating fine threads and medium folds.
        """
        fine_low = ndi.gaussian_filter(luminance, sigma=sigma_fine)
        fine_band = luminance - fine_low

        med_low = ndi.gaussian_filter(luminance, sigma=sigma_medium)
        med_band = fine_low - med_low

        return fine_band, med_band

    def compute_micro_relief(
        self,
        luminance: np.ndarray,
        light_angle_deg: float,
        regions: Dict[str, np.ndarray],
        relief_strength: float = 50.0,
    ) -> Dict[str, np.ndarray]:
        """
        Computes directional micro-relief highlight crests and trailing self-shadows.

        Returns:
            Dict[str, np.ndarray]:
                - micro_relief_field: Net modulation (-0.50 to +0.50)
                - highlight_map: Facing highlight crests (0.0 to 1.0)
                - shadow_map: Trailing self-shadows and fabric under-casts (0.0 to 1.0)
                - relief_mask: Semantic material weighting mask
                - lace_mask: Detected lace/embroidery energy mask
        """
        h, w = luminance.shape
        w_strength = float(relief_strength) / 100.0

        if w_strength <= 0.001:
            zeros = np.zeros((h, w), dtype=np.float32)
            return {
                "micro_relief_field": zeros,
                "highlight_map": zeros,
                "shadow_map": zeros,
                "relief_mask": zeros,
                "lace_mask": zeros,
            }

        # 2D Light Direction Vector in Image Coordinates (pointing toward light source)
        rad = np.radians(light_angle_deg)
        lx = np.sin(rad)
        ly = -np.cos(rad)  # Up is negative Y in canvas coordinates

        # Extract fine thread and fold bands
        fine_band, med_band = self.extract_high_frequency_bands(luminance)

        # Compute Directional Image Gradients
        f_gy, f_gx = np.gradient(fine_band)
        m_gy, m_gx = np.gradient(med_band)

        # Directional Slope Alignment:
        # Facing slopes have -(\nabla I · L) > 0; Trailing slopes have -(\nabla I · L) < 0
        f_slope = -(f_gx * lx + f_gy * ly) * 6.5
        m_slope = -(m_gx * lx + m_gy * ly) * 4.2
        raw_slope = f_slope * 0.70 + m_slope * 0.30

        # 1. Facing Highlight Crests (illuminated side of threads/embroidery)
        raw_highlights = np.maximum(raw_slope, 0.0)

        # 2. Trailing Self-Shadows (shadow on opposite side of thread)
        raw_self_shadows = np.maximum(-raw_slope, 0.0)

        # 3. Localized Fabric Under-Cast Shadows
        # Raised details cast a tiny soft shadow along cast vector C = -L onto underlying receiving fabric
        raised_energy = np.clip(np.abs(fine_band) * 6.0, 0.0, 1.0)
        cast_shift_px = 2.5
        shifted_cast = ndi.shift(
            raised_energy,
            [ly * cast_shift_px, -lx * cast_shift_px],
            order=1,
            mode="nearest",
        )
        # Blur the cast shadow slightly and mask out the thread itself so the shadow falls on the fabric base
        fabric_cast_shadow = ndi.gaussian_filter(shifted_cast, sigma=1.2) * (1.0 - raised_energy * 0.7)
        fabric_cast_shadow = np.clip(fabric_cast_shadow * 1.5, 0.0, 1.0)

        # Semantic Material Weighting (Strict Material Policies)
        m_cloth = regions.get("clothing", np.zeros((h, w), dtype=np.float32))
        m_hair = regions.get("hair", np.zeros((h, w), dtype=np.float32))
        m_face = regions.get("face", np.zeros((h, w), dtype=np.float32))
        m_skin = regions.get("skin", np.zeros((h, w), dtype=np.float32))
        m_acc = regions.get("accessories", np.zeros((h, w), dtype=np.float32))

        # Detect lace / high-contrast embroidery within clothing & stockings
        lace_energy = np.clip(np.abs(fine_band) * 8.0, 0.0, 1.0)
        lace_mask = lace_energy * m_cloth

        # Material-specific relief gains:
        # - Lace / Embroidery: 2.8x gain
        # - Accessories / Hardware: 1.8x gain
        # - General Clothing: 1.3x gain
        # - Hair strands: 1.0x gain
        # - Skin: 0.12x (strictly restrained to preserve smooth anime skin)
        # - Face: 0.0x (strictly 0 to prevent roughness or pore artifacts on face)
        relief_weight = (
            lace_mask * 2.8
            + m_cloth * 1.3
            + m_hair * 1.0
            + m_acc * 1.8
            + (m_skin - m_face) * 0.12
        )
        relief_weight = relief_weight * (1.0 - m_face)
        relief_weight = np.clip(relief_weight, 0.0, 3.0)

        # Modulate highlights and shadows by semantic material weights and slider strength
        modulated_highlights = raw_highlights * relief_weight * 0.50 * w_strength
        total_shadows = (raw_self_shadows * 0.70 + fabric_cast_shadow * 0.30) * relief_weight * 0.45 * w_strength

        # Combined Net Micro-Relief Delta
        micro_relief_field = np.clip(modulated_highlights - total_shadows, -0.50, 0.50)

        return {
            "micro_relief_field": micro_relief_field,
            "highlight_map": np.clip(modulated_highlights, 0.0, 1.0),
            "shadow_map": np.clip(total_shadows, 0.0, 1.0),
            "relief_mask": relief_weight,
            "lace_mask": lace_mask,
        }


micro_relief_engine = MicroReliefEngine()
