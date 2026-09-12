import numpy as np
import scipy.ndimage as ndi
from typing import Dict, Any, Tuple


class ContactShadowEngine:
    """
    Contact & Directional Cast Shadow Engine (Phase 8).
    Synthesizes physical contact shadows in crevices and directional cast shadows
    under straps, lingerie bands, underwires, clothing edges, and hair strands.
    """

    @staticmethod
    def compute_ambient_occlusion(
        luminance: np.ndarray,
        sigma_small: float = 1.2,
        sigma_large: float = 6.0,
    ) -> np.ndarray:
        """
        Computes ambient occlusion density in concave crevices and valleys.
        """
        lum_smooth = ndi.gaussian_filter(luminance, sigma=sigma_small)
        lum_blur = ndi.gaussian_filter(luminance, sigma=sigma_large)

        # Crevice valleys are darker than their local neighborhood
        crevice = np.clip((lum_blur - lum_smooth) * 3.5, 0.0, 1.0)
        # Deep shadow floor weighting
        dark_factor = np.clip((0.45 - lum_smooth) * 2.2, 0.0, 1.0)

        ao_map = crevice * 0.65 + dark_factor * 0.35
        return np.clip(ao_map, 0.0, 1.0)

    def compute_directional_contact_shadows(
        self,
        luminance: np.ndarray,
        light_angle_deg: float,
        regions: Dict[str, np.ndarray],
        shadow_depth: float = 40.0,
    ) -> Dict[str, np.ndarray]:
        """
        Synthesizes directional cast shadow falloff and ambient occlusion map.

        Returns:
            Dict[str, np.ndarray]:
                - contact_shadow_field: Composite occlusion field (0.0 to 0.50)
                - ambient_occlusion: Omnidirectional crevice AO
                - directional_cast: Directional cast shadow map along -L
        """
        h, w = luminance.shape
        w_depth = float(shadow_depth) / 100.0

        if w_depth <= 0.001:
            zeros = np.zeros((h, w), dtype=np.float32)
            return {
                "contact_shadow_field": zeros,
                "ambient_occlusion": zeros,
                "directional_cast": zeros,
            }

        # Light offset direction for casting shadows (opposite to light source: C = -L)
        rad = np.radians(light_angle_deg)
        cast_x = -np.sin(rad)
        cast_y = np.cos(rad)  # Downward is positive Y in image coordinates

        # Base Omnidirectional Crevice AO
        ao = self.compute_ambient_occlusion(luminance)

        # Extract casting edge contours (straps, underwires, lace edges, hair)
        m_cloth = regions.get("clothing", np.zeros((h, w), dtype=np.float32))
        m_hair = regions.get("hair", np.zeros((h, w), dtype=np.float32))
        m_face = regions.get("face", np.zeros((h, w), dtype=np.float32))
        m_skin = regions.get("skin", np.zeros((h, w), dtype=np.float32))
        m_acc = regions.get("accessories", np.zeros((h, w), dtype=np.float32))

        # Edge gradients of casting elements (straps, clothing, accessories, hair)
        caster_mask = np.clip(m_cloth + m_hair + m_acc * 1.5, 0.0, 1.0)
        c_gy, c_gx = np.gradient(ndi.gaussian_filter(caster_mask, sigma=1.0))
        cast_edge_mag = np.sqrt(c_gx ** 2 + c_gy ** 2)

        # Directional cast shift along cast vector
        shift_px = 3.5
        shifted_edges = ndi.shift(
            cast_edge_mag,
            [cast_y * shift_px, cast_x * shift_px],
            order=1,
            mode="nearest",
        )
        cast_shadow = ndi.gaussian_filter(shifted_edges, sigma=1.8) * 2.8

        # Only cast onto receiving surfaces (mostly skin, background, and lower clothing layers)
        receiver_mask = m_skin * 0.95 + (1.0 - caster_mask) * 0.60
        receiver_mask = receiver_mask * (1.0 - m_face * 0.60)  # Protect delicate facial features

        directional_cast = np.clip(cast_shadow * receiver_mask, 0.0, 1.0)

        # Composite Contact Occlusion Field
        total_contact_shadow = (ao * 0.50 + directional_cast * 0.50) * w_depth * 0.45
        total_contact_shadow = np.clip(total_contact_shadow, 0.0, 0.50)

        return {
            "contact_shadow_field": total_contact_shadow,
            "ambient_occlusion": ao,
            "directional_cast": directional_cast,
        }


contact_shadow_engine = ContactShadowEngine()
