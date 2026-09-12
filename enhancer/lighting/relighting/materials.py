import numpy as np
import scipy.ndimage as ndi
from typing import Dict, Any, Tuple


class MaterialShaderEngine:
    """
    Material-Aware BRDF Lighting Shader Engine.
    Computes material-specific diffuse shading, anisotropic/specular highlights,
    optical wetting transmission, and surface response maps.
    """

    @staticmethod
    def compute_material_response_map(
        regions: Dict[str, np.ndarray],
        h: int,
        w: int,
    ) -> Dict[str, np.ndarray]:
        """
        Synthesizes 2D material response weighting maps for diffuse, specular, micro-relief, and transmission.
        """
        m_skin = ndi.gaussian_filter(regions.get("skin", np.zeros((h, w), dtype=np.float32)), sigma=1.5)
        m_face = ndi.gaussian_filter(regions.get("face", np.zeros((h, w), dtype=np.float32)), sigma=2.0)
        m_hair = ndi.gaussian_filter(regions.get("hair", np.zeros((h, w), dtype=np.float32)), sigma=1.5)
        m_cloth = ndi.gaussian_filter(regions.get("clothing", np.zeros((h, w), dtype=np.float32)), sigma=1.5)
        m_acc = ndi.gaussian_filter(regions.get("accessories", np.zeros((h, w), dtype=np.float32)), sigma=1.0)
        m_bg = ndi.gaussian_filter(regions.get("background", np.zeros((h, w), dtype=np.float32)), sigma=2.0)

        # Specular response map
        specular_response = (
            m_acc * 0.95
            + m_hair * 0.65
            + m_cloth * 0.35
            + (m_skin - m_face * 0.5) * 0.14
            + m_face * 0.08
            + m_bg * 0.05
        )

        # Diffuse response map
        diffuse_response = (
            m_skin * 1.05
            + m_cloth * 0.95
            + m_bg * 0.70
            + m_hair * 0.85
            + m_face * 0.90
            + m_acc * 0.30
        )

        return {
            "specular_response": np.clip(specular_response, 0.0, 1.0),
            "diffuse_response": np.clip(diffuse_response, 0.0, 1.2),
            "skin_mask": m_skin,
            "face_mask": m_face,
            "hair_mask": m_hair,
            "clothing_mask": m_cloth,
            "accessories_mask": m_acc,
            "background_mask": m_bg,
        }

    @staticmethod
    def shade_skin(
        normals: np.ndarray,
        light_vec: np.ndarray,
        view_vec: np.ndarray,
        subsurface_wrap: float = 0.35,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsurface-scattering diffuse + subtle sheen for skin."""
        ndotl = np.sum(normals * light_vec, axis=-1)
        diffuse = np.clip((ndotl + subsurface_wrap) / (1.0 + subsurface_wrap), 0.0, 1.0)

        half_vec = light_vec + view_vec
        half_vec = half_vec / (np.linalg.norm(half_vec, axis=-1, keepdims=True) + 1e-6)
        ndoth = np.clip(np.sum(normals * half_vec, axis=-1), 0.0, 1.0)
        specular = (ndoth ** 12.0) * 0.14

        return diffuse, specular

    @staticmethod
    def shade_hair(
        normals: np.ndarray,
        light_vec: np.ndarray,
        view_vec: np.ndarray,
        tangents: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Anisotropic Kajiya-Kay specular highlights along hair flow."""
        ndotl = np.clip(np.sum(normals * light_vec, axis=-1), 0.0, 1.0)
        diffuse = ndotl * 0.85 + 0.15

        tdotl = np.sum(tangents * light_vec, axis=-1)
        tdotv = np.sum(tangents * view_vec, axis=-1)
        sin_tl = np.sqrt(np.clip(1.0 - tdotl ** 2, 0.0, 1.0))
        sin_tv = np.sqrt(np.clip(1.0 - tdotv ** 2, 0.0, 1.0))
        specular = np.clip(sin_tl * sin_tv - tdotl * tdotv, 0.0, 1.0) ** 26.0 * 0.65

        return diffuse, specular

    @staticmethod
    def shade_metal(
        normals: np.ndarray,
        light_vec: np.ndarray,
        view_vec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """High-frequency Blinn-Phong specular for metal hardware."""
        ndotl = np.clip(np.sum(normals * light_vec, axis=-1), 0.0, 1.0)
        diffuse = ndotl * 0.35

        half_vec = light_vec + view_vec
        half_vec = half_vec / (np.linalg.norm(half_vec, axis=-1, keepdims=True) + 1e-6)
        ndoth = np.clip(np.sum(normals * half_vec, axis=-1), 0.0, 1.0)
        specular = (ndoth ** 64.0) * 0.95

        return diffuse, specular


material_shader_engine = MaterialShaderEngine()
