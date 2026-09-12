import numpy as np
import scipy.ndimage as ndi
from typing import Dict, Any, Tuple, Optional
from PIL import Image

from enhancer.lighting.types import LightingProfile
from enhancer.lighting.relighting.materials import material_shader_engine
from enhancer.lighting.relighting.micro_relief import micro_relief_engine
from enhancer.lighting.relighting.contact_shadows import contact_shadow_engine
from enhancer.lighting.relighting.depth_reconstruction import surface_depth_reconstructor


class RelightingEngine:
    """
    Physical-Parametric Relighting Engine (Phase 8 Multi-Light & Fast Preview).
    Reconstructs continuous 3D surface depth volumes and spatial surface normal fields N(x, y),
    then computes physically accurate multi-light rig fields (Key + Rim + Fill) using N · L.
    """

    def __init__(self):
        self.shader = material_shader_engine
        self.relief = micro_relief_engine
        self.contact = contact_shadow_engine
        self.depth_engine = surface_depth_reconstructor

    def estimate_surface_normals(
        self,
        luminance: np.ndarray,
        regions: Optional[Dict[str, np.ndarray]] = None,
        subject_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstructs continuous hierarchical depth maps and spatial surface normals.

        Returns:
            Tuple[normals, tangents, refined_depth, base_depth, normal_rgb]
        """
        h, w = luminance.shape
        if regions is None:
            regions = {}
            if subject_mask is not None:
                regions["subject"] = subject_mask
            else:
                regions["subject"] = np.ones((h, w), dtype=np.float32)

        # 1. Reconstruct Continuous Multi-Material 3D Depth
        base_depth, refined_depth = self.depth_engine.reconstruct_depth_map(
            luminance=luminance,
            regions=regions,
        )

        # 2. Derive Continuous Surface Normal Map N from Depth Gradients
        normals, tangents, normal_rgb = self.depth_engine.compute_surface_normals_from_depth(
            depth_map=refined_depth,
            luminance=luminance,
            gradient_scale=4.8,
        )

        return normals, tangents, refined_depth, base_depth, normal_rgb

    @staticmethod
    def construct_light_vector(
        angle_deg: float, elevation_deg: float = 35.0
    ) -> np.ndarray:
        """Constructs 3D unit light vector pointing from surface to light source."""
        rad_angle = np.radians(angle_deg)
        rad_elev = np.radians(elevation_deg)

        lx = np.sin(rad_angle) * np.cos(rad_elev)
        ly = -np.cos(rad_angle) * np.cos(rad_elev)  # Up is negative Y in canvas
        lz = np.sin(rad_elev)

        vec = np.array([lx, ly, lz], dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-6)

    def compute_lighting_fields(
        self,
        luminance: np.ndarray,
        normals: np.ndarray,
        tangents: np.ndarray,
        regions: Dict[str, np.ndarray],
        target_angle_deg: float,
        orig_angle_deg: float = 287.5,
        elevation_deg: float = 35.0,
        micro_relief_strength: float = 50.0,
        shadow_depth: float = 40.0,
        rim_light_enabled: bool = False,
        rim_light_angle: float = 45.0,
        rim_light_intensity: float = 50.0,
        fill_light_enabled: bool = False,
        fill_light_angle: float = 225.0,
        fill_light_intensity: float = 35.0,
    ) -> Dict[str, Any]:
        """
        Synthesizes complete physical lighting fields including 3-point studio rig
        (Key + Rim + Fill), material responses, micro-relief, and contact shadows.
        """
        h, w, _ = normals.shape
        view_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        target_light_vec = self.construct_light_vector(target_angle_deg, elevation_deg)
        orig_light_vec = self.construct_light_vector(orig_angle_deg, elevation_deg)

        # 1. True Physically Bounded Diffuse Illumination N · L with smooth subsurface wrap
        t_ndotl_raw = np.sum(normals * target_light_vec, axis=-1)
        o_ndotl_raw = np.sum(normals * orig_light_vec, axis=-1)

        # Diffuse with subtle subsurface wrap (w=0.25)
        target_diffuse = np.clip((t_ndotl_raw + 0.25) / 1.25, 0.0, 1.0)
        orig_diffuse = np.clip((o_ndotl_raw + 0.25) / 1.25, 0.0, 1.0)

        # 2. Material Response Maps
        mat_maps = self.shader.compute_material_response_map(regions, h, w)

        # 3. Specular Highlights
        m_hair = mat_maps["hair_mask"]
        m_face = mat_maps["face_mask"]
        m_cloth = mat_maps["clothing_mask"]
        m_acc = mat_maps["accessories_mask"]
        m_skin = mat_maps["skin_mask"]

        # Anisotropic Hair Specular
        tdotl_t = np.sum(tangents * target_light_vec, axis=-1)
        tdotv = np.sum(tangents * view_vec, axis=-1)
        sin_tl_t = np.sqrt(np.clip(1.0 - tdotl_t ** 2, 0.0, 1.0))
        sin_tv = np.sqrt(np.clip(1.0 - tdotv ** 2, 0.0, 1.0))
        hair_spec_t = np.clip(sin_tl_t * sin_tv - tdotl_t * tdotv, 0.0, 1.0) ** 26.0

        tdotl_o = np.sum(tangents * orig_light_vec, axis=-1)
        sin_tl_o = np.sqrt(np.clip(1.0 - tdotl_o ** 2, 0.0, 1.0))
        hair_spec_o = np.clip(sin_tl_o * sin_tv - tdotl_o * tdotv, 0.0, 1.0) ** 26.0

        # Glossy Blinn-Phong Specular
        def blinn_spec(l_vec: np.ndarray, power: float) -> np.ndarray:
            h_vec = l_vec + view_vec
            h_vec = h_vec / (np.linalg.norm(h_vec, axis=-1, keepdims=True) + 1e-6)
            ndoth = np.clip(np.sum(normals * h_vec, axis=-1), 0.0, 1.0)
            return ndoth ** power

        cloth_spec_t = blinn_spec(target_light_vec, 16.0)
        skin_spec_t = blinn_spec(target_light_vec, 12.0)
        metal_spec_t = blinn_spec(target_light_vec, 48.0)

        cloth_spec_o = blinn_spec(orig_light_vec, 16.0)
        skin_spec_o = blinn_spec(orig_light_vec, 12.0)
        metal_spec_o = blinn_spec(orig_light_vec, 48.0)

        hair_surface_weight = m_hair * (1.0 - m_face * 0.7)

        target_specular = (
            hair_spec_t * hair_surface_weight * 0.50
            + cloth_spec_t * m_cloth * 0.25
            + skin_spec_t * np.clip(m_skin - m_face * 0.5, 0.0, 1.0) * 0.16
            + metal_spec_t * m_acc * 0.85
        )

        orig_specular = (
            hair_spec_o * hair_surface_weight * 0.50
            + cloth_spec_o * m_cloth * 0.25
            + skin_spec_o * np.clip(m_skin - m_face * 0.5, 0.0, 1.0) * 0.16
            + metal_spec_o * m_acc * 0.85
        )

        # 4. Multi-Light Rig: Rim / Backlight
        rim_field = np.zeros((h, w), dtype=np.float32)
        if rim_light_enabled and rim_light_intensity > 0.001:
            rim_vec = self.construct_light_vector(rim_light_angle, elevation_deg=15.0)
            nz = np.clip(normals[..., 2], 0.0, 1.0)
            # Fresnel grazing edge falloff
            fresnel = np.clip(1.0 - nz, 0.0, 1.0) ** 2.8
            # Directional rim alignment
            rim_dir_align = np.clip(np.sum(normals * rim_vec, axis=-1), 0.0, 1.0)
            rim_gain = (float(rim_light_intensity) / 100.0)
            rim_mat_weight = m_hair * 1.6 + m_cloth * 0.9 + m_skin * 0.6
            rim_mat_weight = rim_mat_weight * (1.0 - m_face * 0.75)
            rim_field = np.clip(fresnel * rim_dir_align * rim_mat_weight * rim_gain, 0.0, 1.0)

        # 5. Multi-Light Rig: Fill / Bounce Light
        fill_field = np.zeros((h, w), dtype=np.float32)
        if fill_light_enabled and fill_light_intensity > 0.001:
            fill_vec = self.construct_light_vector(fill_light_angle, elevation_deg=20.0)
            fill_ndotl = np.clip(np.sum(normals * fill_vec, axis=-1), 0.0, 1.0)
            fill_gain = (float(fill_light_intensity) / 100.0)
            fill_field = np.clip(fill_ndotl * (m_skin + m_cloth + m_hair) * fill_gain * 0.60, 0.0, 1.0)

        # 6. Micro-Relief Shading (Phase 8)
        relief_dict = self.relief.compute_micro_relief(
            luminance=luminance,
            light_angle_deg=target_angle_deg,
            regions=regions,
            relief_strength=micro_relief_strength,
        )

        # 7. Contact Shadows & AO
        contact_dict = self.contact.compute_directional_contact_shadows(
            luminance=luminance,
            light_angle_deg=target_angle_deg,
            regions=regions,
            shadow_depth=shadow_depth,
        )

        return {
            "target_diffuse": np.clip(target_diffuse, 0.0, 1.0),
            "target_ndotl_raw": t_ndotl_raw,
            "target_specular": np.clip(target_specular, 0.0, 1.0),
            "orig_diffuse": np.clip(orig_diffuse, 0.0, 1.0),
            "orig_specular": np.clip(orig_specular, 0.0, 1.0),
            "rim_light_field": rim_field,
            "fill_light_field": fill_field,
            "material_response_map": mat_maps["specular_response"],
            "micro_relief_field": relief_dict["micro_relief_field"],
            "micro_relief_highlight_map": relief_dict["highlight_map"],
            "micro_relief_shadow_map": relief_dict["shadow_map"],
            "contact_shadow_field": contact_dict["contact_shadow_field"],
            "directional_cast_shadow_map": contact_dict["directional_cast"],
            "ambient_occlusion_map": contact_dict["ambient_occlusion"],
            "target_light_vector": target_light_vec,
            "orig_light_vector": orig_light_vec,
        }

    def fast_relight_preview(
        self,
        image: Image.Image,
        light_direction_angle: float = 287.5,
        light_intensity: float = 100.0,
        light_temperature: float = 0.0,
        shadow_depth: float = 40.0,
        specular_strength: float = 50.0,
        ambient_light: float = 30.0,
        contrast: float = 0.0,
        micro_relief_strength: float = 50.0,
        rim_light_enabled: bool = False,
        rim_light_angle: float = 45.0,
        rim_light_intensity: float = 50.0,
        rim_light_color: str = "#35D6C5",
        fill_light_enabled: bool = False,
        fill_light_angle: float = 225.0,
        fill_light_intensity: float = 35.0,
        fill_light_color: str = "#4F9CFF",
        max_dim: int = 512,
    ) -> Image.Image:
        """
        Ultra-fast (<30ms) interactive proxy relighting preview.
        """
        # Downscale for real-time interactivity
        w, h = image.size
        scale = min(1.0, float(max_dim) / max(w, h))
        if scale < 1.0:
            pw, ph = max(64, int(w * scale)), max(64, int(h * scale))
            img_proxy = image.resize((pw, ph), Image.Resampling.BILINEAR)
        else:
            img_proxy = image

        arr = np.array(img_proxy.convert("RGB"), dtype=np.float32) / 255.0
        lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

        # Fast approximate region estimation from luminance and chroma
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        skin_heur = np.clip((r - b) * 2.5 * (r > g) * (g > b), 0.0, 1.0)
        hair_heur = np.clip((0.35 - lum) * 2.8, 0.0, 1.0)
        cloth_heur = np.clip(1.0 - skin_heur - hair_heur, 0.0, 1.0)

        regions = {
            "subject": np.ones_like(lum),
            "skin": skin_heur,
            "clothing": cloth_heur,
            "hair": hair_heur,
            "face": np.zeros_like(lum),
        }

        # Estimate normals
        normals, tangents, refined_depth, base_depth, normal_rgb = self.estimate_surface_normals(
            luminance=lum, regions=regions
        )

        # Compute multi-light fields
        fields = self.compute_lighting_fields(
            luminance=lum,
            normals=normals,
            tangents=tangents,
            regions=regions,
            target_angle_deg=light_direction_angle,
            micro_relief_strength=micro_relief_strength * 0.8,
            shadow_depth=shadow_depth,
            rim_light_enabled=rim_light_enabled,
            rim_light_angle=rim_light_angle,
            rim_light_intensity=rim_light_intensity,
            fill_light_enabled=fill_light_enabled,
            fill_light_angle=fill_light_angle,
            fill_light_intensity=fill_light_intensity,
        )

        from enhancer.lighting.relighting.compositor import relighting_compositor
        res = relighting_compositor.composite(
            base_image_np=arr,
            target_diffuse=fields["target_diffuse"],
            target_specular=fields["target_specular"],
            orig_diffuse=fields["orig_diffuse"],
            orig_specular=fields["orig_specular"],
            material_response_map=fields["material_response_map"],
            micro_relief_field=fields["micro_relief_field"],
            contact_shadow_field=fields["contact_shadow_field"],
            regions=regions,
            boundaries={},
            confidence_map=np.ones_like(lum),
            rim_light_field=fields.get("rim_light_field"),
            rim_light_color=rim_light_color,
            fill_light_field=fields.get("fill_light_field"),
            fill_light_color=fill_light_color,
            light_intensity=light_intensity,
            light_temperature=light_temperature,
            shadow_depth=shadow_depth,
            specular_strength=specular_strength,
            ambient_fill=ambient_light,
            contrast=contrast,
            export_debug=False,
        )

        out_arr = (np.clip(res["final_image"] * 255.0, 0, 255)).astype(np.uint8)
        return Image.fromarray(out_arr)


relighting_engine = RelightingEngine()
