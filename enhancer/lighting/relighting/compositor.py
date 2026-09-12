import os
import io
import base64
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from typing import Dict, Any, Optional

from enhancer.lighting.types import LightingProfile


class RelightingCompositor:
    """
    Seamless Semantic Relighting Compositor with Multi-Light Rigging (Key + Rim + Fill),
    Material-Aware Micro-Relief, Raised-Detail Highlight/Shadow Separation & Contact Shadows.
    """

    @staticmethod
    def _compute_color_tint(temperature: float) -> np.ndarray:
        """Computes RGB chromaticity multiplier from color temperature slider (-100 to +100)."""
        tau = float(temperature) / 100.0
        if tau > 0.0:
            # Warm Amber
            r_gain = 1.0 + 0.16 * tau
            g_gain = 1.0 + 0.05 * tau
            b_gain = 1.0 - 0.18 * tau
        elif tau < 0.0:
            # Cool Blue
            abs_tau = abs(tau)
            r_gain = 1.0 - 0.20 * abs_tau
            g_gain = 1.0 + 0.02 * abs_tau
            b_gain = 1.0 + 0.22 * abs_tau
        else:
            r_gain, g_gain, b_gain = 1.0, 1.0, 1.0

        tint = np.array([r_gain, g_gain, b_gain], dtype=np.float32)
        return tint / np.mean(tint)

    @staticmethod
    def _parse_color_tint(color_str: str) -> np.ndarray:
        """Parses hex code (e.g. #35D6C5) into normalized RGB float array."""
        if not color_str:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)
        try:
            hex_clean = color_str.lstrip("#")
            if len(hex_clean) == 6:
                r = int(hex_clean[0:2], 16) / 255.0
                g = int(hex_clean[2:4], 16) / 255.0
                b = int(hex_clean[4:6], 16) / 255.0
                arr = np.array([r, g, b], dtype=np.float32)
                return arr / (np.mean(arr) + 1e-6)
        except Exception:
            pass
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def composite(
        self,
        base_image_np: np.ndarray,
        target_diffuse: np.ndarray,
        target_specular: np.ndarray,
        orig_diffuse: np.ndarray,
        orig_specular: np.ndarray,
        material_response_map: np.ndarray,
        micro_relief_field: np.ndarray,
        contact_shadow_field: np.ndarray,
        regions: Dict[str, np.ndarray],
        boundaries: Dict[str, np.ndarray],
        confidence_map: np.ndarray,
        base_depth: Optional[np.ndarray] = None,
        refined_depth: Optional[np.ndarray] = None,
        normal_rgb: Optional[np.ndarray] = None,
        target_ndotl_raw: Optional[np.ndarray] = None,
        inferred_depth: Optional[np.ndarray] = None,
        micro_relief_highlight_map: Optional[np.ndarray] = None,
        micro_relief_shadow_map: Optional[np.ndarray] = None,
        directional_cast_shadow_map: Optional[np.ndarray] = None,
        ambient_occlusion_map: Optional[np.ndarray] = None,
        rim_light_field: Optional[np.ndarray] = None,
        rim_light_color: str = "#35D6C5",
        fill_light_field: Optional[np.ndarray] = None,
        fill_light_color: str = "#4F9CFF",
        light_intensity: float = 100.0,
        light_temperature: float = 0.0,
        shadow_depth: float = 40.0,
        specular_strength: float = 50.0,
        ambient_fill: float = 30.0,
        contrast: float = 0.0,
        export_debug: bool = True,
        output_dir: str = "outputs/enhancer_debug/relighting",
    ) -> Dict[str, Any]:
        """
        Executes complete material-aware physical multi-light relighting and micro-relief compositing.
        """
        h, w, _ = base_image_np.shape

        # Normalize slider gains
        w_intensity = float(light_intensity) / 100.0
        w_shadow_depth = float(shadow_depth) / 100.0
        w_specular = float(specular_strength) / 100.0
        w_ambient = float(ambient_fill) / 100.0
        w_contrast = float(contrast) / 100.0

        # 1. Continuous Diffuse Residual from True Surface Normals (N · L)
        diff_delta = (target_diffuse - 0.46) * 0.85 * w_intensity

        # 2. Specular Highlights Residual
        spec_delta = target_specular * 0.55 * w_specular

        # 3. Micro-Relief Raised-Detail Shading Residual (Facing Highlights & Trailing Self-Shadows)
        relief_delta = micro_relief_field

        # 4. Contact Shadows & Directional Cast Occlusion Residual
        shadow_delta = -contact_shadow_field * (0.75 + 0.35 * w_shadow_depth)

        # 5. Ambient Fill Lift
        lum = (
            0.299 * base_image_np[..., 0]
            + 0.587 * base_image_np[..., 1]
            + 0.114 * base_image_np[..., 2]
        )
        ambient_delta = (1.0 - lum) * 0.10 * w_ambient

        # 6. Total Key Grayscale Illumination Delta
        total_key_delta = diff_delta + spec_delta + relief_delta + shadow_delta + ambient_delta

        # 7. Semantic Spatial Weighting & Protection
        m_subj = ndi.gaussian_filter(
            regions.get("subject", np.ones((h, w), dtype=np.float32)), sigma=2.0
        )
        m_bg = ndi.gaussian_filter(
            regions.get("background", np.zeros((h, w), dtype=np.float32)), sigma=2.0
        )
        m_face = ndi.gaussian_filter(
            regions.get("face", np.zeros((h, w), dtype=np.float32)), sigma=3.0
        )

        weight_map = m_subj * 0.95 + m_bg * 0.70
        weight_map = weight_map * (1.0 - m_face * 0.50)

        conf_smooth = ndi.gaussian_filter(confidence_map, sigma=1.5)
        effective_weight = np.clip(weight_map * conf_smooth, 0.0, 1.0)[..., None]

        # 8. Apply Key Light Color Temperature Chromaticity
        tint_rgb = self._compute_color_tint(light_temperature)  # [3]
        colored_delta = (total_key_delta[..., None] * tint_rgb) * effective_weight

        # 9. Multi-Light Rig: Rim Light Component
        rim_delta = np.zeros((h, w, 3), dtype=np.float32)
        if rim_light_field is not None and np.max(rim_light_field) > 0.001:
            rim_tint = self._parse_color_tint(rim_light_color)
            rim_delta = (rim_light_field[..., None] * rim_tint) * 0.75 * effective_weight

        # 10. Multi-Light Rig: Fill Light Component
        fill_delta = np.zeros((h, w, 3), dtype=np.float32)
        if fill_light_field is not None and np.max(fill_light_field) > 0.001:
            fill_tint = self._parse_color_tint(fill_light_color)
            fill_delta = (fill_light_field[..., None] * fill_tint) * 0.45 * effective_weight

        # 11. Final Blended Output
        relit = base_image_np + colored_delta + rim_delta + fill_delta

        if abs(w_contrast) > 0.01:
            relit = (relit - 0.5) * (1.0 + w_contrast * 0.35) + 0.5

        final_image = np.clip(relit, 0.0, 1.0)
        diff_field = np.clip(np.abs(final_image - base_image_np) * 3.0, 0.0, 1.0)

        # 12. Export Required Diagnostic Maps
        if export_debug:
            os.makedirs(output_dir, exist_ok=True)
            if base_depth is not None:
                Image.fromarray((np.clip(base_depth * 255.0, 0, 255)).astype(np.uint8)).save(
                    os.path.join(output_dir, "depth_map.png")
                )
            if refined_depth is not None:
                Image.fromarray((np.clip(refined_depth * 255.0, 0, 255)).astype(np.uint8)).save(
                    os.path.join(output_dir, "refined_depth_map.png")
                )
            if normal_rgb is not None:
                Image.fromarray(normal_rgb).save(
                    os.path.join(output_dir, "normal_map_rgb.png")
                )
            Image.fromarray((np.clip(target_diffuse * 255.0, 0, 255)).astype(np.uint8)).save(
                os.path.join(output_dir, "diffuse_ndotl_map.png")
            )
            Image.fromarray((np.clip(final_image * 255.0, 0, 255)).astype(np.uint8)).save(
                os.path.join(output_dir, "final_relit_image.png")
            )
            Image.fromarray((np.clip(diff_field * 255.0, 0, 255)).astype(np.uint8)).save(
                os.path.join(output_dir, "amplified_difference_map.png")
            )

            if micro_relief_highlight_map is not None:
                Image.fromarray((np.clip(micro_relief_highlight_map * 255.0, 0, 255)).astype(np.uint8)).save(
                    os.path.join(output_dir, "micro_relief_highlight_map.png")
                )
            if micro_relief_shadow_map is not None:
                Image.fromarray((np.clip(micro_relief_shadow_map * 255.0, 0, 255)).astype(np.uint8)).save(
                    os.path.join(output_dir, "micro_relief_shadow_map.png")
                )
            if directional_cast_shadow_map is not None:
                Image.fromarray((np.clip(directional_cast_shadow_map * 255.0, 0, 255)).astype(np.uint8)).save(
                    os.path.join(output_dir, "directional_cast_shadow_map.png")
                )
            if rim_light_field is not None:
                Image.fromarray((np.clip(rim_light_field * 255.0, 0, 255)).astype(np.uint8)).save(
                    os.path.join(output_dir, "rim_light_map.png")
                )
            if fill_light_field is not None:
                Image.fromarray((np.clip(fill_light_field * 255.0, 0, 255)).astype(np.uint8)).save(
                    os.path.join(output_dir, "fill_light_map.png")
                )

        return {
            "final_image": final_image,
            "lighting_field": target_diffuse,
            "specular_field": target_specular,
            "rim_light_field": rim_light_field,
            "fill_light_field": fill_light_field,
            "material_response_map": material_response_map,
            "micro_relief_field": micro_relief_field,
            "micro_relief_highlight_map": micro_relief_highlight_map,
            "micro_relief_shadow_map": micro_relief_shadow_map,
            "contact_shadow_field": contact_shadow_field,
            "directional_cast_shadow_map": directional_cast_shadow_map,
            "base_depth": base_depth,
            "refined_depth": refined_depth,
            "normal_rgb": normal_rgb,
            "effective_weight": effective_weight[..., 0],
            "difference_map": diff_field,
        }


relighting_compositor = RelightingCompositor()
