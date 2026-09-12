import os
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional

from enhancer.lighting.types import LightingMaps, LightingProfile


class LightingDebugExporter:
    """
    Exports lighting debug artifacts to disk (outputs/enhancer_debug/lighting/)
    and generates Base64 Data URLs for frontend UI consumption.
    """

    @staticmethod
    def _array_to_b64(arr_np: np.ndarray, is_rgb: bool = False) -> str:
        """Converts float array in [0.0, 1.0] to base64 PNG data URL."""
        uint8_arr = np.clip(arr_np * 255.0, 0, 255).astype(np.uint8)
        mode = "RGB" if is_rgb else "L"
        pil_img = Image.fromarray(uint8_arr, mode=mode)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    def build_direction_vis(self, h: int, w: int, profile: LightingProfile) -> Image.Image:
        """Draws a directional visual compass arrow on dark canvas."""
        img = Image.new("RGB", (w, h), color=(15, 19, 27))
        draw = ImageDraw.Draw(img)

        # Draw grid
        draw.line([(0, h // 2), (w, h // 2)], fill=(37, 44, 58), width=1)
        draw.line([(w // 2, 0), (w // 2, h)], fill=(37, 44, 58), width=1)

        # Light source origin
        sx = int(profile.light_source_position[0] * w)
        sy = int(profile.light_source_position[1] * h)

        # Target center
        cx, cy = w // 2, h // 2

        # Draw light ray from source towards center
        draw.line([(sx, sy), (cx, cy)], fill=(245, 158, 11), width=4)
        draw.ellipse([(sx - 16, sy - 16), (sx + 16, sy + 16)], fill=(245, 158, 11), outline=(255, 255, 255), width=2)
        draw.ellipse([(cx - 8, cy - 8), (cx + 8, cy + 8)], fill=(59, 130, 246), outline=(255, 255, 255), width=1)

        # Text banner
        draw.text(
            (20, 20),
            f"Estimated Key Light: {profile.direction_angle_deg:.1f}° | Temp: {profile.color_temperature_k:.0f}K ({profile.color_bias.upper()})",
            fill=(245, 158, 11),
        )
        return img

    def build_composite_overlay(
        self,
        base_rgb: np.ndarray,
        highlights: np.ndarray,
        shadows: np.ndarray,
        profile: LightingProfile,
    ) -> Image.Image:
        """Constructs rich diagnostic overlay showing illumination vector, highlights, and shadow zones."""
        h, w, _ = base_rgb.shape
        overlay = base_rgb.copy()

        # Highlights in warm golden tint: [1.0, 0.8, 0.2]
        high_mask = (highlights > 0.3)[..., None]
        overlay = np.where(high_mask, overlay * 0.5 + np.array([0.5, 0.4, 0.1]) * highlights[..., None], overlay)

        # Shadows in cool indigo tint: [0.1, 0.1, 0.4]
        shadow_mask = (shadows > 0.35)[..., None]
        overlay = np.where(shadow_mask, overlay * 0.5 + np.array([0.05, 0.05, 0.25]) * shadows[..., None], overlay)

        uint8_overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(uint8_overlay, mode="RGB")
        draw = ImageDraw.Draw(pil_img)

        # Draw light direction vector
        sx = int(profile.light_source_position[0] * w)
        sy = int(profile.light_source_position[1] * h)
        cx, cy = w // 2, h // 2

        draw.line([(sx, sy), (cx, cy)], fill=(245, 158, 11), width=4)
        draw.ellipse([(sx - 14, sy - 14), (sx + 14, sy + 14)], fill=(245, 158, 11), outline=(255, 255, 255), width=2)

        # Diagnostic HUD
        draw.rectangle([(15, 15), (380, 85)], fill=(15, 19, 27))
        draw.text((25, 22), f"LIGHT ANGLE: {profile.direction_angle_deg:.1f}° ({profile.color_bias.upper()})", fill=(245, 158, 11))
        draw.text((25, 42), f"INTENSITY: {profile.key_intensity*100:.1f}% | CCT: {profile.color_temperature_k:.0f}K", fill=(232, 236, 244))
        draw.text((25, 62), f"SHADOW ANGLE: {profile.shadow_direction_deg:.1f}° | CONF: {profile.global_confidence*100:.1f}%", fill=(137, 147, 167))

        return pil_img

    def export_and_encode(
        self,
        base_rgb: np.ndarray,
        luminance: np.ndarray,
        shadows: np.ndarray,
        highlights: np.ndarray,
        ambient: np.ndarray,
        confidence: np.ndarray,
        profile: LightingProfile,
        output_dir: str = "outputs/enhancer_debug/lighting",
        save_disk: bool = True,
    ) -> LightingMaps:
        """Exports debug PNG files to disk and returns LightingMaps with base64 Data URLs."""
        h, w, _ = base_rgb.shape

        dir_vis = self.build_direction_vis(h, w, profile)
        comp_vis = self.build_composite_overlay(base_rgb, highlights, shadows, profile)

        if save_disk:
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray((np.clip(luminance * 255.0, 0, 255)).astype(np.uint8)).save(os.path.join(output_dir, "lighting_luminance.png"))
            Image.fromarray((np.clip(shadows * 255.0, 0, 255)).astype(np.uint8)).save(os.path.join(output_dir, "lighting_shadows.png"))
            Image.fromarray((np.clip(highlights * 255.0, 0, 255)).astype(np.uint8)).save(os.path.join(output_dir, "lighting_highlights.png"))
            Image.fromarray((np.clip(ambient * 255.0, 0, 255)).astype(np.uint8)).save(os.path.join(output_dir, "lighting_ambient.png"))
            Image.fromarray((np.clip(confidence * 255.0, 0, 255)).astype(np.uint8)).save(os.path.join(output_dir, "lighting_confidence.png"))
            dir_vis.save(os.path.join(output_dir, "lighting_direction.png"))
            comp_vis.save(os.path.join(output_dir, "lighting_overlay.png"))

        # Generate base64 Data URLs
        buf_dir = io.BytesIO()
        dir_vis.save(buf_dir, format="PNG")
        b64_dir = f"data:image/png;base64,{base64.b64encode(buf_dir.getvalue()).decode('utf-8')}"

        buf_comp = io.BytesIO()
        comp_vis.save(buf_comp, format="PNG")
        b64_comp = f"data:image/png;base64,{base64.b64encode(buf_comp.getvalue()).decode('utf-8')}"

        return LightingMaps(
            luminance_b64=self._array_to_b64(luminance),
            shadows_b64=self._array_to_b64(shadows),
            highlights_b64=self._array_to_b64(highlights),
            ambient_b64=self._array_to_b64(ambient),
            direction_vis_b64=b64_dir,
            confidence_b64=self._array_to_b64(confidence),
            composite_overlay_b64=b64_comp,
        )


lighting_debug_exporter = LightingDebugExporter()
