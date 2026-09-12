"""
MaterialRenderer: Physical Micro-Surface & Subsurface Scattering Engine.
Simulates physical material properties on upscaled canvases:
1. Thread Depth & Weave Relief: Crest specular glints vs. trough micro-ambient occlusion.
2. Subsurface Scattering (SSS): Epidermal light diffusion and warm terminator glow on skin.
3. Micro-Contrast Separation: Distinguishes specular highlights from deep fabric crevices.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class MaterialRenderer:
    """
    Physical surface response & micro-geometry enhancement for image upscaling.
    """

    def __init__(
        self,
        crest_strength: float = 0.35,
        trough_depth: float = 0.40,
        sss_strength: float = 0.30,
        micro_grain: float = 0.08,
    ):
        self.crest_strength = crest_strength
        self.trough_depth = trough_depth
        self.sss_strength = sss_strength
        self.micro_grain = micro_grain

    @staticmethod
    def _detect_skin_mask(img_rgb_float: np.ndarray) -> np.ndarray:
        """
        Detects skin regions using robust YCrCb and HSV thresholding.
        Returns a smooth float32 mask [0.0, 1.0].
        """
        img_uint8 = (img_rgb_float * 255.0).clip(0, 255).astype(np.uint8)
        ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

        # Standard physiological skin color bounds
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        cond_ycrcb = (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 130)
        cond_hsv = ((h <= 25) | (h >= 165)) & (s >= 20) & (s <= 180) & (v >= 40)

        skin_raw = (cond_ycrcb & cond_hsv).astype(np.float32)
        skin_smooth = cv2.GaussianBlur(skin_raw, (15, 15), 5.0)
        return skin_smooth

    def apply_crest_trough_depth(
        self,
        img_rgb_float: np.ndarray,
        crest_strength: Optional[float] = None,
        trough_depth: Optional[float] = None,
    ) -> np.ndarray:
        """
        Separates microscopic peaks (crests) from valleys (troughs).
        - Crests: Specular fiber/pore highlights are boosted with a non-linear glint curve.
        - Troughs: Crevices and fabric weave valleys receive localized ambient occlusion (AO).
        Operates in pure float32 space to prevent quantization artifacts.
        """
        c_str = self.crest_strength if crest_strength is None else crest_strength
        t_str = self.trough_depth if trough_depth is None else trough_depth

        if c_str <= 0.0 and t_str <= 0.0:
            return img_rgb_float

        img_f32 = img_rgb_float.astype(np.float32)
        # Compute accurate luminance (Rec. 709)
        l_chan = (
            0.2126 * img_f32[:, :, 0]
            + 0.7152 * img_f32[:, :, 1]
            + 0.0722 * img_f32[:, :, 2]
        ).astype(np.float32)

        # Multi-scale decomposition
        # Fine scale: captures individual thread weaves, hairs, pores (sigma=1.0)
        # Base scale: local ambient lighting (sigma=4.5)
        fine_blur = cv2.GaussianBlur(l_chan, (0, 0), 1.0)
        base_blur = cv2.GaussianBlur(l_chan, (0, 0), 4.5)

        # Micro-difference: positive = ridge/crest, negative = valley/trough
        diff = fine_blur - base_blur

        # Crest mask: positive micro-elevations
        crests = np.maximum(diff, 0.0)
        crest_boost = np.power(crests * 3.5, 1.1) * c_str

        # Trough mask: negative micro-depressions (crevices between threads)
        troughs = np.maximum(-diff, 0.0)
        # Micro Ambient Occlusion (AO): localized self-shadowing in crevices
        trough_ao = np.power(troughs * 3.5, 1.1) * t_str

        # Apply to luminance: crests gain light, troughs deepen
        l_enhanced = np.clip(l_chan + crest_boost - trough_ao, 0.0, 1.0)

        # Modulate RGB channels with ratio of enhanced luminance to original
        scale_factor = l_enhanced / (l_chan + 1e-6)
        # Soft clamp extreme multipliers to maintain photographic stability
        scale_factor = np.clip(scale_factor, 0.2, 2.5)

        enhanced_rgb = np.clip(img_f32 * scale_factor[:, :, None], 0.0, 1.0)
        return enhanced_rgb

    def apply_subsurface_scattering(
        self,
        img_rgb_float: np.ndarray,
        sss_strength: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simulates optical subsurface scattering (SSS):
        1. Identifies shadow terminators (half-light transition boundaries) on skin.
        2. Injects epidermal chromatic light bleeding (warm amber/crimson scatter).
        3. Softens subsurface chrominance while keeping luminance micro-pores razor sharp.
        """
        strength = self.sss_strength if sss_strength is None else sss_strength
        if strength <= 0.0:
            return img_rgb_float

        img_f32 = img_rgb_float.astype(np.float32)
        skin_mask = self._detect_skin_mask(img_f32)
        if np.max(skin_mask) < 0.02:
            return img_f32

        # Calculate luminance
        luminance = (
            0.299 * img_f32[:, :, 0]
            + 0.587 * img_f32[:, :, 1]
            + 0.114 * img_f32[:, :, 2]
        ).astype(np.float32)

        if luminance.shape[0] < 2 or luminance.shape[1] < 2:
            return img_f32

        # Pure numpy gradient to avoid OpenCV buffer format issues
        grad_y, grad_x = np.gradient(luminance)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_norm = grad_mag / (np.max(grad_mag) + 1e-6)

        # The terminator line is where skin transitions between light and shadow (midtones: 0.25 - 0.75)
        midtone_gate = np.exp(-((luminance - 0.5) ** 2) / 0.08)
        terminator_mask = grad_norm * midtone_gate * skin_mask
        terminator_mask = cv2.GaussianBlur(terminator_mask, (9, 9), 3.0)

        # SSS color profile: warm amber/red subcutaneous scatter
        r_scatter = terminator_mask * (strength * 0.18)
        g_scatter = terminator_mask * (strength * 0.06)
        b_scatter = -terminator_mask * (strength * 0.05)

        out_rgb = img_f32.copy()
        out_rgb[:, :, 0] += r_scatter
        out_rgb[:, :, 1] += g_scatter
        out_rgb[:, :, 2] += b_scatter

        # Subsurface chrominance blur: light scatters beneath the surface
        blur_rgb = cv2.GaussianBlur(out_rgb, (7, 7), 2.5)
        skin_factor = skin_mask[:, :, None] * (strength * 0.35)
        out_rgb = out_rgb * (1.0 - skin_factor) + blur_rgb * skin_factor

        return np.clip(out_rgb, 0.0, 1.0)

    def process_pil(
        self,
        pil_image: Image.Image,
        crest_strength: Optional[float] = None,
        trough_depth: Optional[float] = None,
        sss_strength: Optional[float] = None,
    ) -> Image.Image:
        """
        Full material and subsurface rendering pass on a PIL Image.
        """
        img_np = np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.0

        # Step 1: Subsurface scattering on skin & organic tissue
        img_sss = self.apply_subsurface_scattering(img_np, sss_strength=sss_strength)

        # Step 2: Crest & trough micro-geometry relief (threads, fabric weave, pores)
        img_relief = self.apply_crest_trough_depth(
            img_sss, crest_strength=crest_strength, trough_depth=trough_depth
        )

        out_uint8 = (img_relief * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out_uint8, mode="RGB")


material_renderer = MaterialRenderer()
