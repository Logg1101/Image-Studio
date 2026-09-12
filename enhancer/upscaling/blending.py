import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi
from typing import Dict, Any, Optional
from PIL import Image

from enhancer.types import SemanticAnalysisResult


class SemanticMaskBlender:
    """
    Semantic-Aware Multi-Layer Detail Reconstruction & Mask Blender.
    Applies region-specific enhancement policies and boundary defenses to synthesize
    material-accurate high-frequency details while strictly anchoring output to the source image.
    """

    @staticmethod
    def _resample_mask(mask_b64: str, target_h: int, target_w: int) -> np.ndarray:
        """Decodes base64 mask and resamples to target dimensions [target_h, target_w]."""
        import base64, io
        if "base64," in mask_b64:
            mask_b64 = mask_b64.split("base64,")[1]
        raw_bytes = base64.b64decode(mask_b64)
        pil_mask = Image.open(io.BytesIO(raw_bytes)).convert("L")
        if pil_mask.size != (target_w, target_h):
            pil_mask = pil_mask.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.array(pil_mask, dtype=np.float32) / 255.0

    @staticmethod
    def _apply_unsharp_mask(
        image_np: np.ndarray, strength: float, sigma: float = 1.0
    ) -> np.ndarray:
        """Applies Gaussian unsharp masking for controlled edge sharpening."""
        if strength <= 0.0:
            return image_np
        blurred = np.zeros_like(image_np)
        for c in range(3):
            blurred[..., c] = ndi.gaussian_filter(image_np[..., c], sigma=sigma)
        high_pass = image_np - blurred
        sharpened = image_np + high_pass * strength
        return np.clip(sharpened, 0.0, 1.0)

    @staticmethod
    def _apply_denoise(image_np: np.ndarray, strength: float) -> np.ndarray:
        """Applies conservative spatial bilateral/gaussian denoising."""
        if strength <= 0.0:
            return image_np
        sigma = 0.5 + (strength * 0.8)
        denoised = np.zeros_like(image_np)
        for c in range(3):
            denoised[..., c] = ndi.gaussian_filter(image_np[..., c], sigma=sigma)
        # Blend with original
        return np.clip(image_np * (1.0 - strength * 0.5) + denoised * (strength * 0.5), 0.0, 1.0)

    def blend(
        self,
        base_upscaled: np.ndarray,
        neural_upscaled: np.ndarray,
        semantic_result: SemanticAnalysisResult,
        detail_recovery: float = 50.0,
        texture_synthesis: float = 40.0,
        sharpen_strength: float = 30.0,
        denoise_strength: float = 20.0,
        face_restoration_strength: float = 75.0,
    ) -> Dict[str, Any]:
        """
        Executes material-aware detail blending.

        Args:
            base_upscaled (np.ndarray): Original image upscaled via high-order interpolation [H, W, 3] in [0, 1].
            neural_upscaled (np.ndarray): Neural upscaled reconstruction [H, W, 3] in [0, 1].
            semantic_result (SemanticAnalysisResult): Phase 2 perception hierarchy.

        Returns:
            Dict[str, Any]: Final composite image, detail residual, and diagnostic blend maps.
        """
        target_h, target_w, _ = base_upscaled.shape

        # Normalize control sliders to [0.0, 1.0]
        w_detail = detail_recovery / 100.0
        w_texture = texture_synthesis / 100.0
        w_sharpen = sharpen_strength / 100.0
        w_denoise = denoise_strength / 100.0
        w_face_restore = face_restoration_strength / 100.0

        # Optional Denoise on Base
        base_cleaned = self._apply_denoise(base_upscaled, strength=w_denoise)

        # -------------------------------------------------------------
        # 1. Resample Semantic Masks to Target Resolution
        # -------------------------------------------------------------
        m_subj = self._resample_mask(semantic_result.regions["subject"].mask_b64, target_h, target_w)
        m_bg = self._resample_mask(semantic_result.regions["background"].mask_b64, target_h, target_w)
        m_skin = self._resample_mask(semantic_result.regions["skin"].mask_b64, target_h, target_w)
        m_hair = self._resample_mask(semantic_result.regions["hair"].mask_b64, target_h, target_w)
        m_cloth = self._resample_mask(semantic_result.regions["clothing"].mask_b64, target_h, target_w)
        m_face = self._resample_mask(semantic_result.regions["face"].mask_b64, target_h, target_w)
        m_acc = self._resample_mask(semantic_result.regions["accessories"].mask_b64, target_h, target_w)
        conf_map = self._resample_mask(semantic_result.confidence_map_b64, target_h, target_w)

        # -------------------------------------------------------------
        # 2. Compute Neural High-Frequency Detail Residual
        # -------------------------------------------------------------
        # High-frequency residual = Neural Reconstruction - Base
        raw_detail_residual = neural_upscaled - base_cleaned

        # -------------------------------------------------------------
        # 3. Material-Specific Enhancement Policies
        # -------------------------------------------------------------
        # Policy Weights:
        # - Background: 0.35 * w_detail (conservative, suppresses noise hallucination)
        # - Skin: 0.45 * w_detail (preserves smooth gradients and anime cel-look)
        # - Face: 0.25 * (1.0 - w_face_restore * 0.5) (Highest protection; preserves eye & lip geometry)
        # - Hair: 0.85 * w_detail + 0.30 * w_texture (sharp anisotropic strand recovery)
        # - Clothing: 0.75 * w_detail + 0.40 * w_texture (texture weave & seam crispness)
        # - Accessories: 0.90 * w_detail (hard chrome highlight preservation)
        region_weight_map = np.zeros((target_h, target_w), dtype=np.float32)

        region_weight_map += m_bg * (0.35 * w_detail)
        region_weight_map += m_skin * (0.45 * w_detail)
        region_weight_map += m_hair * (0.85 * w_detail + 0.30 * w_texture)
        region_weight_map += m_cloth * (0.75 * w_detail + 0.40 * w_texture)
        region_weight_map += m_acc * (0.90 * w_detail)

        # Subject Residual: Any subject area not covered by sub-masks (e.g. wings, armor, props)
        m_sub_parts = np.clip(m_skin + m_hair + m_cloth + m_acc, 0.0, 1.0)
        m_subj_other = np.clip(m_subj - m_sub_parts, 0.0, 1.0)
        region_weight_map += m_subj_other * (0.60 * w_detail + 0.20 * w_texture)

        # Face Protection override: face detail is strictly conservative to protect identity
        face_attenuation = 0.20 + 0.30 * w_face_restore
        region_weight_map[m_face > 0.3] = np.minimum(
            region_weight_map[m_face > 0.3], face_attenuation
        )

        # -------------------------------------------------------------
        # 4. Boundary Protection Gating
        # -------------------------------------------------------------
        # Attenuate detail weight near transition boundaries and low confidence
        effective_weight = region_weight_map * conf_map
        effective_weight = np.clip(effective_weight, 0.0, 1.2)[..., None]  # [H, W, 1]

        # Synthesize Region-Guided Reconstruction
        reconstructed = base_cleaned + raw_detail_residual * effective_weight
        reconstructed = np.clip(reconstructed, 0.0, 1.0)

        # -------------------------------------------------------------
        # 5. Semantic-Guided Selective Sharpening
        # -------------------------------------------------------------
        if w_sharpen > 0.0:
            # Sharpen hair and clothing more than skin and face
            sharpen_mask = (
                m_hair * 1.0
                + m_cloth * 0.8
                + m_acc * 1.0
                + m_subj_other * 0.6
                + m_skin * 0.2
                + m_face * 0.1
            )
            sharpen_mask = np.clip(sharpen_mask * conf_map, 0.0, 1.0)[..., None]

            sharpened_full = self._apply_unsharp_mask(reconstructed, strength=w_sharpen * 0.8)
            final_output = reconstructed * (1.0 - sharpen_mask) + sharpened_full * sharpen_mask
        else:
            final_output = reconstructed

        final_output = np.clip(final_output, 0.0, 1.0)

        return {
            "final_image": final_output,
            "detail_residual": np.clip(raw_detail_residual * 0.5 + 0.5, 0.0, 1.0),
            "effective_weight_map": effective_weight[..., 0],
            "base_cleaned": base_cleaned,
            "neural_upscaled": neural_upscaled,
        }


semantic_mask_blender = SemanticMaskBlender()
