import os
import time
import io
import base64
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

from enhancer.analyzer.semantic_analyzer import SemanticAnalyzer
from enhancer.upscaling.registry import upscaler_registry
from enhancer.upscaling.tiled import tiled_upscaler
from enhancer.upscaling.blending import semantic_mask_blender
from enhancer.upscaling.validation import output_validator
from enhancer.perception.vram_manager import vram_manager
from enhancer.lighting.relighting.engine import relighting_engine
from enhancer.lighting.relighting.compositor import relighting_compositor
from enhancer.lighting.relighting.validation import relighting_validator


class EnhancementEngine:
    """
    Primary Unified Neural Enhancement & Semantic Relighting Engine.
    Orchestrates Neural Perception, Tiled Upscaling, Detail Synthesis,
    Material-Aware Relighting, and Quality Validation.
    """

    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.registry = upscaler_registry
        self.tiled_upscaler = tiled_upscaler
        self.blender = semantic_mask_blender
        self.relighter = relighting_engine
        self.relight_comp = relighting_compositor

    def _decode_mask(self, b64_str: str, target_h: int, target_w: int) -> np.ndarray:
        """Decodes base64 mask and resamples to [target_h, target_w]."""
        if "base64," in b64_str:
            b64_str = b64_str.split("base64,")[1]
        raw = base64.b64decode(b64_str)
        p = Image.open(io.BytesIO(raw)).convert("L")
        if p.size != (target_w, target_h):
            p = p.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.array(p, dtype=np.float32) / 255.0

    def enhance(
        self,
        image: Image.Image,
        scale: int = 2,
        model_id: str = "4x-realcugan",
        detail_recovery: float = 50.0,
        texture_synthesis: float = 40.0,
        sharpen: float = 30.0,
        denoise: float = 20.0,
        face_restoration: float = 75.0,
        relighting_enabled: bool = False,
        light_direction_angle: float = 287.5,
        light_intensity: float = 100.0,
        light_temperature: float = 0.0,
        shadow_recovery: float = 25.0,
        highlight_recovery: float = 50.0,
        ambient_light: float = 30.0,
        contrast: float = 0.0,
        micro_relief_strength: float = 50.0,
        shadow_depth: float = 40.0,
        specular_strength: float = 50.0,
        material_response: float = 100.0,
        rim_light_enabled: bool = False,
        rim_light_angle: float = 45.0,
        rim_light_intensity: float = 50.0,
        rim_light_color: str = "#35D6C5",
        fill_light_enabled: bool = False,
        fill_light_angle: float = 225.0,
        fill_light_intensity: float = 35.0,
        fill_light_color: str = "#4F9CFF",
        output_format: str = "PNG",
        output_dir: str = "outputs/upscale",
        export_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes complete neural enhancement, detail restoration, and optional semantic relighting.

        Returns:
            Dict[str, Any]: Enhancement result dictionary with telemetry.
        """
        start_time = time.time()
        orig_w, orig_h = image.size
        target_w, target_h = orig_w * scale, orig_h * scale

        vram_before = vram_manager.get_cuda_memory_mb()

        # -------------------------------------------------------------
        # 1. Phase 2 Neural Semantic Analysis
        # -------------------------------------------------------------
        t0_seg = time.time()
        semantic_result = self.semantic_analyzer.analyze(image, export_debug=export_debug)
        t_seg_ms = (time.time() - t0_seg) * 1000

        # -------------------------------------------------------------
        # 2. Select & Prepare Upscaler Adapter
        # -------------------------------------------------------------
        adapter = self.registry.get(model_id)
        if adapter is None or not adapter.is_available:
            raise FileNotFoundError(
                f"No suitable upscaler adapter could be initialized for '{model_id}'."
            )

        img_rgb = image.convert("RGB")
        base_upscaled_pil = img_rgb.resize(
            (target_w, target_h), Image.Resampling.BICUBIC
        )
        base_upscaled_np = np.array(base_upscaled_pil, dtype=np.float32) / 255.0

        # -------------------------------------------------------------
        # 3. Staged Neural Upscaling Inference (Tiled)
        # -------------------------------------------------------------
        t0_up = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with vram_manager.staged_execution(f"upscaling_{adapter.model_id}"):
            adapter.load(device=device)

            img_tensor = torch.from_numpy(
                np.array(img_rgb, dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)

            def _upscale_cb(tile: torch.Tensor, sc: int) -> torch.Tensor:
                return adapter.upscale(tile, scale=sc)

            neural_tensor = self.tiled_upscaler.upscale_tiled(
                image_tensor=img_tensor, upscale_fn=_upscale_cb, scale=scale
            )

            adapter.unload()

        t_up_ms = (time.time() - t0_up) * 1000
        neural_upscaled_np = (
            neural_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        )
        del neural_tensor

        # -------------------------------------------------------------
        # 4. Semantic-Aware Detail Reconstruction & Mask Blending
        # -------------------------------------------------------------
        t0_blend = time.time()
        blend_result = self.blender.blend(
            base_upscaled=base_upscaled_np,
            neural_upscaled=neural_upscaled_np,
            semantic_result=semantic_result,
            detail_recovery=detail_recovery,
            texture_synthesis=texture_synthesis,
            sharpen_strength=sharpen,
            denoise_strength=denoise,
            face_restoration_strength=face_restoration,
        )
        t_blend_ms = (time.time() - t0_blend) * 1000

        enhanced_np = blend_result["final_image"]

        # -------------------------------------------------------------
        # 5. Phase 4B Controlled Semantic Relighting (Optional)
        # -------------------------------------------------------------
        t_relight_ms = 0.0
        if relighting_enabled:
            t0_relight = time.time()
            lum_enhanced = (
                0.299 * enhanced_np[..., 0]
                + 0.587 * enhanced_np[..., 1]
                + 0.114 * enhanced_np[..., 2]
            )

            # Extract resampled masks at target resolution
            region_masks = {
                name: self._decode_mask(reg.mask_b64, target_h, target_w)
                for name, reg in semantic_result.regions.items()
            }
            boundary_masks = {
                name: self._decode_mask(b.mask_b64, target_h, target_w)
                for name, b in semantic_result.boundaries.items()
            }
            conf_map = self._decode_mask(
                semantic_result.confidence_map_b64, target_h, target_w
            )

            normals, tangents, refined_depth, base_depth, normal_rgb = self.relighter.estimate_surface_normals(
                luminance=lum_enhanced,
                regions=region_masks,
            )

            fields = self.relighter.compute_lighting_fields(
                luminance=lum_enhanced,
                normals=normals,
                tangents=tangents,
                regions=region_masks,
                target_angle_deg=light_direction_angle,
                orig_angle_deg=287.5,
                micro_relief_strength=micro_relief_strength,
                shadow_depth=shadow_depth,
                rim_light_enabled=rim_light_enabled,
                rim_light_angle=rim_light_angle,
                rim_light_intensity=rim_light_intensity,
                fill_light_enabled=fill_light_enabled,
                fill_light_angle=fill_light_angle,
                fill_light_intensity=fill_light_intensity,
            )

            relit_res = self.relight_comp.composite(
                base_image_np=enhanced_np,
                target_diffuse=fields["target_diffuse"],
                target_specular=fields["target_specular"],
                orig_diffuse=fields["orig_diffuse"],
                orig_specular=fields["orig_specular"],
                material_response_map=fields["material_response_map"],
                micro_relief_field=fields["micro_relief_field"],
                contact_shadow_field=fields["contact_shadow_field"],
                base_depth=base_depth,
                refined_depth=refined_depth,
                normal_rgb=normal_rgb,
                micro_relief_highlight_map=fields.get("micro_relief_highlight_map"),
                micro_relief_shadow_map=fields.get("micro_relief_shadow_map"),
                directional_cast_shadow_map=fields.get("directional_cast_shadow_map"),
                ambient_occlusion_map=fields.get("ambient_occlusion_map"),
                rim_light_field=fields.get("rim_light_field"),
                rim_light_color=rim_light_color,
                fill_light_field=fields.get("fill_light_field"),
                fill_light_color=fill_light_color,
                regions=region_masks,
                boundaries=boundary_masks,
                confidence_map=conf_map,
                light_intensity=light_intensity,
                light_temperature=light_temperature,
                shadow_depth=shadow_depth,
                specular_strength=specular_strength,
                ambient_fill=ambient_light,
                contrast=contrast,
                export_debug=export_debug,
                output_dir="outputs/enhancer_debug/relighting",
            )
            final_np = relit_res["final_image"]
            t_relight_ms = (time.time() - t0_relight) * 1000
        else:
            final_np = enhanced_np

        # -------------------------------------------------------------
        # 6. Output Validation
        # -------------------------------------------------------------
        val_report = output_validator.validate_output(
            final_np, expected_shape=(target_h, target_w, 3)
        )
        if not val_report["is_valid"]:
            raise ValueError(f"Output validation failed: {val_report['errors']}")

        # -------------------------------------------------------------
        # 7. Save Final Image & Export Diagnostics
        # -------------------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"enhanced_{timestamp}_{target_w}x{target_h}.{output_format.lower()}"
        out_filepath = os.path.join(output_dir, filename)

        final_uint8 = np.clip(final_np * 255.0, 0, 255).astype(np.uint8)
        final_pil = Image.fromarray(final_uint8, mode="RGB")
        final_pil.save(out_filepath, format=output_format.upper())

        buf = io.BytesIO()
        final_pil.save(buf, format="PNG")
        final_b64 = (
            f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        )

        if export_debug:
            debug_dir = "outputs/enhancer_debug"
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray(
                np.clip(base_upscaled_np * 255.0, 0, 255).astype(np.uint8)
            ).save(os.path.join(debug_dir, "phase3_original.png"))
            Image.fromarray(
                np.clip(neural_upscaled_np * 255.0, 0, 255).astype(np.uint8)
            ).save(os.path.join(debug_dir, "phase3_upscaled.png"))
            Image.fromarray(
                np.clip(blend_result["detail_residual"] * 255.0, 0, 255).astype(
                    np.uint8
                )
            ).save(os.path.join(debug_dir, "phase3_detail_residual.png"))
            Image.fromarray(
                np.clip(
                    blend_result["effective_weight_map"] * 255.0, 0, 255
                ).astype(np.uint8)
            ).save(os.path.join(debug_dir, "phase3_semantic_blend.png"))
            final_pil.save(os.path.join(debug_dir, "phase3_final.png"))

        total_time_ms = (time.time() - start_time) * 1000
        vram_after = vram_manager.get_cuda_memory_mb()

        return {
            "success": True,
            "output_path": out_filepath,
            "image_url": final_b64,
            "source_resolution": [orig_w, orig_h],
            "output_resolution": [target_w, target_h],
            "scale": scale,
            "engine_used": adapter.name,
            "model_id": adapter.model_id,
            "relighting_applied": relighting_enabled,
            "total_time_ms": total_time_ms,
            "telemetry": {
                "perception_time_ms": t_seg_ms,
                "upscaling_time_ms": t_up_ms,
                "blending_time_ms": t_blend_ms,
                "relighting_time_ms": t_relight_ms,
                "total_time_ms": total_time_ms,
                "vram_before": vram_before,
                "vram_after": vram_after,
                "tiled": (orig_h > 512 or orig_w > 512),
                "validation": val_report,
            },
            "warnings": val_report["warnings"],
        }


enhancement_engine = EnhancementEngine()
