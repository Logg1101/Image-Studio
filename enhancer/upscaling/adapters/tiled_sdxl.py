import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple

from enhancer.upscaling.base import BaseUpscalerAdapter
from enhancer.perception.vram_manager import vram_manager
from core.device import get_torch_device, get_torch_dtype


class TiledSDXLUpscaler:
    """
    Seamless Overlap-Aware Tiled SDXL Generative Diffusion Upscaler.
    Slices large canvases into 1024x1024 tiles with 128px overlap, runs low-denoising (0.35)
    img2img diffusion for micro-detail synthesis, and composites with 2D cosine Hann masks.
    """

    def __init__(self, tile_size: int = 1024, overlap: int = 128):
        self.tile_size = tile_size
        self.overlap = overlap

    def _get_tiles(self, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
        stride = self.tile_size - self.overlap
        x_starts = list(range(0, max(1, img_w - self.tile_size + 1), stride))
        if not x_starts or x_starts[-1] + self.tile_size < img_w:
            x_starts.append(max(0, img_w - self.tile_size))
        y_starts = list(range(0, max(1, img_h - self.tile_size + 1), stride))
        if not y_starts or y_starts[-1] + self.tile_size < img_h:
            y_starts.append(max(0, img_h - self.tile_size))

        x_starts = sorted(list(set(x_starts)))
        y_starts = sorted(list(set(y_starts)))
        return [(x, y, x + self.tile_size, y + self.tile_size) for y in y_starts for x in x_starts]

    def _build_directional_tile_mask(
        self, h: int, w: int, overlap: int,
        ramp_top: bool, ramp_bottom: bool, ramp_left: bool, ramp_right: bool
    ) -> np.ndarray:
        wy = torch.ones(h, dtype=torch.float32)
        if ramp_top and overlap > 0:
            wy[:overlap] = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, overlap))
        if ramp_bottom and overlap > 0:
            wy[-overlap:] = 0.5 + 0.5 * torch.cos(torch.linspace(0, torch.pi, overlap))

        wx = torch.ones(w, dtype=torch.float32)
        if ramp_left and overlap > 0:
            wx[:overlap] = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, overlap))
        if ramp_right and overlap > 0:
            wx[-overlap:] = 0.5 + 0.5 * torch.cos(torch.linspace(0, torch.pi, overlap))

        return (wy[:, None] * wx[None, :]).numpy()

    def upscale(
        self,
        image: Image.Image,
        scale: float = 2.0,
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 0.35,
        steps: int = 20,
        seed: int = 12345,
        pipeline=None,
        step_callback=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        enable_tactile_materials: bool = True,
        crest_strength: float = 0.35,
        trough_depth: float = 0.40,
        sss_strength: float = 0.30,
    ) -> Image.Image:
        from diffusers import AutoPipelineForImage2Image
        from enhancer.upscaling.adapters.realcugan import RealCUGANAdapter

        orig_w, orig_h = image.size
        target_w = int(orig_w * scale)
        target_h = int(orig_h * scale)

        # 1. Base pre-upscale using Real-CUGAN for pristine line and color bases
        try:
            cugan = RealCUGANAdapter()
            if cugan.is_available:
                base_scaled = cugan.upscale_pil(image, scale=int(round(scale)))
                if base_scaled.size != (target_w, target_h):
                    base_scaled = base_scaled.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                base_scaled = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        except Exception:
            base_scaled = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Ensure dimensions are at least 512 for SDXL diffusion and multiples of 8
        eff_w = max(512, (target_w // 8) * 8)
        eff_h = max(512, (target_h // 8) * 8)

        if base_scaled.size != (eff_w, eff_h):
            base_scaled = base_scaled.resize((eff_w, eff_h), Image.Resampling.LANCZOS)

        # 2. Material & Subsurface Pre-Emphasis (Thread Depth, Crests/Troughs & SSS)
        if enable_tactile_materials:
            try:
                from enhancer.upscaling.material_renderer import material_renderer
                base_scaled = material_renderer.process_pil(
                    base_scaled,
                    crest_strength=crest_strength,
                    trough_depth=trough_depth,
                    sss_strength=sss_strength,
                )
            except Exception as e:
                import logging
                logging.getLogger("TiledSDXL").warning(f"Material pre-emphasis skipped: {e}")

        if eff_w <= self.tile_size and eff_h <= self.tile_size:
            tiles = [(0, 0, eff_w, eff_h)]
        else:
            tiles = self._get_tiles(eff_w, eff_h)

        # 3. Acquire or construct SDXL Img2Img Pipeline
        img2img_pipe = None
        loaded_pipe = None
        should_unload_pipe = False

        if pipeline is not None:
            img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipeline)
        else:
            # Check backend coordinator for active engine
            try:
                import backend_bridge
                if getattr(backend_bridge, "coordinator", None) and backend_bridge.coordinator.active_engine:
                    engine = backend_bridge.coordinator.active_engine
                    if getattr(engine, "pipeline", None) is not None:
                        img2img_pipe = AutoPipelineForImage2Image.from_pipe(engine.pipeline)
            except Exception:
                pass
            if img2img_pipe is None:
                try:
                    from ImageStudio_Tauri.backend_bridge import coordinator
                    if coordinator.active_engine and getattr(coordinator.active_engine, "pipeline", None) is not None:
                        img2img_pipe = AutoPipelineForImage2Image.from_pipe(coordinator.active_engine.pipeline)
                except Exception:
                    pass

        if img2img_pipe is None:
            # Standalone load default model
            from core.model_manager import ModelManager
            from engines.sdxl.loader import load_sdxl_pipeline
            mm = ModelManager()
            models = list(mm.available_models.values())
            chosen = next((m for m in models if "hassaku" in m.id.lower() and m.architecture == "sdxl"), None)
            if not chosen:
                chosen = next((m for m in models if m.architecture == "sdxl"), None)
            if not chosen:
                return base_scaled
            loaded_pipe, _ = load_sdxl_pipeline(chosen)
            img2img_pipe = AutoPipelineForImage2Image.from_pipe(loaded_pipe)
            should_unload_pipe = True

        pipe_device = getattr(img2img_pipe, "device", None) or get_torch_device()
        pipe_dtype = getattr(img2img_pipe, "dtype", None) or get_torch_dtype(pipe_device)
        generator = torch.Generator(pipe_device).manual_seed(seed if seed >= 0 else 12345)
        refined_prompt = prompt or "masterpiece, highly detailed, sharp focus, 8k resolution, cinematic lighting"
        refined_neg = negative_prompt or "blurry, low quality, artifacts, distorted"

        if enable_tactile_materials:
            tactile_pos = (
                "intricate woven fabric thread texture, visible micro-weave depth, thread crest specular highlights, "
                "self-shadowed troughs, subsurface scattering on skin, translucent epidermal glow, "
                "ambient occlusion in micro-crevices, physical material rendering, raytraced specular micro-contrast"
            )
            tactile_neg = (
                "flat 2d, smooth plastic, airbrushed skin, claymation, flat shading, washed out shadows, "
                "muddy troughs, blown highlights, featureless cloth"
            )
            refined_prompt = f"{refined_prompt}, {tactile_pos}"
            refined_neg = f"{refined_neg}, {tactile_neg}"
            # Re-encode with Compel to include tactile tokens in UNet cross-attention
            prompt_embeds = None

        # 4. Pre-encode prompt embeddings via Compel to prevent CLIP 77-token truncation
        if prompt_embeds is None and hasattr(img2img_pipe, "tokenizer") and hasattr(img2img_pipe, "text_encoder"):
            try:
                from compel import Compel, ReturnedEmbeddingsType
                compel = Compel(
                    tokenizer=[img2img_pipe.tokenizer, img2img_pipe.tokenizer_2],
                    text_encoder=[img2img_pipe.text_encoder, img2img_pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False
                )
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = compel(refined_prompt)
                    negative_prompt_embeds, negative_pooled_prompt_embeds = compel(refined_neg)
                    if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
                        empty_c, _ = compel("")
                        prompt_embeds, negative_prompt_embeds = compel.pad_conditioning_tensors_to_same_length(
                            [prompt_embeds, negative_prompt_embeds],
                            precomputed_padding=empty_c
                        )
                    prompt_embeds = prompt_embeds.to(device=pipe_device, dtype=pipe_dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(device=pipe_device, dtype=pipe_dtype)
                    negative_prompt_embeds = negative_prompt_embeds.to(device=pipe_device, dtype=pipe_dtype)
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=pipe_device, dtype=pipe_dtype)
            except Exception as e:
                import logging
                logging.getLogger("TiledSDXL").warning(f"Compel prompt encoding failed, falling back to raw prompt: {e}")

        canvas = np.zeros((eff_h, eff_w, 3), dtype=np.float32)
        weights = np.zeros((eff_h, eff_w, 1), dtype=np.float32)

        total_tiles = len(tiles)
        print(f"[TiledSDXL] Upscaling {eff_w}x{eff_h} ({total_tiles} tiles, strength={strength})...")

        from diffusers.utils import logging as diffusers_logging
        prev_verbosity = diffusers_logging.get_verbosity()
        diffusers_logging.set_verbosity_error()

        try:
            for idx, (x0, y0, x1, y1) in enumerate(tiles):
                tile_crop = base_scaled.crop((x0, y0, x1, y1))
                act_w, act_h = tile_crop.size

                mask = self._build_directional_tile_mask(
                    act_h, act_w, self.overlap,
                    ramp_top=(y0 > 0),
                    ramp_bottom=(y1 < eff_h),
                    ramp_left=(x0 > 0),
                    ramp_right=(x1 < eff_w)
                )

                if step_callback:
                    step_callback(idx + 1, total_tiles, f"Diffusing SDXL Tile {idx + 1}/{total_tiles}...")

                pipe_kwargs = {
                    "image": tile_crop,
                    "strength": strength,
                    "num_inference_steps": max(10, int(steps)),
                    "guidance_scale": 6.5,
                    "generator": generator,
                }
                if prompt_embeds is not None and pooled_prompt_embeds is not None:
                    pipe_kwargs["prompt_embeds"] = prompt_embeds
                    pipe_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
                    pipe_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
                    pipe_kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
                else:
                    pipe_kwargs["prompt"] = refined_prompt
                    pipe_kwargs["negative_prompt"] = refined_neg

                # Run diffusion img2img on this tile
                with torch.inference_mode():
                    res_tile = img2img_pipe(**pipe_kwargs).images[0]

                tile_np = np.array(res_tile, dtype=np.float32)
                canvas[y0:y1, x0:x1] += tile_np * mask[:, :, None]
                weights[y0:y1, x0:x1] += mask[:, :, None]

            final_np = (canvas / np.maximum(weights, 1e-6)).clip(0, 255).astype(np.uint8)
            final_img = Image.fromarray(final_np)
            if final_img.size != (target_w, target_h):
                final_img = final_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            return final_img

        finally:
            try:
                diffusers_logging.set_verbosity(prev_verbosity)
            except Exception:
                pass
            if should_unload_pipe:
                if img2img_pipe is not None:
                    del img2img_pipe
                if loaded_pipe is not None:
                    del loaded_pipe
                from core.memory import clear_vram
                clear_vram()


class TiledSDXLAdapter(BaseUpscalerAdapter):
    """
    Adapter wrapper around TiledSDXLUpscaler for Enhancer system registry.
    """

    def __init__(self):
        self._scaler = TiledSDXLUpscaler()

    @property
    def name(self) -> str:
        return "Tiled SDXL (Generative Diffusion Upscale)"

    @property
    def model_id(self) -> str:
        return "tiled-sdxl"

    @property
    def native_scale(self) -> int:
        return 2

    @property
    def is_available(self) -> bool:
        return True

    def supports_scale(self, scale: int) -> bool:
        return scale in (1, 2, 4)

    def estimated_vram_mb(self) -> float:
        return 2560.0

    def load(self, device: Optional[str] = None) -> None:
        pass

    def unload(self) -> None:
        pass

    def upscale(self, image_tensor: torch.Tensor, scale: int = 2) -> torch.Tensor:
        orig_device = image_tensor.device
        np_img = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img)

        out_pil = self._scaler.upscale(pil_img, scale=float(scale))
        out_np = np.array(out_pil, dtype=np.float32) / 255.0
        out_tensor = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0)
        return out_tensor.to(orig_device)

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "native_scale": self.native_scale,
            "supported_scales": [1, 2, 4],
            "is_available": self.is_available,
            "architecture": "Tiled SDXL (Diffusion Generative Super-Resolution)",
            "estimated_vram_mb": self.estimated_vram_mb(),
        }
