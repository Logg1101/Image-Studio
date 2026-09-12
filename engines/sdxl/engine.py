import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from PIL import PngImagePlugin, Image, ImageFilter
from compel import Compel, ReturnedEmbeddingsType

from engines.base import GenerationEngine
from core.types import GenerationRequest, GenerationResult, ModelInfo
from core.exceptions import GenerationError, UnsupportedModelError
from engines.sdxl.loader import load_sdxl_pipeline
from engines.sdxl.schedulers import build_sdxl_scheduler
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    AutoPipelineForInpainting,
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline
)
import config.paths as paths
from adapters.registry import adapter_registry
from core.memory import clear_vram
from core.device import get_torch_device, get_torch_dtype
from core.project_manager import ProjectManager
from core.usage_tracker import UsageTracker
from pipelines.regional_prompting import generate_regional_sdxl

class SDXLEngine(GenerationEngine):
    """
    Production-grade SDXL generation engine with universal adapter lifecycle integration,
    Compel multi-encoder prompt conditioning, and clean memory management.
    """
    def __init__(self):
        self.pipeline = None
        self.compel_processor = None
        self.current_model_info = None
        self.base_scheduler_config = None

    def supports(self, model_info: ModelInfo) -> bool:
        return model_info.architecture.lower() == "sdxl"

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def _log_cuda_diagnostics(self, stage: str) -> None:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
            res = torch.cuda.memory_reserved(0) / (1024 ** 2)
            max_alloc = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
            active_names = list(adapter_registry.active_loras.keys())
            print(f"[CUDA DIAGNOSTICS - {stage}] Allocated: {alloc:.2f} MB | Reserved: {res:.2f} MB | MaxAlloc: {max_alloc:.2f} MB | Active LoRAs ({len(active_names)}): {active_names}")

    def load(self, model_info: ModelInfo) -> None:
        if not self.supports(model_info):
            raise UnsupportedModelError(f"SDXLEngine cannot load architecture: {model_info.architecture}")

        self.unload()

        try:
            print(f"Loading model: {model_info.id} into SDXLEngine")
            self.pipeline, _ = load_sdxl_pipeline(model_info)
            self.current_model_info = model_info

            dev = get_torch_device()
            self.compel_processor = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False,
                device=dev,
            )

            self.base_scheduler_config = dict(self.pipeline.scheduler.config)
            self._log_cuda_diagnostics("After Model Load")
        except Exception as e:
            raise GenerationError(f"Failed to load SDXL model: {e}") from e

    def unload(self) -> None:
        if self.pipeline is not None:
            adapter_registry.unload_all(self.pipeline)
            del self.pipeline
            self.pipeline = None

        if self.compel_processor is not None:
            del self.compel_processor
            self.compel_processor = None

        self.current_model_info = None
        self.base_scheduler_config = None
        clear_vram()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self.is_loaded():
            raise GenerationError("Cannot generate: Model is not loaded.")

        ip_adapter_active = False
        controlnet_model = None
        controlnet_image = None

        self._log_cuda_diagnostics("Pre-Gen")

        device = getattr(self.pipeline, "device", None) or get_torch_device()

        # 1. Scheduler Setup
        sampler_name = request.sampler
        if request.init_image and request.mask_image and "DPM" in str(sampler_name):
            sampler_name = "Euler a"

        self.pipeline.scheduler = build_sdxl_scheduler(
            sampler_name,
            request.scheduler,
            self.base_scheduler_config,
        )
        try:
            self.pipeline.scheduler.set_timesteps(num_inference_steps=int(request.steps), device=device)
        except (TypeError, ValueError, RuntimeError):
            try:
                self.pipeline.scheduler.set_timesteps(num_inference_steps=int(request.steps))
            except (TypeError, ValueError, RuntimeError):
                pass

        # Multi-Character Regional Routing
        if getattr(request, "characters", None) and len(request.characters) >= 2:
            return self._generate_regional(request)

        # 2. Universal LoRA Synchronization (Reuses loaded adapters, 0 reload overhead)
        if request.loras is not None:
            adapter_registry.sync_loras(self.pipeline, request.loras, target_architecture="sdxl")
        else:
            adapter_registry.unload_all_loras(self.pipeline)

        start_time = time.time()
        generator = torch.Generator(device=device).manual_seed(int(request.seed))
        conditioning = None
        pooled = None
        neg_conditioning = None
        neg_pooled = None

        def diffusers_step_callback(pipe, step_index, timestep, callback_kwargs):
            if request.step_callback:
                request.step_callback(step_index + 1, request.steps)
            return callback_kwargs

        def legacy_step_callback(step: int, timestep: int, latents):
            if request.step_callback:
                request.step_callback(step + 1, request.steps)

        try:
            has_inpaint_compositing = False
            orig_init_image = None
            orig_raw_mask = None

            # 3. Prompt Conditioning (Compel with Long Sequence Alignment)
            negative_prompt = request.negative_prompt if request.negative_prompt else ""
            if request.init_image and request.mask_image:
                import numpy as np
                mask_chk = np.array(request.mask_image.convert("L"))
                if np.mean(mask_chk > 128) > 0.40:
                    anti_distort = "duplicate head, extra head, double face, extra face, cloned face, deformed neck, severed head, floating head, mutated anatomy, bad proportions, disfigured, malformed limbs"
                    negative_prompt = f"{negative_prompt}, {anti_distort}" if negative_prompt else anti_distort

            with torch.no_grad():
                conditioning, pooled = self.compel_processor(request.prompt)
                neg_conditioning, neg_pooled = self.compel_processor(negative_prompt)

                if conditioning.shape[1] != neg_conditioning.shape[1]:
                    empty_c, _ = self.compel_processor("")
                    conditioning, neg_conditioning = self.compel_processor.pad_conditioning_tensors_to_same_length(
                        [conditioning, neg_conditioning],
                        precomputed_padding=empty_c
                    )

                dev = get_torch_device()
                dt = get_torch_dtype(dev)
                conditioning = conditioning.contiguous().to(device=dev, dtype=dt)
                pooled = pooled.contiguous().to(device=dev, dtype=dt)
                neg_conditioning = neg_conditioning.contiguous().to(device=dev, dtype=dt)
                neg_pooled = neg_pooled.contiguous().to(device=dev, dtype=dt)

            # 4. ControlNet & IP-Adapter Preparation
            if request.controlnet_type and request.controlnet_type.lower() != "none":
                cnets = adapter_registry.scan_controlnets()
                cnet_key = request.controlnet_type.lower()
                if cnet_key in cnets:
                    cnet_path = cnets[cnet_key]
                    dev = get_torch_device()
                    dt = get_torch_dtype(dev)
                    if cnet_path.is_dir():
                        controlnet_model = ControlNetModel.from_pretrained(str(cnet_path), torch_dtype=dt).to(dev)
                    else:
                        controlnet_model = ControlNetModel.from_single_file(str(cnet_path), torch_dtype=dt).to(dev)

                    raw_ctrl = request.controlnet_image if request.controlnet_image else request.init_image
                    if raw_ctrl:
                        controlnet_image = raw_ctrl.resize((int(request.width), int(request.height)), Image.Resampling.LANCZOS)

            if request.ip_adapter_name and request.ip_adapter_name.lower() != "none":
                ips = adapter_registry.scan_ip_adapters()
                ip_key = request.ip_adapter_name.lower()
                if ip_key in ips:
                    ip_path = ips[ip_key]
                    if hasattr(self.pipeline, "load_ip_adapter"):
                        try:
                            dev = get_torch_device()
                            dt = get_torch_dtype(dev)
                            if getattr(self.pipeline, "image_encoder", None) is None:
                                from transformers import CLIPVisionModelWithProjection
                                try:
                                    self.pipeline.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                                        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                                        torch_dtype=dt,
                                        local_files_only=True,
                                    ).to(dev)
                                except Exception:
                                    self.pipeline.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                                        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                                        torch_dtype=dt,
                                    ).to(dev)

                            self.pipeline.load_ip_adapter(
                                str(ip_path.parent),
                                subfolder="",
                                weight_name=ip_path.name,
                            )
                            if hasattr(self.pipeline, "set_ip_adapter_scale"):
                                self.pipeline.set_ip_adapter_scale(float(request.ip_adapter_scale))
                            ip_adapter_active = True
                            print(f"[SDXLEngine] IP-Adapter attached: {ip_path.name} (scale: {request.ip_adapter_scale})")
                        except Exception as err:
                            raise GenerationError(f"Failed to attach IP-Adapter '{request.ip_adapter_name}': {err}") from err
                else:
                    raise GenerationError(f"Requested IP-Adapter '{request.ip_adapter_name}' not found in registry.")

            ip_adapter_ref_img = request.pulid_image if (request.pulid_enabled and request.pulid_image) else request.ip_adapter_image

            # 5. Image Generation (Txt2Img vs Img2Img Routing)
            pipe_kwargs = {
                "prompt_embeds": conditioning,
                "pooled_prompt_embeds": pooled,
                "negative_prompt_embeds": neg_conditioning,
                "negative_pooled_prompt_embeds": neg_pooled,
                "num_inference_steps": int(request.steps),
                "guidance_scale": float(request.guidance_scale),
                "generator": generator,
                "output_type": "pil",
                "callback_on_step_end": diffusers_step_callback,
                "callback_on_step_end_tensor_inputs": ["latents"],
                "callback": legacy_step_callback,
                "callback_steps": 1
            }

            if ip_adapter_active and ip_adapter_ref_img:
                pipe_kwargs["ip_adapter_image"] = ip_adapter_ref_img

            if request.init_image and request.mask_image:
                req_w = int(request.width)
                req_h = int(request.height)

                init_rgb = request.init_image.convert("RGB")
                if init_rgb.size != (req_w, req_h):
                    init_rgb = init_rgb.resize((req_w, req_h), Image.Resampling.LANCZOS)

                raw_mask = request.mask_image.convert("L")
                if raw_mask.size != (req_w, req_h):
                    raw_mask = raw_mask.resize((req_w, req_h), Image.Resampling.NEAREST)

                import numpy as np
                mask_np = np.array(raw_mask)
                # Binary mask (255 = inpaint from prompt, 0 = keep pristine)
                # Keep binary during UNet denoising so intermediate latents are not blurred across the mask
                bin_mask = raw_mask.point(lambda p: 255 if p > 128 else 0)

                effective_strength = float(request.denoising_strength if request.denoising_strength is not None else 0.85)
                effective_strength = max(0.2, min(1.0, effective_strength))

                inpaint_pipe = AutoPipelineForInpainting.from_pipe(self.pipeline)
                pipe_kwargs["image"] = init_rgb
                pipe_kwargs["mask_image"] = bin_mask
                pipe_kwargs["strength"] = effective_strength
                pipe_kwargs["width"] = req_w
                pipe_kwargs["height"] = req_h
                output = inpaint_pipe(**pipe_kwargs)

                has_inpaint_compositing = True
                orig_init_image = init_rgb
                orig_raw_mask = bin_mask
            elif controlnet_model and controlnet_image:
                if request.init_image:
                    active_pipe = StableDiffusionXLControlNetImg2ImgPipeline(
                        **self.pipeline.components, controlnet=controlnet_model
                    )
                    pipe_kwargs["image"] = request.init_image
                    pipe_kwargs["control_image"] = controlnet_image
                    pipe_kwargs["strength"] = float(request.denoising_strength)
                    pipe_kwargs["controlnet_conditioning_scale"] = float(request.controlnet_scale)
                else:
                    active_pipe = StableDiffusionXLControlNetPipeline(
                        **self.pipeline.components, controlnet=controlnet_model
                    )
                    pipe_kwargs["image"] = controlnet_image
                    pipe_kwargs["width"] = int(request.width)
                    pipe_kwargs["height"] = int(request.height)
                    pipe_kwargs["controlnet_conditioning_scale"] = float(request.controlnet_scale)
                output = active_pipe(**pipe_kwargs)
            elif request.init_image:
                i2i_pipeline = StableDiffusionXLImg2ImgPipeline(**self.pipeline.components)
                pipe_kwargs["image"] = request.init_image
                pipe_kwargs["strength"] = float(request.denoising_strength)
                output = i2i_pipeline(**pipe_kwargs)
            else:
                pipe_kwargs["width"] = int(request.width)
                pipe_kwargs["height"] = int(request.height)
                output = self.pipeline(**pipe_kwargs)

        except Exception as e:
            del conditioning, pooled, neg_conditioning, neg_pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise GenerationError(f"SDXL generation failed: {e}") from e

        finally:

            if ip_adapter_active and hasattr(self.pipeline, "unload_ip_adapter"):
                try:
                    self.pipeline.unload_ip_adapter()
                except Exception as e:
                    import logging
                    logging.getLogger("SDXLEngine").warning(f"Error unloading IP-Adapter: {e}")
            if controlnet_model is not None:
                del controlnet_model

            self._log_cuda_diagnostics("Post-Gen")

        # 6. Post-Processing (Compositing & Upscaling)
        image = output.images[0]
        del output

        return self._postprocess_and_save(
            image=image,
            request=request,
            start_time=start_time,
            has_inpaint_compositing=has_inpaint_compositing,
            orig_init_image=orig_init_image,
            orig_raw_mask=orig_raw_mask,
            conditioning=conditioning,
            pooled=pooled,
            neg_conditioning=neg_conditioning,
            neg_pooled=neg_pooled
        )

    def _generate_regional(self, request: GenerationRequest) -> GenerationResult:
        """Executes multi-character regional prompting generation pipeline."""
        start_time = time.time()
        device = getattr(self.pipeline, "device", None) or get_torch_device()
        generator = torch.Generator(device=device).manual_seed(int(request.seed))

        try:
            image = generate_regional_sdxl(
                pipeline=self.pipeline,
                compel_processor=self.compel_processor,
                request=request,
                generator=generator,
                adapter_registry=adapter_registry
            )
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise GenerationError(f"Multi-character regional SDXL generation failed: {e}") from e

        return self._postprocess_and_save(
            image=image,
            request=request,
            start_time=start_time
        )

    def _postprocess_and_save(
        self,
        image: Image.Image,
        request: GenerationRequest,
        start_time: float,
        has_inpaint_compositing: bool = False,
        orig_init_image: Optional[Image.Image] = None,
        orig_raw_mask: Optional[Image.Image] = None,
        conditioning: Any = None,
        pooled: Any = None,
        neg_conditioning: Any = None,
        neg_pooled: Any = None
    ) -> GenerationResult:
        end_time = time.time()

        # Pixel-Perfect Seamless Full-Opacity Compositing for Inpainting
        if has_inpaint_compositing and orig_init_image is not None and orig_raw_mask is not None:
            if image.size != orig_init_image.size:
                image = image.resize(orig_init_image.size, Image.Resampling.LANCZOS)

            # Feather mask ensures 100% opacity in the inpainted center while seamlessly feathering the border seam
            blur_radius = max(1, min(int(request.mask_blur if request.mask_blur is not None else 2), 4))
            feather_mask = orig_raw_mask.filter(ImageFilter.GaussianBlur(blur_radius))
            # Image.composite: image1 where mask is 255 (generated inpaint), image2 where mask is 0 (original background)
            image = Image.composite(image, orig_init_image, feather_mask)

        # Upscaling
        if request.upscale_method and str(request.upscale_method).strip().lower() not in ("none", "false", ""):
            scale_factor = float(request.upscale_factor or 2.0)
            method_str = str(request.upscale_method).strip()
            method_lower = method_str.lower()

            if "tiled" in method_lower and "sdxl" in method_lower:
                try:
                    from enhancer.upscaling.adapters.tiled_sdxl import TiledSDXLUpscaler
                    upscaler = TiledSDXLUpscaler(tile_size=1024, overlap=128)
                    image = upscaler.upscale(
                        image=image,
                        scale=scale_factor,
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        strength=0.35,
                        steps=20,
                        seed=request.seed if request.seed is not None else 12345,
                        pipeline=self.pipeline,
                        prompt_embeds=conditioning,
                        pooled_prompt_embeds=pooled,
                        negative_prompt_embeds=neg_conditioning,
                        negative_pooled_prompt_embeds=neg_pooled,
                        enable_tactile_materials=True,
                        crest_strength=0.35,
                        trough_depth=0.40,
                        sss_strength=0.30,
                    )
                except Exception as e:
                    import logging
                    logging.getLogger("SDXLEngine").error(f"Tiled SDXL upscale failed: {e}, falling back to Lanczos")
                    new_width = int(image.width * scale_factor)
                    new_height = int(image.height * scale_factor)
                    image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
            elif any(k in method_lower for k in ["realcugan", "cugan"]):
                try:
                    from enhancer.upscaling.adapters.realcugan import RealCUGANAdapter
                    cugan = RealCUGANAdapter()
                    if cugan.is_available:
                        image = cugan.upscale_pil(image, scale=int(round(scale_factor)))
                    else:
                        raise FileNotFoundError("Real-CUGAN model weights not found")
                except Exception as e:
                    import logging
                    logging.getLogger("SDXLEngine").error(f"Real-CUGAN upscale failed: {e}, falling back to Lanczos")
                    new_width = int(image.width * scale_factor)
                    new_height = int(image.height * scale_factor)
                    image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
            else:
                new_width = int(image.width * scale_factor)
                new_height = int(image.height * scale_factor)
                resample_filter = Image.Resampling.LANCZOS if "lanczos" in method_lower else Image.Resampling.BICUBIC
                image = image.resize((new_width, new_height), resample=resample_filter)

        # Clean up conditioning tensors and free VRAM
        if conditioning is not None:
            del conditioning
        if pooled is not None:
            del pooled
        if neg_conditioning is not None:
            del neg_conditioning
        if neg_pooled is not None:
            del neg_pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Metadata and Saving
        character = request.character_name or ProjectManager.detect_character(request.prompt)
        output_path = ProjectManager.get_output_destination(
            character=character,
            project=request.project_name,
            seed=request.seed
        )
        metadata = PngImagePlugin.PngInfo()
        generation_params = {
            "model": request.model.id,
            "architecture": request.model.architecture,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "sampler": request.sampler,
            "scheduler": request.scheduler,
            "seed": request.seed,
            "width": image.width,
            "height": image.height,
            "loras": request.loras,
            "characters": [
                {
                    "prompt": c.prompt,
                    "lora_name": c.lora_name,
                    "lora_strength": c.lora_strength,
                    "box": list(c.box)
                } for c in getattr(request, "characters", [])
            ],
            "upscale_method": request.upscale_method,
            "upscale_factor": request.upscale_factor,
            "project": request.project_name or "None",
            "character": character
        }

        metadata.add_text("parameters", json.dumps(generation_params, ensure_ascii=False))
        image.save(output_path, pnginfo=metadata)

        vram_peak = 0.0
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()

        # Log usage metrics
        UsageTracker.log_generation(
            checkpoint=request.model.id,
            sampler=request.sampler,
            loras_count=len(request.loras) + len([c for c in getattr(request, "characters", []) if c.lora_name]),
            is_img2img=(request.init_image is not None),
            gen_time_sec=(end_time - start_time),
            vram_peak_gb=vram_peak
        )

        return GenerationResult(
            image_path=str(output_path),
            metadata_path="",
            generation_time_ms=(end_time - start_time) * 1000,
            vram_peak_gb=vram_peak,
        )
