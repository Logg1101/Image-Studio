import json
import logging
import time
import torch
import transformers
from pathlib import Path

logger = logging.getLogger("FluxEngine")

# Suppress noisy HuggingFace CLIPTokenizer truncation warnings for Flux
transformers.logging.set_verbosity_error()
from engines.base import GenerationEngine
from core.types import GenerationRequest, GenerationResult, ModelInfo
from core.exceptions import GenerationError, UnsupportedModelError
from engines.flux.loader import load_flux_pipeline
from engines.flux.config import get_config_for_variant
from engines.flux.memory import optimize_memory_for_12gb, clear_vram
import config.paths as paths
from PIL import PngImagePlugin, Image
from core.project_manager import ProjectManager
from core.usage_tracker import UsageTracker

class FluxEngine(GenerationEngine):
    def __init__(self):
        self.pipeline = None
        self.current_model_info = None
        
    def supports(self, model_info: ModelInfo) -> bool:
        return model_info.architecture.lower() == "flux"
        
    def is_loaded(self) -> bool:
        return self.pipeline is not None
        
    def load(self, model_info: ModelInfo) -> None:
        if not self.supports(model_info):
            raise UnsupportedModelError(f"FluxEngine cannot load architecture: {model_info.architecture}")
            
        self.unload()
        
        self.pipeline = load_flux_pipeline(model_info)
        optimize_memory_for_12gb(self.pipeline)
        self.current_model_info = model_info
        
    def unload(self) -> None:
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.current_model_info = None
            clear_vram()
            
    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self.is_loaded():
            raise GenerationError("Cannot generate: Model is not loaded.")
            
        variant_config = get_config_for_variant(self.current_model_info.variant)
        
        steps = request.steps if request.steps > 0 else variant_config.steps
        guidance = request.guidance_scale if request.guidance_scale >= 0 else variant_config.guidance_scale
        
        # --- LoRA Multi-Adapter Loadout ---
        loaded_adapter_names = []
        loaded_adapter_weights = []
        
        if request.loras:
            for lora_path, weight in request.loras.items():
                adapter_name = Path(lora_path).stem
                print(f"Loading LoRA into Flux: {Path(lora_path).name} at weight {weight}")
                try:
                    self.pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
                    loaded_adapter_names.append(adapter_name)
                    loaded_adapter_weights.append(float(weight))
                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                    logger.warning(f"Failed to load LoRA {lora_path}: {e}")
            
            if loaded_adapter_names:
                try:
                    self.pipeline.set_adapters(loaded_adapter_names, adapter_weights=loaded_adapter_weights)
                    logger.info(f"Flux active LoRAs set: {loaded_adapter_names}")
                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                    logger.warning(f"Failed to set Flux adapters: {e}")
        
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        start_time = time.time()
        
        def flux_step_callback(pipe, step_index, timestep, callback_kwargs):
            if request.step_callback:
                request.step_callback(step_index + 1, steps)
            return callback_kwargs

        try:
            output = self.pipeline(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                max_sequence_length=variant_config.max_sequence_length,
                generator=generator,
                output_type="pil",
                callback_on_step_end=flux_step_callback
            )
        except Exception as e:
            raise GenerationError(f"Flux generation failed: {str(e)}") from e
        finally:
            # Unload Adapters to prevent bleed-through
            if loaded_adapter_names:
                try:
                    self.pipeline.unload_lora_weights()
                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                    logger.warning(f"[FluxEngine] Failed to unload LoRA weights {loaded_adapter_names}: {e}")
                    if hasattr(self.pipeline, "disable_lora"):
                        try:
                            self.pipeline.disable_lora()
                        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as dis_err:
                            logger.warning(f"[FluxEngine] Fallback disable_lora failed: {dis_err}")
                    # Invalidate pipeline to guarantee clean state and prevent bleed-through
                    self.pipeline = None
                    self.current_model_info = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
        end_time = time.time()
        image = output.images[0]
        
        # --- Post-Processing (Upscaling) ---
        if request.upscale_method and str(request.upscale_method).strip().lower() not in ("none", "false", ""):
            scale_factor = float(request.upscale_factor or 2.0)
            method_str = str(request.upscale_method).strip()
            method_lower = method_str.lower()

            if any(k in method_lower for k in ["realcugan", "cugan"]):
                try:
                    from enhancer.upscaling.adapters.realcugan import RealCUGANAdapter
                    cugan = RealCUGANAdapter()
                    if cugan.is_available:
                        image = cugan.upscale_pil(image, scale=int(round(scale_factor)))
                    else:
                        raise FileNotFoundError("Real-CUGAN weights not found")
                except Exception as e:
                    print(f"Flux Real-CUGAN upscale failed: {e}, falling back to Lanczos")
                    new_width = int(image.width * scale_factor)
                    new_height = int(image.height * scale_factor)
                    image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
            else:
                new_width = int(image.width * scale_factor)
                new_height = int(image.height * scale_factor)
                if "lanczos" in method_lower:
                    resample_filter = Image.Resampling.LANCZOS
                elif "bicubic" in method_lower:
                    resample_filter = Image.Resampling.BICUBIC
                else:
                    resample_filter = Image.Resampling.NEAREST
                print(f"Flux Upscaling {scale_factor}x using {request.upscale_method} to {new_width}x{new_height}")
                image = image.resize((new_width, new_height), resample=resample_filter)
        
        # --- Unique Output Destination ---
        character = request.character_name or ProjectManager.detect_character(request.prompt)
        output_path = ProjectManager.get_output_destination(
            character=character,
            project=request.project_name,
            seed=request.seed
        )
        
        # --- Bake Metadata into PNG ---
        metadata = PngImagePlugin.PngInfo()
        generation_params = {
            "model": request.model.id,
            "architecture": request.model.architecture,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "width": image.width,
            "height": image.height,
            "loras": request.loras,
            "upscale_method": request.upscale_method,
            "upscale_factor": request.upscale_factor,
            "project": request.project_name or "None",
            "character": character
        }
        metadata.add_text("parameters", json.dumps(generation_params))
        image.save(output_path, pnginfo=metadata)

        # --- Track Peak VRAM ---
        vram_peak = 0.0
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()

        # Log usage metrics
        UsageTracker.log_generation(
            checkpoint=request.model.id,
            sampler="Flux Default",
            loras_count=len(request.loras),
            is_img2img=False,
            gen_time_sec=(end_time - start_time),
            vram_peak_gb=vram_peak
        )

        return GenerationResult(
            image_path=str(output_path),
            metadata_path="",
            generation_time_ms=(end_time - start_time) * 1000,
            vram_peak_gb=vram_peak,
        )