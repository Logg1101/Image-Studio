import os
import torch
from pathlib import Path
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from core.types import ModelInfo
from core.exceptions import ModelLoadError
import config.paths as paths

# Cache dictionary for shared Flux components (CLIP, T5, Tokenizers, VAE)
_FLUX_COMPONENT_CACHE = {}

def clear_flux_cache():
    """Explicitly clears the cached Flux sub-components."""
    global _FLUX_COMPONENT_CACHE
    _FLUX_COMPONENT_CACHE.clear()

def load_flux_pipeline(model_info: ModelInfo) -> FluxPipeline:
    """
    Loads the Flux pipeline efficiently with shared component caching,
    FP8 T5 text encoder support, and CPU offloading for 12GB VRAM performance.
    """
    try:
        config_dir = Path(model_info.config_path) if model_info.config_path else paths.FLUX_MODELS_DIR / "flux-schnell-config"
        
        # 1. Load Transformer
        print(f"Loading Flux Transformer ({model_info.format}): {Path(model_info.transformer_path).name}")
        if model_info.format.lower() == "gguf":
            quant_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
            transformer = FluxTransformer2DModel.from_single_file(
                model_info.transformer_path,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                config=str(config_dir),
                subfolder="transformer",
                local_files_only=True
            )
        elif model_info.format.lower() == "safetensors":
            transformer = FluxTransformer2DModel.from_single_file(
                model_info.transformer_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
        else:
            raise ValueError(f"Unsupported model format: {model_info.format}")
        
        # 2. Build or Reuse Cached Pipeline
        if "base_pipeline" in _FLUX_COMPONENT_CACHE and _FLUX_COMPONENT_CACHE["base_config"] == str(config_dir):
            print("Reusing cached Flux text encoders and VAE components (Fast Swap)")
            base_pipe = _FLUX_COMPONENT_CACHE["base_pipeline"]
            pipeline = FluxPipeline(
                scheduler=base_pipe.scheduler,
                text_encoder=base_pipe.text_encoder,
                text_encoder_2=base_pipe.text_encoder_2,
                tokenizer=base_pipe.tokenizer,
                tokenizer_2=base_pipe.tokenizer_2,
                vae=base_pipe.vae,
                transformer=transformer
            )
        else:
            print(f"Loading base Flux pipeline components from {config_dir}")
            pipeline = FluxPipeline.from_pretrained(
                str(config_dir),
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
            # Cache common components
            _FLUX_COMPONENT_CACHE["base_pipeline"] = pipeline
            _FLUX_COMPONENT_CACHE["base_config"] = str(config_dir)

        # 3. 12GB VRAM Optimizations
        pipeline.enable_model_cpu_offload()
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        elif hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()

        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        elif hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        
        return pipeline
        
    except Exception as e:
        raise ModelLoadError(f"Failed to load Flux pipeline: {str(e)}") from e