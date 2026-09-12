import torch
from diffusers import StableDiffusionXLPipeline
from compel import Compel, ReturnedEmbeddingsType
from core.types import ModelInfo
from core.exceptions import ModelLoadError
from core.device import get_torch_device, get_torch_dtype

def load_sdxl_pipeline(model_info: ModelInfo):
    """
    Loads the SDXL pipeline safely, enforcing offline mode, 
    and initializes the Compel processor for unlimited token lengths across any GPU/CPU.
    """
    try:
        device = get_torch_device()
        dtype = get_torch_dtype(device)

        # 1. Load the Base SDXL Pipeline
        pipeline = StableDiffusionXLPipeline.from_single_file(
            model_info.transformer_path,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True
        )
        
        # 2. Bind Compel to both Text Encoders for unlimited lengths and weighting
        compel_processor = Compel(
            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=device
        )
        
        if device.type == "cuda":
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device)
            
        pipeline.vae.enable_slicing()
        
        return pipeline, compel_processor
        
    except Exception as e:
        raise ModelLoadError(f"Failed to load SDXL pipeline: {str(e)}") from e