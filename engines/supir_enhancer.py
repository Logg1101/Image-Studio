import torch
import gc
from PIL import Image

class SupirEngine:
    def __init__(self):
        self.caption_processor = None
        self.caption_model = None
        self.pipeline = None

    def _flush_vram(self):
        """Aggressively purges the GPU cache to prevent VRAM overflow."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def generate_caption(self, image: Image.Image) -> str:
        """Loads BLIP (under 2GB), analyzes the image, and unloads instantly."""
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        print("Loading BLIP Vision Model into VRAM...")
        model_id = "Salesforce/blip-image-captioning-large"
        
        from core.device import get_torch_device, get_torch_dtype
        dev = get_torch_device()
        dt = get_torch_dtype(dev)

        if self.caption_model is None:
            self.caption_processor = BlipProcessor.from_pretrained(model_id)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dt
            ).to(dev)

        print("Analyzing image...")
        # Force RGB to avoid alpha channel errors
        image = image.convert("RGB")
        inputs = self.caption_processor(image, return_tensors="pt").to(dev, dt)
        
        out = self.caption_model.generate(**inputs, max_new_tokens=50)
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        
        print("Purging BLIP from VRAM...")
        del self.caption_model
        del self.caption_processor
        self.caption_model = None
        self.caption_processor = None
        self._flush_vram()
        
        # We append texture keywords automatically for the upscale
        return f"highly detailed, photorealistic, cinematic lighting, {caption}"

    def enhance_image(self, image: Image.Image, prompt: str, negative_prompt: str, upscale_factor: float, model_name: str) -> Image.Image:
        """Fires up native SDXL ControlNet Tile to inject photorealistic textures."""
        from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
        import os
        
        # Track the loaded model to prevent redundant loading times
        if not hasattr(self, "current_model"):
            self.current_model = None

        if self.pipeline is None or self.current_model != model_name:
            print(f"Loading {model_name} and ControlNet Tile into VRAM...")
            
            import config.paths as paths
            local_cnet = paths.CONTROLNET_DIR / "controlnet-tile-sdxl-1.0"
            if local_cnet.exists():
                controlnet = ControlNetModel.from_pretrained(str(local_cnet), torch_dtype=torch.float16)
            else:
                try:
                    controlnet = ControlNetModel.from_pretrained(
                        "xinsir/controlnet-tile-sdxl-1.0",
                        torch_dtype=torch.float16,
                        local_files_only=True,
                    )
                except Exception:
                    try:
                        controlnet = ControlNetModel.from_pretrained(
                            "xinsir/controlnet-tile-sdxl-1.0",
                            torch_dtype=torch.float16,
                        )
                    except Exception as err:
                        raise FileNotFoundError(
                            f"ControlNet Tile model not found locally or in cache: {err}"
                        ) from err
            local_model_path = str(paths.SDXL_MODELS_DIR / model_name)
            
            self.pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
                local_model_path,
                controlnet=controlnet,
                torch_dtype=torch.float16
            )
            
            self.pipeline.enable_model_cpu_offload() 
            self.pipeline.vae.enable_slicing()
            self.pipeline.vae.enable_tiling()         
            
            try:
                self.pipeline.enable_xformers_memory_efficient_attention() 
            except Exception:
                pass
                
            self.current_model = model_name

        print("Executing Native Upscale Pipeline...")
        # ... (Keep the rest of your image prep and generation execution exactly the same below this line)
        
        # 1. Image Prep
        image = image.convert("RGB")
        target_width = int(image.width * upscale_factor)
        target_height = int(image.height * upscale_factor)
        
        # Standardize dimensions to multiples of 8 for diffusers
        target_width = (target_width // 8) * 8
        target_height = (target_height // 8) * 8
        
        high_res_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # 2. Execute Generation
        with torch.inference_mode():
            restored_image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=high_res_image,          # The base image to denoise
                control_image=high_res_image,  # The structural map for ControlNet
                controlnet_conditioning_scale=1.0, 
                strength=0.35,                 # Keeps 65% of the original image, hallucinates 35% new texture
                num_inference_steps=30,
                guidance_scale=7.0
            ).images[0]
            
        print("Enhancement complete. Returning image.")
        return restored_image