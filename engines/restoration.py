import time
import sys
from PIL import Image
from rembg import remove, new_session
from simple_lama_inpainting import SimpleLama

class RestorationEngine:
    def __init__(self):
        # Initialized lazily to conserve VRAM until clicked
        self.rembg_session = None
        self.lama_model = None

    def remove_background(self, input_image: Image.Image) -> Image.Image:
        """Strips the background using U-2-Net."""
        t0 = time.time()
        if self.rembg_session is None:
            print("[ImageStudio Rembg] Initializing U-2-Net session on GPU...", flush=True)
            self.rembg_session = new_session("u2net")
        
        print(f"[ImageStudio Rembg] Processing background cutout for {input_image.size[0]}x{input_image.size[1]} image...", flush=True)
        result = remove(input_image, session=self.rembg_session)
        elapsed = time.time() - t0
        print(f"[ImageStudio Rembg] Background removal completed in {elapsed:.2f}s! Ready.", flush=True)
        return result

    def remove_object(self, input_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        """Erases objects/watermarks and reconstructs the background using LaMa."""
        t0 = time.time()
        if self.lama_model is None:
            print("[ImageStudio LaMa] Initializing LaMa model (big-lama)...", flush=True)
            self.lama_model = SimpleLama() 
        
        orig_size = input_image.size
        # The mask must be a single-channel grayscale image
        if mask_image.mode != 'L':
            mask_image = mask_image.convert('L')
            
        # Ensure mask dimensions strictly match input image
        if mask_image.size != orig_size:
            mask_image = mask_image.resize(orig_size, Image.Resampling.NEAREST)
            
        print(f"[ImageStudio LaMa] Running structural inpainting on {orig_size[0]}x{orig_size[1]} image...", flush=True)
        result = self.lama_model(input_image, mask_image)
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        elapsed = time.time() - t0
        print(f"[ImageStudio LaMa] Inpainting completed in {elapsed:.2f}s! Ready.", flush=True)
        return result

    def fix_artifacts(self, input_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        """Fixes glitches, anatomy artifacts, or deformities within the masked area using LaMa."""
        return self.remove_object(input_image, mask_image)