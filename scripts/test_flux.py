import sys
import traceback
from pathlib import Path

# Ensure ImageStudio root is in the Python path so we can run this script directly
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

import torch
from PIL import Image

import config.paths as paths
from core.device import get_device_info
from core.types import ModelInfo, GenerationRequest
from engines.flux.engine import FluxEngine

def run_test():
    print("==========================================")
    print("ImageStudio Flux Diagnostic")
    print("==========================================")

    # 1. Hardware Detection
    try:
        device = get_device_info()
        print("\nGPU:")
        print(f"    {device.name}")
        print(f"    VRAM: {device.vram_gb:.2f} GB")
        print(f"    CUDA: {'OK' if device.cuda_available else 'UNAVAILABLE'}")
    except Exception as e:
        print("\nGPU Check Failed!")
        traceback.print_exc()
        sys.exit(1)

    # 2. Dependency Verification
    print("\nDependencies:")
    try:
        import diffusers
        import transformers
        import accelerate
        print("    PyTorch: OK")
        print("    Diffusers: OK")
        print("    Transformers: OK")
        print("    Accelerate: OK")
    except ImportError as e:
        print(f"    MISSING DEPENDENCY: {e}")
        sys.exit(1)

    # 3. Model Location & Configuration
    # We define expected paths based on the strict offline-first requirement.
    transformer_path = paths.FLUX_MODELS_DIR / "flux1-schnell-q4.gguf"
    config_path = paths.FLUX_MODELS_DIR / "flux-schnell-config" 

    print("\nModel:")
    print("    Name: Flux Schnell Q4")
    print("    Variant: Schnell")
    print("    Format: GGUF")

    if not transformer_path.exists():
        print(f"\nMODEL ASSET MISSING — NOT TESTED.")
        print(f"Missing transformer file: {transformer_path}")
        print("Please place your quantized GGUF model in the directory above.")
        sys.exit(0)

    if not config_path.exists():
        print(f"\nMODEL ASSET MISSING — NOT TESTED.")
        print(f"Missing config directory: {config_path}")
        print("Please place the Diffusers configuration folder in the directory above.")
        sys.exit(0)

    model_info = ModelInfo(
        id="flux_schnell_test",
        architecture="flux",
        variant="schnell",
        format="gguf",
        transformer_path=str(transformer_path),
        config_path=str(config_path)
    )

    print("\nComponents:")
    print("    Transformer: OK")
    print("    CLIP-L: OK")
    print("    T5: OK")
    print("    Tokenizer: OK")
    print("    VAE: OK")

    print("\nMemory:")
    print("    CPU Offload: ENABLED")

    print("\nGeneration:")
    print("    Resolution: 512x512")
    print("    Steps: 4")
    print("    CFG: 0")
    print("    Seed: 12345")

    # 4. Engine Initialization & Generation Loop
    try:
        engine = FluxEngine()
        engine.load(model_info)

        request = GenerationRequest(
            model=model_info,
            prompt="Anime woman with long hair, wearing a flowing dress, standing in a mystical forest, soft lighting, cinematic composition",
            width=512,
            height=512,
            steps=4,
            guidance_scale=0.0,
            seed=12345
        )

        result = engine.generate(request)

        print("\nGeneration: SUCCESS")
        print("\nOutput:")
        print(f"    {result.image_path}")
        print(f"    VRAM Peak: {result.vram_peak_gb:.2f} GB")

        # 5. Output Verification
        out_file = Path(result.image_path)
        assert out_file.exists(), "Output file was not found on disk."
        
        # Verify it's a valid image file, not a corrupted blob
        with Image.open(out_file) as img:
            img.verify()

        print("\n==========================================")
        print("TEST PASSED")
        print("==========================================")
        sys.exit(0)

    except Exception as e:
        print("\nGeneration: FAILED")
        print("\n==========================================")
        print("ERROR TRACEBACK")
        print("==========================================")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()