import sys
from pathlib import Path

# Ensure ImageStudio root is in the Python path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

import config.paths as paths
from huggingface_hub import hf_hub_download

def download_test_lora():
    print("Fetching XLabs Anime LoRA for testing...")
    paths.LORAS_DIR.mkdir(parents=True, exist_ok=True)
    
    target_dir = paths.LORAS_DIR / "styles"
    target_dir.mkdir(exist_ok=True)
    
    try:
        downloaded_path = hf_hub_download(
            repo_id="XLabs-AI/flux-lora-collection",
            filename="anime_lora.safetensors",
            local_dir=str(target_dir)
        )
        print(f"\nSUCCESS! Test LoRA saved to: {downloaded_path}")
        print("Go to the ImageStudio UI, click '🔄 Refresh' in the LoRA Settings, and it will appear!")
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    download_test_lora()