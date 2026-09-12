import sys
from pathlib import Path

# Ensure ImageStudio root is in the Python path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from huggingface_hub import snapshot_download

def download_flux_config():
    print("Downloading lightweight configuration files for FLUX.1-schnell...")
    
    # Target directory matching your error log
    target_dir = PROJECT_ROOT / "models" / "checkpoints" / "flux" / "flux-schnell-config"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-schnell",
            local_dir=str(target_dir),
            allow_patterns=["*.json", "*.txt"],  # Only grab configs
            ignore_patterns=["*.safetensors", "*.bin", "*.msgpack", "*.h5", "*.ot"] # Ignore heavy weights
        )
        print(f"\nSUCCESS! Configuration files saved to: {target_dir}")
        print("You can now run ImageStudio entirely offline.")
    except Exception as e:
        print(f"Failed to download configs: {e}")

if __name__ == "__main__":
    download_flux_config()