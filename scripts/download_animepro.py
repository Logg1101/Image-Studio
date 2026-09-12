import sys
from pathlib import Path

# Ensure ImageStudio root is in the Python path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

import config.paths as paths
from huggingface_hub import hf_hub_download

def download_animepro():
    repo_id = "advokat/AnimePro-FLUX"
    filename = "animepro-Q5_K_M.gguf"
    
    print("==========================================")
    print("ImageStudio Model Downloader")
    print("==========================================")
    print(f"Repository: {repo_id}")
    print(f"File:       {filename}")
    print(f"Destination: {paths.FLUX_MODELS_DIR}")
    print("==========================================\n")
    print("Starting download. This will take a while depending on your network speed...")
    
    # Ensure the target directory exists
    paths.FLUX_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the specific file directly into the local directory
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(paths.FLUX_MODELS_DIR)
        )
        print(f"\nSUCCESS! File saved to: {downloaded_path}")
    except Exception as e:
        print(f"\nCRITICAL FAILURE: Could not download the file.")
        print(f"Check your internet connection or verify the filename on Hugging Face.")
        print(f"Error Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_animepro()