from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
METADATA_DIR = OUTPUTS_DIR / "metadata"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FLUX_MODELS_DIR = CHECKPOINTS_DIR / "flux"
SDXL_MODELS_DIR = CHECKPOINTS_DIR / "sdxl"
LORAS_DIR = MODELS_DIR / "loras"
CONTROLNET_DIR = MODELS_DIR / "controlnet"
IPADAPTER_DIR = MODELS_DIR / "ipadapter"
IP_ADAPTER_DIR = IPADAPTER_DIR
PULID_DIR = MODELS_DIR / "pulid"
CLIP_VISION_DIR = MODELS_DIR / "clip_vision"
PREPROCESSORS_DIR = MODELS_DIR / "preprocessors"
UPSCALERS_DIR = MODELS_DIR / "upscalers"

def ensure_directories():
    """Ensure all required project directories exist."""
    directories = [
        DATA_DIR,
        OUTPUTS_DIR,
        IMAGES_DIR,
        METADATA_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        FLUX_MODELS_DIR,
        SDXL_MODELS_DIR,
        LORAS_DIR,
        CONTROLNET_DIR,
        IPADAPTER_DIR,
        PULID_DIR,
        CLIP_VISION_DIR,
        PREPROCESSORS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)