from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Any, List, Tuple
import random
from PIL import Image

MAX_SEED = 2147483647  # Maximum 32-bit signed integer (Qt QSpinBox limit)

def sanitize_seed(val: Any) -> int:
    """
    Deterministically normalizes any seed (64-bit integer, string, or negative value)
    into a valid Qt-safe signed 32-bit integer range [0, 2147483647].
    Prevents OverflowError across UI spinboxes, parameter serialization, and reuse.
    """
    if val is None:
        return random.randint(0, MAX_SEED)
    try:
        s = int(val)
        if s < 0:
            s = abs(s)
        return s % (MAX_SEED + 1)
    except Exception:
        return random.randint(0, MAX_SEED)

@dataclass
class CharacterRegion:
    prompt: str
    lora_name: Optional[str] = None
    lora_strength: float = 1.0
    # Normalized bounding box [x_min, y_min, x_max, y_max] in [0.0, 1.0]
    box: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    feather: float = 0.04

@dataclass
class ModelInfo:
    id: str
    architecture: str
    variant: str
    format: str
    transformer_path: str
    text_encoder_paths: Dict[str, str] = field(default_factory=dict)
    vae_path: Optional[str] = None
    config_path: Optional[str] = None

@dataclass
class GenerationRequest:
    model: ModelInfo
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance_scale: float = 7.0
    seed: int = 12345
    sampler: str = "Euler a"
    scheduler: str = "Normal"
    negative_prompt: Optional[str] = None
    loras: Dict[str, float] = field(default_factory=dict)
    characters: List[CharacterRegion] = field(default_factory=list)
    
    # Post-processing parameters
    upscale_method: str = "None"
    upscale_factor: float = 2.0
    
    # Progress tracking hook
    step_callback: Optional[Callable[[int, int], None]] = None
    
    # Image-to-Image & Inpainting parameters
    init_image: Optional[Image.Image] = None
    mask_image: Optional[Image.Image] = None
    mask_blur: int = 4
    denoising_strength: float = 0.65
    
    # ControlNet parameters
    controlnet_type: str = "None"
    controlnet_image: Optional[Image.Image] = None
    controlnet_scale: float = 0.8
    
    # IP-Adapter parameters
    ip_adapter_name: str = "None"
    ip_adapter_image: Optional[Image.Image] = None
    ip_adapter_scale: float = 0.6
    
    # PuLID / FaceID parameters
    pulid_enabled: bool = False
    pulid_image: Optional[Image.Image] = None
    pulid_strength: float = 0.8
    pulid_scale: float = 0.8

    # Project & Character Organization
    project_name: Optional[str] = None
    character_name: Optional[str] = None

    def __post_init__(self):
        # 1. Normalize seed to signed 32-bit range
        self.seed = sanitize_seed(self.seed)
        
        # 2. Sync PuLID aliases
        if self.pulid_strength != 0.8 and self.pulid_scale == 0.8:
            self.pulid_scale = self.pulid_strength
        elif self.pulid_scale != 0.8 and self.pulid_strength == 0.8:
            self.pulid_strength = self.pulid_scale

        # 3. Ensure dimensions are multiples of 8
        self.width = max(64, (int(self.width) // 8) * 8)
        self.height = max(64, (int(self.height) // 8) * 8)
        self.steps = max(1, int(self.steps))
        self.guidance_scale = float(self.guidance_scale)
        self.denoising_strength = max(0.0, min(1.0, float(self.denoising_strength)))

@dataclass
class GenerationResult:
    image_path: str
    metadata_path: str
    generation_time_ms: float
    vram_peak_gb: float
