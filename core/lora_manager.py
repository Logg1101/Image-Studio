from pathlib import Path
from typing import Dict, List
import config.paths as paths

class LoraManager:
    """Discovers and manages LoRA weights available in the storage directory."""
    
    def __init__(self, loras_dir: Path = paths.LORAS_DIR):
        self.loras_dir = loras_dir
        self.loras_dir.mkdir(parents=True, exist_ok=True)
        
    SUPPORTED_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}

    def scan_loras(self) -> Dict[str, str]:
        """
        Scans models/loras recursively for all supported weight files.
        Returns a dictionary mapping display names to absolute file paths.
        """
        lora_map: Dict[str, str] = {}
        if not self.loras_dir.exists():
            return lora_map
            
        for file_path in self.loras_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                relative_name = str(file_path.relative_to(self.loras_dir)).replace("\\", "/")
                display_name = relative_name.rsplit(".", 1)[0]
                lora_map[display_name] = str(file_path.resolve())
            
        return lora_map

    def get_lora_names(self) -> List[str]:
        """Returns sorted list of display names for UI dropdowns."""
        return sorted(list(self.scan_loras().keys()))