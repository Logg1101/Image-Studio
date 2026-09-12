from pathlib import Path
from typing import Dict
from core.types import ModelInfo
from core.exceptions import ModelLoadError
import config.paths as paths

class ModelManager:
    def __init__(self):
        self.available_models: Dict[str, ModelInfo] = {}
        self.scan_models()

    def scan_models(self):
        """Scans the checkpoint directories dynamically and registers only models currently present on disk."""
        self.available_models.clear()
        if not paths.CHECKPOINTS_DIR.exists():
            return

        for file_path in sorted(paths.CHECKPOINTS_DIR.rglob("*")):
            if not file_path.is_file():
                continue
            parts = [p.lower() for p in file_path.parts]
            if any(p.startswith(".") or p.endswith("-config") for p in parts):
                continue
            ext = file_path.suffix.lower()
            if ext not in [".safetensors", ".gguf"]:
                continue

            model_id = file_path.stem
            is_flux = "flux" in parts or "flux" in model_id.lower() or ext == ".gguf"
            arch = "flux" if is_flux else "sdxl"
            fmt = "gguf" if ext == ".gguf" else "safetensors"
            cfg = str(paths.FLUX_MODELS_DIR / "flux-schnell-config") if arch == "flux" else None

            self.available_models[model_id] = ModelInfo(
                id=model_id,
                architecture=arch,
                variant="schnell" if arch == "flux" else "base",
                format=fmt,
                transformer_path=str(file_path.resolve()),
                config_path=cfg
            )

    def get_model_info(self, model_id: str) -> ModelInfo:
        if model_id not in self.available_models:
            raise ModelLoadError(f"Model '{model_id}' not found. Please ensure it is in the correct directory.")
        return self.available_models[model_id]