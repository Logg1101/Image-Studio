import json
from pathlib import Path
import config.paths as paths
from core.types import GenerationRequest, GenerationResult
from history.database import HistoryDB

class HistoryManager:
    def __init__(self):
        self.db = HistoryDB()
        
    def save_generation(self, request: GenerationRequest, result: GenerationResult) -> str:
        """Saves metadata to JSON and logs the record in the database."""
        # 1. Prepare Metadata
        metadata = {
            "model": request.model.id,
            "architecture": request.model.architecture,
            "variant": request.model.variant,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "seed": request.seed,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "width": request.width,
            "height": request.height,
            "generation_time_ms": result.generation_time_ms,
            "vram_peak_gb": result.vram_peak_gb
        }
        
        # 2. Insert into Database first to ensure Vault records are NEVER lost
        try:
            db_record = metadata.copy()
            db_record["model_id"] = db_record.pop("model", request.model.id)
            db_record["image_path"] = result.image_path
            self.db.insert_record(db_record)
        except Exception as e:
            print(f"Warning: Failed to save to history database: {e}")

        # 3. Save JSON metadata file next to the image
        meta_path = ""
        try:
            image_name = Path(result.image_path).stem
            paths.METADATA_DIR.mkdir(parents=True, exist_ok=True)
            meta_path = paths.METADATA_DIR / f"{image_name}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save metadata JSON file: {e}")
            
        return str(meta_path)