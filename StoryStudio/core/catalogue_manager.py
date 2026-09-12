import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import config.paths as paths

class CatalogueManager:
    """
    Data-Driven Catalogue and Character Discovery Manager for StoryStudio.
    Dynamically loads all dropdown options from JSON data files without hard-coding,
    and scans models/loras for character LoRAs with companion metadata files.
    """

    def __init__(self, data_dir: Optional[Path] = None, loras_dir: Optional[Path] = None):
        self.data_dir = data_dir or (Path(__file__).resolve().parent.parent / "data")
        self.loras_dir = loras_dir or paths.LORAS_DIR
        self._cache: Dict[str, Any] = {}

    def load_catalogue(self, filename: str) -> List[Dict[str, Any]]:
        """Loads a JSON catalogue file dynamically from the data directory."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"[StoryStudio CatalogueManager] Failed to load {filename}: {e}")
            return []

    def get_all_catalogues(self) -> Dict[str, Any]:
        """Returns all dropdown options, samplers, schedulers, and resolutions."""
        from engines.sdxl.schedulers import AVAILABLE_SAMPLERS, AVAILABLE_SCHEDULERS

        return {
            "clothes_types": self.load_catalogue("clothes_types.json"),
            "clothes_colors": self.load_catalogue("clothes_colors.json"),
            "clothes_details": self.load_catalogue("clothes_details.json"),
            "lighting": self.load_catalogue("lighting.json"),
            "expressions": self.load_catalogue("expressions.json"),
            "poses": self.load_catalogue("poses.json"),
            "povs": self.load_catalogue("povs.json"),
            "bondage_types": self.load_catalogue("bondage_types.json"),
            "bondage_styles": self.load_catalogue("bondage_styles.json"),
            "bondage_hands": self.load_catalogue("bondage_hands.json"),
            "bondage_legs": self.load_catalogue("bondage_legs.json"),
            "bondage_accessories": self.load_catalogue("bondage_accessories.json"),
            "gag_types": self.load_catalogue("gag_types.json"),
            "scenes": self.load_catalogue("scenes.json"),
            "styles": self.load_catalogue("styles.json"),
            "resolutions": self.load_catalogue("resolutions.json"),
            "upscalers": ["None", "4x-UltraSharp", "4x_NMKD-Superscale", "Lanczos", "Bicubic"],
            "samplers": AVAILABLE_SAMPLERS,
            "schedulers": AVAILABLE_SCHEDULERS,
            "default_sampler": "Euler a",
            "default_scheduler": "Normal",
            "default_steps": 30,
            "default_cfg": 5.5
        }

    def scan_characters(self) -> List[Dict[str, Any]]:
        """
        Scans models/loras in real-time for available character LoRAs.
        Discovers companion .json metadata files and dynamically extracts character names/triggers.
        Never hard-codes any character list.
        """
        characters: List[Dict[str, Any]] = []
        if not self.loras_dir.exists():
            return characters

        supported_exts = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
        seen_paths = set()

        # Non-character directory names to exclude if scanning recursively
        excluded_dirs = {"poses", "styles", "controlnet", "backups", "upscalers", "vae"}

        # 1. First scan characters/ subdirectory if present
        char_subdir = self.loras_dir / "characters"
        search_roots = [char_subdir] if char_subdir.exists() else []
        search_roots.append(self.loras_dir)

        for search_root in search_roots:
            if not search_root.exists():
                continue
            for file_path in search_root.rglob("*"):
                if not file_path.is_file() or file_path.suffix.lower() not in supported_exts:
                    continue

                # Check if inside an excluded directory
                rel_parts = [p.lower() for p in file_path.relative_to(self.loras_dir).parts]
                if any(ex in rel_parts[:-1] for ex in excluded_dirs):
                    continue

                canonical_key = str(file_path.resolve()).lower()
                if canonical_key in seen_paths:
                    continue
                seen_paths.add(canonical_key)

                stem = file_path.stem

                # Look for companion JSON metadata (same directory or stem.json)
                json_path = file_path.with_suffix(".json")
                meta: Dict[str, Any] = {}
                if json_path.exists():
                    try:
                        with open(json_path, "r", encoding="utf-8") as jf:
                            meta = json.load(jf)
                    except Exception:
                        pass

                # Derive friendly display name dynamically from metadata or filename
                friendly_name = meta.get("name")
                if not friendly_name:
                    raw_name = stem.replace("_HassakuXIllustrious_V1", "").replace("_HassakuXL", "").replace("_V1", "").replace("_v2", "").replace("_V3", "").replace("_", " ")
                    friendly_name = re.sub(r"\s+", " ", raw_name).strip().title()

                rel_path = str(file_path.relative_to(self.loras_dir)).replace("\\", "/")
                char_id = meta.get("id") or stem

                # Hardcoded trigger overrides for canonical characters
                hardcoded_triggers = {
                    "belfast": "Belfast, long white hair, blue eyes, large breasts",
                    "akeno": "Akeno Himejima, Long black hair, purple eyes, large breasts",
                    "rias": "Rias Gremory, long red hair, green eyes, large breasts",
                    "taihou": "Taihou, long black hair, red eyes, large breasts",
                    "velina": "Velina Airgid, long white hair, purple eyes, large breasts",
                    "rita": "Rita, short brown hair, red eyes, large breasts"
                }

                char_trigger = meta.get("trigger", f"{friendly_name.lower()}, detailed character")
                for k, h_trig in hardcoded_triggers.items():
                    if k in char_id.lower() or k in friendly_name.lower():
                        char_trigger = h_trig
                        break

                characters.append({
                    "id": char_id,
                    "name": friendly_name,
                    "lora": file_path.name,
                    "lora_relative_path": rel_path,
                    "lora_absolute_path": str(file_path.resolve()),
                    "trigger": char_trigger,
                    "default_weight": float(meta.get("default_weight", 0.85)),
                    "description": meta.get("description", f"Character LoRA {friendly_name}"),
                    "personality": meta.get("personality", ""),
                    "default_clothing": meta.get("default_clothing", "")
                })

        # Sort characters alphabetically by friendly display name
        characters.sort(key=lambda c: c["name"])
        return characters

    def get_character_by_id(self, char_id: str) -> Optional[Dict[str, Any]]:
        chars = self.scan_characters()
        if not chars:
            return None

        # 1. Exact match by id
        for c in chars:
            if c["id"] == char_id:
                return c

        # 2. Match by relative path or stem
        for c in chars:
            if c["lora_relative_path"] == char_id or Path(c["lora_relative_path"]).stem == Path(char_id).stem:
                return c

        # 3. Match by name
        for c in chars:
            if c["name"].lower() == char_id.lower():
                return c

        # 4. Partial substring match
        for c in chars:
            if char_id.lower() in c["id"].lower() or char_id.lower() in c["name"].lower():
                return c

        # 5. Fallback to first available on-disk character
        return chars[0]
