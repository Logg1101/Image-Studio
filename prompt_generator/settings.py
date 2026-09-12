import json
from pathlib import Path
from typing import Dict, Any
import config.paths as paths
from prompt_generator.types import LLMSettings

SETTINGS_FILE = paths.PROJECT_ROOT / "config" / "llm_settings.json"

class SettingsManager:
    def __init__(self, file_path: Path = SETTINGS_FILE):
        self.file_path = file_path
        self._settings: LLMSettings = self.load_settings()

    def load_settings(self) -> LLMSettings:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return LLMSettings.from_dict(data)
            except Exception as e:
                print(f"[PromptGenerator] Failed to load settings from {self.file_path}: {e}")

        return LLMSettings()

    def get_settings(self) -> LLMSettings:
        return self._settings

    def save_settings(self, settings_dict: Dict[str, Any]) -> LLMSettings:
        merged = self._settings.to_dict()
        merged.update(settings_dict)
        self._settings = LLMSettings.from_dict(merged)

        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._settings.to_dict(), f, indent=2)
        except Exception as e:
            print(f"[PromptGenerator] Failed to save settings to {self.file_path}: {e}")

        return self._settings


settings_manager = SettingsManager()
