import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict
import config.paths as paths

class ProjectManager:
    """
    Manages intelligent file organization, project folders, character detection,
    and guaranteed unique timestamped filenames for generated and edited images.
    """
    PROJECTS_DIR = paths.OUTPUTS_DIR / "Projects"

    @classmethod
    def ensure_directories(cls):
        cls.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        paths.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def list_projects(cls) -> List[str]:
        cls.ensure_directories()
        projects = ["No Project", "Personal"]
        if cls.PROJECTS_DIR.exists():
            for p in sorted(cls.PROJECTS_DIR.iterdir()):
                if p.is_dir() and p.name not in projects:
                    projects.append(p.name)
        return projects

    @classmethod
    def create_project(cls, name: str) -> Path:
        cls.ensure_directories()
        clean_name = re.sub(r'[\\/*?:"<>|]', "", name.strip())
        if not clean_name:
            clean_name = "Untitled_Project"
        proj_dir = cls.PROJECTS_DIR / clean_name
        proj_dir.mkdir(parents=True, exist_ok=True)
        return proj_dir

    @classmethod
    def detect_character(cls, prompt: str) -> str:
        """
        Attempts to detect character name from positive prompt.
        Defaults to 'General' if no obvious character name is matched.
        """
        if not prompt:
            return "General"

        # Known common character patterns or title-cased character tokens
        prompt_lower = prompt.lower()
        known_characters = [
            "belfast", "akeno", "enterprise", "taihou", "raiden", "ganyu",
            "hutao", "frieren", "fern", "march 7th", "kafka", "firefly",
            "asuka", "rei", "miku", "2b", "tifa", "aerith", "yor"
        ]

        for char in known_characters:
            if re.search(rf"\b{re.escape(char)}\b", prompt_lower):
                return char.capitalize()

        # Fallback: check first comma-separated tag if short (< 25 chars and not quality prompt)
        tags = [t.strip() for t in prompt.split(",") if t.strip()]
        for tag in tags[:2]:
            tag_clean = re.sub(r'[^a-zA-Z0-9_\- ]', '', tag).strip()
            if 2 <= len(tag_clean) <= 20 and not any(w in tag_clean.lower() for w in ["masterpiece", "quality", "1girl", "1boy", "solo", "detailed", "art"]):
                return tag_clean.title().replace(" ", "_")

        return "General"

    @classmethod
    def get_output_destination(
        cls,
        character: Optional[str] = None,
        project: Optional[str] = None,
        seed: int = 0,
        suffix: str = "png"
    ) -> Path:
        """
        Calculates the destination folder and generates a guaranteed unique filename.
        Hierarchy:
        - If Project is specified and != 'No Project' and != 'Personal':
            outputs/Projects/[ProjectName]/YYYY-MM-DD/
        - Else:
            outputs/YYYY-MM-DD/[Character]/
        Filename:
            [Character]_[YYYYMMDD]_[HHMMSS]_[seed].png (with _01, _02 if needed)
        """
        cls.ensure_directories()
        now = time.localtime()
        date_str = time.strftime("%Y-%m-%d", now)
        date_compact = time.strftime("%Y%m%d", now)
        time_compact = time.strftime("%H%M%S", now)

        char_name = re.sub(r'[\\/*?:"<>|]', "", (character or "General").strip().replace(" ", "_"))
        if not char_name:
            char_name = "General"

        # Determine target directory
        if project and project not in ["No Project", "Personal", "None"]:
            clean_proj = re.sub(r'[\\/*?:"<>|]', "", project.strip())
            target_dir = cls.PROJECTS_DIR / clean_proj / date_str
        else:
            target_dir = paths.OUTPUTS_DIR / date_str / char_name

        target_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique base name
        base_filename = f"{char_name}_{date_compact}_{time_compact}_{seed}"
        candidate_path = target_dir / f"{base_filename}.{suffix}"

        counter = 1
        while candidate_path.exists():
            candidate_path = target_dir / f"{base_filename}_{counter:02d}.{suffix}"
            counter += 1

        return candidate_path
