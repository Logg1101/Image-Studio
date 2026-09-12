import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

class ProjectManager:
    """
    Manages Story Studio project storage, persistence, and assets.
    Each story project is contained within StoryStudio/projects/<ProjectName>/
    with project.json, story.txt, images/, and prompts/.
    """

    def __init__(self, projects_root: Optional[Path] = None):
        self.projects_root = projects_root or (Path(__file__).resolve().parent.parent / "projects")
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> List[Dict[str, Any]]:
        """Returns a list of all existing story project summaries."""
        projects = []
        for p in self.projects_root.iterdir():
            if p.is_dir():
                pj_file = p / "project.json"
                if pj_file.exists():
                    try:
                        with open(pj_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            projects.append({
                                "name": p.name,
                                "character": data.get("character", ""),
                                "line_count": len(data.get("lines", [])),
                                "completed_count": sum(1 for line in data.get("lines", []) if line.get("status") == "Completed"),
                                "updated_at": data.get("updated_at", "")
                            })
                    except Exception:
                        pass
        return projects

    def get_project_dir(self, name: str) -> Path:
        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_", " ")).strip() or "UntitledStory"
        return self.projects_root / safe_name

    def get_project(self, name: str) -> Optional[Dict[str, Any]]:
        p_dir = self.get_project_dir(name)
        pj_file = p_dir / "project.json"
        if not pj_file.exists():
            return None
        try:
            with open(pj_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[StoryStudio ProjectManager] Error loading project {name}: {e}")
            return None

    def create_or_update_project(
        self,
        name: str,
        story_lines: Optional[List[str]] = None,
        character_id: str = "",
        default_settings: Optional[Dict[str, Any]] = None,
        scenes_data: Optional[List[Dict[str, Any]]] = None,
        raw_story_text: Optional[str] = None
    ) -> Dict[str, Any]:
        p_dir = self.get_project_dir(name)
        images_dir = p_dir / "images"
        prompts_dir = p_dir / "prompts"
        images_dir.mkdir(parents=True, exist_ok=True)
        prompts_dir.mkdir(parents=True, exist_ok=True)

        existing = self.get_project(name) or {}
        existing_lines_map = {}
        if existing and "lines" in existing:
            for l in existing["lines"]:
                existing_lines_map[l.get("index")] = l

        lines_struct = []

        if scenes_data is not None:
            for i, scene in enumerate(scenes_data):
                prev = existing_lines_map.get(i)
                lines_struct.append({
                    "index": i,
                    "text": scene.get("text", ""),
                    "positive_prompt": scene.get("positive_prompt", ""),
                    "negative_prompt": scene.get("negative_prompt", ""),
                    "directive": scene.get("directive", {}),
                    "status": scene.get("status") or (prev.get("status", "Pending") if prev else "Pending"),
                    "image_filename": scene.get("image_filename") or (prev.get("image_filename") if prev else None),
                    "prompt_data": scene.get("prompt_data") or (prev.get("prompt_data") if prev else None),
                    "seed": scene.get("seed") if scene.get("seed") is not None else (prev.get("seed") if prev else None),
                    "error": None
                })
        else:
            story_lines = story_lines or []
            for i, text_line in enumerate(story_lines):
                prev = existing_lines_map.get(i)
                if prev and prev.get("text") == text_line:
                    lines_struct.append(prev)
                else:
                    lines_struct.append({
                        "index": i,
                        "text": text_line,
                        "positive_prompt": prev.get("positive_prompt", "") if prev else "",
                        "negative_prompt": prev.get("negative_prompt", "") if prev else "",
                        "directive": prev.get("directive", {}) if prev else {},
                        "status": prev.get("status", "Pending") if prev else "Pending",
                        "image_filename": prev.get("image_filename") if prev else None,
                        "prompt_data": prev.get("prompt_data") if prev else None,
                        "seed": prev.get("seed") if prev else None,
                        "error": None
                    })

        import datetime
        project_data = {
            "name": p_dir.name,
            "character": character_id or existing.get("character", ""),
            "raw_story_text": raw_story_text or existing.get("raw_story_text", ""),
            "default_settings": default_settings or existing.get("default_settings", {}),
            "lines": lines_struct,
            "continuity": existing.get("continuity", {}),
            "updated_at": datetime.datetime.now().isoformat()
        }

        # Save project.json
        with open(p_dir / "project.json", "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=2)

        # Save story.txt
        with open(p_dir / "story.txt", "w", encoding="utf-8") as f:
            if raw_story_text:
                f.write(raw_story_text + "\n\n---\n")
            for i, line_item in enumerate(lines_struct):
                f.write(f"{i + 1}. {line_item.get('text', '')}\n")

        return project_data

    def update_project_scenes(self, project_name: str, scenes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        project = self.get_project(project_name)
        if not project:
            return None
        return self.create_or_update_project(
            name=project_name,
            character_id=project.get("character", ""),
            default_settings=project.get("default_settings", {}),
            scenes_data=scenes,
            raw_story_text=project.get("raw_story_text")
        )

    def save_line_result(
        self,
        project_name: str,
        line_index: int,
        image: Image.Image,
        prompt_meta: Dict[str, Any],
        continuity_snapshot: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Saves generated line image to images/00X.png and metadata to prompts/00X.json,
        and updates project.json status to 'Completed'.
        """
        p_dir = self.get_project_dir(project_name)
        images_dir = p_dir / "images"
        prompts_dir = p_dir / "prompts"
        images_dir.mkdir(parents=True, exist_ok=True)
        prompts_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{line_index + 1:03d}"
        img_filename = f"{prefix}.png"
        prompt_filename = f"{prefix}.json"

        img_path = images_dir / img_filename
        image.save(img_path, format="PNG")

        prompt_path = prompts_dir / prompt_filename
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(prompt_meta, f, indent=2)

        # Update project.json
        project = self.get_project(project_name)
        if project and "lines" in project:
            if line_index < len(project["lines"]):
                project["lines"][line_index]["status"] = "Completed"
                project["lines"][line_index]["image_filename"] = img_filename
                project["lines"][line_index]["prompt_data"] = prompt_meta
                project["lines"][line_index]["seed"] = prompt_meta.get("seed")
                project["lines"][line_index]["error"] = None

            if continuity_snapshot:
                project["continuity"] = continuity_snapshot

            with open(p_dir / "project.json", "w", encoding="utf-8") as f:
                json.dump(project, f, indent=2)

        return str(img_path.resolve())

    def update_line_status(self, project_name: str, line_index: int, status: str, error: Optional[str] = None):
        p_dir = self.get_project_dir(project_name)
        project = self.get_project(project_name)
        if project and "lines" in project and line_index < len(project["lines"]):
            project["lines"][line_index]["status"] = status
            if error:
                project["lines"][line_index]["error"] = error
            with open(p_dir / "project.json", "w", encoding="utf-8") as f:
                json.dump(project, f, indent=2)
