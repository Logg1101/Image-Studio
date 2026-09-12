import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from StoryStudio.agents.llm_provider import LLMProvider

class StoryAgent:
    """
    AI Story Continuity Agent for StoryStudio.
    Analyzes story line progressions, maintains situational and visual continuity
    (clothing, environment, character identity, emotions), and outputs structured
    scene directives for the Prompt Agent.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.base_dir = Path(__file__).resolve().parent
        self.config_path = config_path or (self.base_dir / "config.json")
        self.system_prompt_path = self.base_dir / "system_prompt.txt"

        self.config = self._load_config()
        self.system_prompt = self._load_system_prompt()
        self.llm_provider = LLMProvider(self.config)

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"provider": "local", "base_url": "http://localhost:1234/v1", "temperature": 0.7}

    def _load_system_prompt(self) -> str:
        if self.system_prompt_path.exists():
            try:
                with open(self.system_prompt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return "You are the StoryStudio Story Agent. Maintain narrative continuity and output structured scene directives."

    def interpret_story_line(
        self,
        current_line: str,
        line_index: int,
        full_story: List[str],
        continuity_state: Optional[Dict[str, Any]] = None,
        character_info: Optional[Dict[str, Any]] = None,
        default_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interprets a story line against previous context and continuity state.
        Tries LLM interpretation first; seamlessly falls back to heuristic parsing if offline.
        """
        continuity = continuity_state or {}
        character = character_info or {}
        defaults = default_settings or {}

        user_prompt = self._format_user_prompt(
            current_line=current_line,
            line_index=line_index,
            full_story=full_story,
            continuity=continuity,
            character=character,
            defaults=defaults
        )

        # 1. Attempt LLM generation
        try:
            llm_result = self.llm_provider.generate_json(self.system_prompt, user_prompt)
            if llm_result and isinstance(llm_result, dict):
                # Ensure essential fields exist
                return self._normalize_scene_directive(llm_result, character, continuity, defaults, current_line)
        except Exception:
            pass

        # 2. Heuristic offline analysis fallback
        return self._heuristic_story_interpretation(
            current_line=current_line,
            line_index=line_index,
            continuity=continuity,
            character=character,
            defaults=defaults
        )

    def _format_user_prompt(
        self,
        current_line: str,
        line_index: int,
        full_story: List[str],
        continuity: Dict[str, Any],
        character: Dict[str, Any],
        defaults: Dict[str, Any]
    ) -> str:
        previous_lines = full_story[:line_index] if full_story else []
        payload = {
            "current_line_number": line_index + 1,
            "current_story_line": current_line,
            "previous_story_lines": previous_lines[-5:],
            "active_continuity_state": {
                "character": continuity.get("character") or character.get("name", ""),
                "current_location": continuity.get("location", "initial setting"),
                "current_clothing": continuity.get("clothes_type", defaults.get("clothes_type", "casual dress")),
                "current_clothes_color": continuity.get("clothes_color", defaults.get("clothes_color", "white")),
                "current_clothes_details": continuity.get("clothes_details", defaults.get("clothes_details", [])),
                "last_emotion": continuity.get("emotion", "calm"),
                "last_action": continuity.get("action", "")
            },
            "character_reference": {
                "name": character.get("name", ""),
                "trigger": character.get("trigger", ""),
                "description": character.get("description", ""),
                "default_clothing": character.get("default_clothing", "")
            },
            "default_style": defaults.get("style", "romantic_anime")
        }
        return f"Interpret this story line and generate structured scene directives:\n```json\n{json.dumps(payload, indent=2)}\n```"

    def _normalize_scene_directive(
        self,
        raw: Dict[str, Any],
        character: Dict[str, Any],
        continuity: Dict[str, Any],
        defaults: Dict[str, Any],
        current_line: str
    ) -> Dict[str, Any]:
        return {
            "character": raw.get("character") or character.get("id") or character.get("name", "character"),
            "character_name": character.get("name", raw.get("character", "")),
            "character_trigger": character.get("trigger", ""),
            "character_description": character.get("description", ""),
            "action": raw.get("action", "posing"),
            "emotion": raw.get("emotion", "gentle expression"),
            "location": raw.get("location") or continuity.get("location", "luxurious room"),
            "clothes_type": raw.get("clothes_type") or continuity.get("clothes_type") or defaults.get("clothes_type", "dress"),
            "clothes_color": raw.get("clothes_color") or continuity.get("clothes_color") or defaults.get("clothes_color", "white"),
            "clothes_details": raw.get("clothes_details") or continuity.get("clothes_details") or defaults.get("clothes_details", []),
            "pose": raw.get("pose", "standing"),
            "pov": raw.get("pov", "three_quarter_front"),
            "scene": raw.get("scene_context", "romantic situation"),
            "scene_description": raw.get("scene_context", current_line),
            "style": raw.get("style") or defaults.get("style", "romantic_anime"),
            "continuity_summary": raw.get("continuity_summary", f"Located in {raw.get('location', 'scene')}, wearing {raw.get('clothes_color', '')} {raw.get('clothes_type', '')}"),
            "story_line_text": current_line
        }

    def _heuristic_story_interpretation(
        self,
        current_line: str,
        line_index: int,
        continuity: Dict[str, Any],
        character: Dict[str, Any],
        defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Intelligent offline heuristic parser that extracts actions, poses, emotions,
        environmental cues, and clothing states from raw story text.
        """
        text = current_line.lower()

        # 1. Location detection / preservation
        location = continuity.get("location") or "luxurious room"
        if "bedroom" in text:
            location = "private luxury bedroom with soft lighting"
        elif "mansion" in text or "hall" in text or "corridor" in text:
            location = "grand mansion hallway with ornate pillars"
        elif "mirror" in text or "dressing" in text or "vanity" in text:
            location = "elegant dressing room with ornate full-length mirror"
        elif "balcony" in text or "garden" in text:
            location = "moonlit balcony overlooking lush garden"
        elif "bed" in text:
            location = "luxurious bedroom near ornate soft bed"

        # 2. Clothing detection / preservation
        clothes_type = continuity.get("clothes_type") or defaults.get("clothes_type", "maid")
        clothes_color = continuity.get("clothes_color") or defaults.get("clothes_color", "white")
        clothes_details = list(continuity.get("clothes_details") or defaults.get("clothes_details", ["lace", "ribbons"]))

        if "maid" in text:
            clothes_type = "maid"
        elif "school" in text or "uniform" in text:
            clothes_type = "school_uniform"
        elif "kimono" in text or "yukata" in text:
            clothes_type = "traditional_kimono"
        elif "lingerie" in text or "sleepwear" in text or "undress" in text:
            clothes_type = "lingerie"
        elif "dress" in text or "outfit" in text:
            if "changes into" in text or "wearing" in text:
                clothes_type = "casual_dress"

        if "black" in text:
            clothes_color = "black"
        elif "white" in text:
            clothes_color = "white"
        elif "red" in text or "crimson" in text:
            clothes_color = "red"
        elif "blue" in text or "navy" in text:
            clothes_color = "blue"

        # 3. Pose & POV detection
        pose = "standing gracefully"
        pov = "three_quarter_front"

        if "mirror" in text:
            pose = "looking into full-length mirror, examining reflection"
            pov = "3/4 front view with mirror reflection"
        elif "bed" in text or "sit" in text:
            pose = "sitting gracefully on soft bed"
            pov = "3/4 front view"
        elif "turn" in text or "look back" in text or "shoulder" in text:
            pose = "turning around in surprise, looking back over shoulder"
            pov = "three-quarter back view"
        elif "walk" in text or "enters" in text or "arrives" in text:
            pose = "walking gently forward, looking around with curiosity"
            pov = "front view"
        elif "tied" in text or "bound" in text or "bondage" in text or "shibari" in text:
            pose = "tied up with ornate shibari ropes, kneeling bound"
            pov = "front view"
        elif "kneel" in text:
            pose = "kneeling gracefully"
            pov = "from slightly above"

        # 4. Emotion detection
        emotion = "gentle and curious"
        if "nervous" in text or "hesitant" in text:
            emotion = "nervous expression, subtle blushing cheeks, bashful gaze"
        elif "surprise" in text or "hear" in text or "shock" in text:
            emotion = "wide surprised eyes, flushed cheeks, startled yet alluring expression"
        elif "happy" in text or "smile" in text or "delighted" in text:
            emotion = "sweet radiant smile, tender affectionate eyes"
        elif "shy" in text or "blush" in text:
            emotion = "bashful blushing cheeks, looking down shyly"

        scene_ctx = f"{location}, {emotion}, {pose}"
        style = defaults.get("style", "romantic_anime")

        return {
            "character": character.get("id") or character.get("name", "character"),
            "character_name": character.get("name", "Character"),
            "character_trigger": character.get("trigger", ""),
            "character_description": character.get("description", ""),
            "action": f"performing: {current_line}",
            "emotion": emotion,
            "location": location,
            "clothes_type": clothes_type,
            "clothes_color": clothes_color,
            "clothes_details": clothes_details,
            "pose": pose,
            "pov": pov,
            "scene": scene_ctx,
            "scene_description": current_line,
            "style": style,
            "continuity_summary": f"Line {line_index + 1}: In {location}, wearing {clothes_color} {clothes_type}. Mood: {emotion}.",
            "story_line_text": current_line
        }
