from typing import Dict, Any, Optional, List

class ContinuityTracker:
    """
    Tracks and updates visual and narrative continuity across episodic story generations.
    Ensures clothing, location, character features, and ongoing context persist cleanly
    unless explicitly altered by the narrative.
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.state: Dict[str, Any] = {
            "character_id": "",
            "character_name": "",
            "character_trigger": "",
            "character_description": "",
            "clothes_type": "",
            "clothes_color": "",
            "clothes_details": [],
            "location": "",
            "emotion": "calm and focused",
            "action": "posing",
            "pose": "standing",
            "pov": "three_quarter_front",
            "scene_context": "",
            "style": "",
            "history_events": []
        }
        if initial_state:
            self.state.update(initial_state)

    def get_state(self) -> Dict[str, Any]:
        return dict(self.state)

    def update_from_directive(self, directive: Dict[str, Any]):
        """Updates internal continuity state based on structured directives from the Story Agent."""
        if directive.get("character"):
            self.state["character_id"] = directive["character"]
        if directive.get("character_name"):
            self.state["character_name"] = directive["character_name"]
        if directive.get("character_trigger"):
            self.state["character_trigger"] = directive["character_trigger"]
        if directive.get("character_description"):
            self.state["character_description"] = directive["character_description"]
        if directive.get("clothes_type"):
            self.state["clothes_type"] = directive["clothes_type"]
        if directive.get("clothes_color"):
            self.state["clothes_color"] = directive["clothes_color"]
        if directive.get("clothes_details") is not None:
            self.state["clothes_details"] = list(directive["clothes_details"])
        if directive.get("location"):
            self.state["location"] = directive["location"]
        if directive.get("emotion"):
            self.state["emotion"] = directive["emotion"]
        if directive.get("action"):
            self.state["action"] = directive["action"]
        if directive.get("pose"):
            self.state["pose"] = directive["pose"]
        if directive.get("pov"):
            self.state["pov"] = directive["pov"]
        if directive.get("scene_context"):
            self.state["scene_context"] = directive["scene_context"]
        if directive.get("style"):
            self.state["style"] = directive["style"]

        summary = directive.get("continuity_summary") or directive.get("story_line_text")
        if summary:
            self.state["history_events"].append(summary)

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        self.__init__(initial_state)

    def serialize(self) -> Dict[str, Any]:
        return self.get_state()

    def deserialize(self, data: Dict[str, Any]):
        if isinstance(data, dict):
            self.state.update(data)
