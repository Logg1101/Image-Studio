import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from StoryStudio.agents.llm_provider import LLMProvider

class PromptAgent:
    """
    AI Prompt Agent for StoryStudio.
    Converts structured selections (character, clothes, pose, pov, scene, style)
    into high-cohesion, quality-focused positive and negative prompts tailored
    for Hassaku XL Illustrious and SDXL generation.
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
        return "You are the StoryStudio Prompt Agent. Generate positive and negative prompts as structured JSON."

    def generate_prompts(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Generates positive and negative prompts from structured inputs.
        Attempts LLM generation first; seamlessly falls back to rule-based synthesis if offline.
        """
        user_prompt = self._format_user_prompt(params)
        
        # 1. Attempt LLM generation
        try:
            llm_result = self.llm_provider.generate_json(self.system_prompt, user_prompt)
            if llm_result and "positive_prompt" in llm_result and "negative_prompt" in llm_result:
                pos = str(llm_result.get("positive_prompt", "")).strip()
                neg = str(llm_result.get("negative_prompt", "")).strip()
                if pos:
                    return {
                        "positive_prompt": pos,
                        "negative_prompt": neg or self._default_negative_prompt()
                    }
        except Exception:
            pass

        # 2. Rule-based offline heuristic synthesis fallback
        return self._synthesize_fallback_prompt(params)

    def _format_user_prompt(self, params: Dict[str, Any]) -> str:
        payload = {
            "character_name": params.get("character_name") or params.get("character", ""),
            "character_trigger": params.get("character_trigger", ""),
            "character_description": params.get("character_description", ""),
            "expression": params.get("expression", ""),
            "expression_prompt": params.get("expression_prompt", ""),
            "clothes_type": params.get("clothes_type", ""),
            "clothes_color": params.get("clothes_color", ""),
            "clothes_details": params.get("clothes_details", []),
            "pose": params.get("pose", ""),
            "pov": params.get("pov", ""),
            "bondage_style": params.get("bondage_style") or params.get("bondage", ""),
            "gag_type": params.get("gag_type") or params.get("gag", ""),
            "scene": params.get("scene", ""),
            "scene_description": params.get("scene_description", ""),
            "additional_context": params.get("additional_context", "")
        }
        return f"Synthesize generation prompts for the following structured configuration:\n```json\n{json.dumps(payload, indent=2)}\n```"

    # Hardcoded canonical character descriptions
    CHARACTER_HARDCODED_TRIGGERS = {
        "belfast": "Belfast, long white hair, blue eyes, large breasts",
        "akeno": "Akeno Himejima, Long black hair, purple eyes, large breasts",
        "rias": "Rias Gremory, long red hair, green eyes, large breasts",
        "taihou": "Taihou, long black hair, red eyes, large breasts",
        "velina": "Velina Airgid, long white hair, purple eyes, large breasts",
        "rita": "Rita, short brown hair, red eyes, large breasts"
    }

    def _resolve_character_tag(self, params: Dict[str, Any]) -> str:
        """Resolves canonical character appearance tags with hardcoded overrides."""
        char_id = str(params.get("character_id") or "").lower()
        char_name = str(params.get("character_name") or params.get("character") or "").lower()
        trigger = str(params.get("character_trigger") or "")
        char_desc = str(params.get("character_description") or "")

        for key, hardcoded_str in self.CHARACTER_HARDCODED_TRIGGERS.items():
            if key in char_id or key in char_name or key in trigger.lower():
                return hardcoded_str

        if trigger:
            return trigger
        if char_desc:
            return char_desc
        if char_name and char_name != "none":
            return params.get("character_name") or params.get("character") or ""
        return ""

    def _synthesize_fallback_prompt(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Deterministic, high-quality rule-based synthesizer tailored for Hassaku XL Illustrious.
        Assembles tags in strict semantic order.
        """
        segments: List[str] = [
            "masterpiece",
            "best quality",
            "ultra-detailed",
            "1girl"
        ]

        # 1. Character LoRA Trigger & Appearance (with hardcoded mappings)
        char_tag = self._resolve_character_tag(params)
        if char_tag:
            segments.append(char_tag)

        # 2. Facial Expression
        expr_prompt = params.get("expression_prompt") or params.get("expression") or ""
        if expr_prompt and str(expr_prompt).lower() not in ("none", "null", ""):
            segments.append(expr_prompt)

        # 2. Clothing Configuration
        c_color = (params.get("clothes_color") or "").strip().lower()
        c_type = (params.get("clothes_type_prompt") or params.get("clothes_type") or "").strip()
        c_details = params.get("clothes_details") or []
        if isinstance(c_details, str):
            c_details = [c_details]

        clothes_clauses = []
        if c_color and c_type:
            clothes_clauses.append(f"{c_color} {c_type}")
        elif c_type:
            clothes_clauses.append(c_type)
        elif c_color:
            clothes_clauses.append(f"{c_color} attire")

        for d in c_details:
            if d and d not in clothes_clauses:
                clothes_clauses.append(d)

        if clothes_clauses:
            segments.append(", ".join(clothes_clauses))

        # 3. Bondage & Restraints Suite
        bondage_type = params.get("bondage_type_prompt") or params.get("bondage_type") or ""
        if bondage_type and str(bondage_type).lower() not in ("none", "null", ""):
            segments.append(bondage_type)

        bondage_style = params.get("bondage_style_prompt") or params.get("bondage_prompt") or params.get("bondage_style") or params.get("bondage") or ""
        if bondage_style and str(bondage_style).lower() not in ("none", "null", ""):
            segments.append(bondage_style)

        bondage_hands = params.get("bondage_hands_prompt") or params.get("bondage_hands") or ""
        if bondage_hands and str(bondage_hands).lower() not in ("none", "null", ""):
            segments.append(bondage_hands)

        bondage_legs = params.get("bondage_legs_prompt") or params.get("bondage_legs") or ""
        if bondage_legs and str(bondage_legs).lower() not in ("none", "null", ""):
            segments.append(bondage_legs)

        bondage_accessories = params.get("bondage_accessories_prompt") or params.get("bondage_accessories") or ""
        if bondage_accessories and str(bondage_accessories).lower() not in ("none", "null", ""):
            segments.append(bondage_accessories)

        # 4. Gag Type (if selected)
        gag_prompt = params.get("gag_prompt") or params.get("gag_type") or params.get("gag") or ""
        if gag_prompt and str(gag_prompt).lower() not in ("none", "null", ""):
            segments.append(gag_prompt)

        # 5. Pose
        pose_prompt = params.get("pose_prompt") or params.get("pose") or ""
        if pose_prompt:
            segments.append(pose_prompt)

        # 6. POV / Camera Angle
        pov_prompt = params.get("pov_prompt") or params.get("pov") or ""
        if pov_prompt:
            segments.append(pov_prompt)

        # 7. Scene Context
        scene_prompt = params.get("scene_prompt") or params.get("scene_description") or params.get("scene") or ""
        if scene_prompt:
            segments.append(scene_prompt)

        # 8. Additional Context / Story Line
        add_ctx = params.get("additional_context") or ""
        if add_ctx:
            segments.append(add_ctx)

        # 9. Fixed Aesthetic Suffix
        segments.append("realistic fabric texture, realistic lighting, cinematic")

        # Clean duplicates while preserving order
        seen = set()
        clean_parts = []
        for seg in segments:
            for piece in [p.strip() for p in seg.split(",") if p.strip()]:
                low = piece.lower()
                if low not in seen:
                    seen.add(low)
                    clean_parts.append(piece)

        positive_prompt = ", ".join(clean_parts)
        negative_prompt = self._default_negative_prompt()

        return {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt
        }

    @staticmethod
    def _default_negative_prompt() -> str:
        return (
            "worst quality, low quality, bad anatomy, bad hands, missing fingers, extra fingers, "
            "fused fingers, distorted limbs, mutated hands, blurry, watermark, signature, text, "
            "cropped, out of frame, deformed, duplicate, morbid, bad proportions, disfigured"
        )
