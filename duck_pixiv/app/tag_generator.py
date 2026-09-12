"""Tag generator for Duck Pixiv Assistant.
Controlled tag dictionary lookup, fixed tags injection, deduplication, and max 10 tag prioritization.
"""
import os
import json
import re
from typing import List, Dict, Any, Set

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class TagGenerator:
    """Manages strictly controlled Japanese Pixiv tag derivation."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.tags_path = os.path.join(data_dir, "tags.json")
        self.fixed_tags_path = os.path.join(data_dir, "fixed_tags.json")
        self.characters_path = os.path.join(data_dir, "characters.json")
        self.refresh()

    def refresh(self):
        """Reloads dictionary files from disk without restarting application."""
        self.dictionary: Dict[str, str] = {}
        if os.path.exists(self.tags_path):
            with open(self.tags_path, "r", encoding="utf-8") as f:
                self.dictionary = json.load(f)

        self.fixed_tags: List[str] = []
        if os.path.exists(self.fixed_tags_path):
            with open(self.fixed_tags_path, "r", encoding="utf-8") as f:
                self.fixed_tags = json.load(f)

        self.characters: Dict[str, Any] = {}
        if os.path.exists(self.characters_path):
            with open(self.characters_path, "r", encoding="utf-8") as f:
                self.characters = json.load(f)

    @staticmethod
    def normalize_concept(concept: str) -> str:
        """Normalizes tag strings for dictionary lookup."""
        clean = concept.strip().lower()
        clean = clean.replace("_", " ")
        clean = re.sub(r"\s+", " ", clean)
        clean = re.sub(r"[\(\)\{\}\[\]]", "", clean)
        return clean.strip()

    def lookup_tag(self, concept: str) -> List[str]:
        """Looks up a tag or concept in the dictionary and character databases."""
        norm = self.normalize_concept(concept)
        results = []

        # 1. Check direct character / series match
        for char_key, char_info in self.characters.items():
            if norm == char_key.lower():
                if "default_tags" in char_info:
                    results.extend(char_info["default_tags"])
                else:
                    results.append(char_info.get("jp_name", char_key))
                    if char_info.get("series"):
                        results.append(char_info["series"])
                return results

        # 2. Check dictionary exact match (case-insensitive)
        for k, v in self.dictionary.items():
            if self.normalize_concept(k) == norm:
                results.append(v)
                return results

        # 3. Check partial key match for multi-word or compound tags
        # e.g. "red sensual lace lingerie" -> "ランジェリー", "レース"
        for k, v in self.dictionary.items():
            norm_k = self.normalize_concept(k)
            # Match word boundary or exact component
            pattern = r"\b" + re.escape(norm_k) + r"\b"
            if re.search(pattern, norm):
                if v not in results:
                    results.append(v)

        return results

    def generate_tags(self, metadata: Dict[str, Any], vision_tags: List[str] = None, max_tags: int = 10) -> List[str]:
        """
        Executes the tag pipeline:
        generation metadata -> normalize concepts -> dictionary lookup -> fixed tags -> vision-derived tags
        -> deduplicate -> prioritize -> maximum 10 tags
        """
        self.refresh()
        
        # Priority tiers
        # Tier 1: Character & Series tags (Highest)
        tier1_character_tags: List[str] = []
        # Tier 2: Core prompt tags (clothing, appearance, pose, expression)
        tier2_prompt_tags: List[str] = []
        # Tier 3: Fixed user tags (e.g. AIイラスト, 巨乳, ランジェリー, セクシー)
        tier3_fixed_tags: List[str] = list(self.fixed_tags)
        # Tier 4: Vision fallback tags
        tier4_vision_tags: List[str] = vision_tags or []

        prompt = metadata.get("prompt", "")
        extracted_tokens = [t.strip() for t in prompt.split(",") if t.strip()]

        # Check detected character from metadata reader
        char_detected = metadata.get("character_detected")
        if char_detected:
            char_res = self.lookup_tag(char_detected)
            for t in char_res:
                if t not in tier1_character_tags:
                    tier1_character_tags.append(t)

        for token in extracted_tokens:
            matched_jp = self.lookup_tag(token)
            for tag_jp in matched_jp:
                # Is it a character / series tag?
                is_char = False
                for c_info in self.characters.values():
                    if tag_jp == c_info.get("jp_name") or tag_jp == c_info.get("series") or tag_jp in c_info.get("default_tags", []):
                        if tag_jp not in tier1_character_tags:
                            tier1_character_tags.append(tag_jp)
                        is_char = True
                        break
                if not is_char:
                    if tag_jp not in tier2_prompt_tags:
                        tier2_prompt_tags.append(tag_jp)

        # Merge in priority order: Tier 1 -> Tier 2 -> Tier 3 -> Tier 4
        combined: List[str] = []
        seen: Set[str] = set()

        def add_tag(tag: str):
            if tag and tag not in seen:
                seen.add(tag)
                combined.append(tag)

        for t in tier1_character_tags:
            add_tag(t)

        for t in tier2_prompt_tags:
            add_tag(t)

        for t in tier3_fixed_tags:
            add_tag(t)

        for t in tier4_vision_tags:
            add_tag(t)

        # Truncate to max_tags (Default 10 for Pixiv)
        return combined[:max_tags]
