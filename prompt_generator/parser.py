import re
from typing import List, Set

class TagParser:
    """
    Deterministic post-processing for SDXL tags:
    - Normalizes whitespace and strips formatting noise
    - Case-insensitive deduplication while preserving original ordering
    - Preserves weighting syntax e.g. (tag:1.2), [tag], (tag)
    - Preserves ComfyUI / SDXL LoRA syntax e.g. <lora:name:0.8>
    """

    @staticmethod
    def clean_raw_output(text: str) -> str:
        """Strips markdown code blocks, prefixes like 'Tags:', and quotes."""
        if not text:
            return ""
        # Remove code blocks ```...```
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
        cleaned = re.sub(r"```$", "", cleaned.strip())

        # Remove common introductory phrases if output by LLM
        cleaned = re.sub(
            r"^(?:Tags|Prompt|Output|Visual Tags|SDXL Prompt|Result|Caption|JoyCaption|JoyCaption Prompt):\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        return cleaned.strip()

    @classmethod
    def parse_tags(cls, raw_text: str) -> List[str]:
        """
        Splits raw LLM output into clean, deduplicated, normalized tag list.
        """
        cleaned = cls.clean_raw_output(raw_text)
        if not cleaned:
            return []

        # Split by comma or newline
        # But do NOT split inside parentheses or angle brackets if possible
        # Standard approach: split by comma, and also split by newline
        raw_chunks: List[str] = []
        for line in cleaned.splitlines():
            line = line.strip()
            if not line:
                continue
            for part in line.split(","):
                raw_chunks.append(part)

        parsed_tags: List[str] = []
        seen_normalized: Set[str] = set()

        for chunk in raw_chunks:
            tag = chunk.strip()
            # Strip outer quotes if any
            if (tag.startswith('"') and tag.endswith('"')) or (tag.startswith("'") and tag.endswith("'")):
                tag = tag[1:-1].strip()

            # Normalize internal whitespace
            tag = re.sub(r"\s+", " ", tag)

            if not tag:
                continue

            # Normalized key for case-insensitive deduplication
            norm_key = tag.lower()

            if norm_key not in seen_normalized:
                seen_normalized.add(norm_key)
                parsed_tags.append(tag)

        return parsed_tags

    @classmethod
    def merge_prompts(cls, existing_prompt: str, new_tags: List[str], mode: str = "append") -> str:
        """
        Combines new tags with existing positive prompt.
        - replace: replaces existing prompt with new tags
        - append: appends new tags to existing prompt without duplicating existing tags or breaking LoRA syntax
        """
        if mode == "replace" or not existing_prompt.strip():
            return ", ".join(new_tags)

        # Parse existing prompt into tags to establish baseline
        existing_tags = cls.parse_tags(existing_prompt)
        existing_keys = {t.lower() for t in existing_tags}

        merged = list(existing_tags)
        for tag in new_tags:
            if tag.lower() not in existing_keys:
                merged.append(tag)
                existing_keys.add(tag.lower())

        return ", ".join(merged)
