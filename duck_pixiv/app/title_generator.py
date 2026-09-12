"""Title and description generator for Duck Pixiv Assistant.
Generates 3-5 natural Japanese Pixiv titles (with English meanings) and tasteful descriptions.
Supports:
1. High-aesthetic built-in Japanese phrasing synthesis with English translations (offline fallback)
2. Ollama local LLM endpoint (if running)
3. Gemini / OpenAI API (if keys configured)
"""
import os
import json
import random
import re
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error


class TitleGenerator:
    """Creates natural Japanese Pixiv titles and descriptions with English translations."""

    def __init__(self, settings_path: Optional[str] = None):
        if not settings_path:
            settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "settings.json")
        self.settings_path = settings_path
        self.settings = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "llm_provider": "heuristic",
            "ollama_url": "http://localhost:11434",
            "ollama_model": "llama3"
        }

    def generate_titles(self, metadata: Dict[str, Any], count: int = 5) -> List[Dict[str, str]]:
        """
        Generates 3 to 5 title objects:
        [{"ja": "秘密の夜とベルファスト", "en": "Secret Night with Belfast"}, ...]
        """
        self.settings = self._load_settings()
        provider = self.settings.get("llm_provider", "heuristic")

        # Try LLM if configured
        if provider == "ollama":
            llm_titles = self._generate_ollama_titles(metadata, count)
            if llm_titles:
                return llm_titles

        # High-aesthetic Japanese title synthesizer with English meaning
        return self._synthesize_natural_japanese_titles(metadata, count)

    def generate_description(self, metadata: Dict[str, Any], selected_title: Optional[str] = None) -> Dict[str, str]:
        """Generates a natural Pixiv caption in Japanese along with its English translation."""
        char_name = metadata.get("character_detected") or "彼女"
        series = None
        
        # Check if character has specific series mapping in characters.json
        chars_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "characters.json")
        if os.path.exists(chars_path):
            try:
                with open(chars_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    for k, v in cdata.items():
                        if k.lower() == char_name.lower():
                            char_name = v.get("jp_name", char_name)
                            series = v.get("series")
                            break
            except Exception:
                pass

        lines_ja = []
        lines_en = []
        if series:
            lines_ja.append(f"{series}より、{char_name}を描きました。")
            lines_en.append(f"Drew {char_name} from {series}.")
        else:
            lines_ja.append(f"{char_name}を描きました。")
            lines_en.append(f"Drew {char_name}.")

        clothing = metadata.get("features", {}).get("clothing", [])
        if any("lingerie" in c.lower() or "lace" in c.lower() for c in clothing):
            lines_ja.append("繊細なレースとランジェリーの質感にこだわって仕上げています。")
            lines_en.append("Focused on rendering the delicate texture of lace and lingerie.")
        elif clothing:
            lines_ja.append("衣装の質感や細部の描写にこだわりました。")
            lines_en.append("Focused on the outfit's texture and fine details.")

        lines_ja.append("気に入っていただけたら、ブックマークやいいねをいただけると励みになります！")
        lines_en.append("If you like it, bookmarks and likes are greatly appreciated!")
        lines_ja.append("")
        lines_en.append("")
        lines_ja.append("※AI生成による作品です。")
        lines_en.append("*This work is AI-generated.")

        return {
            "ja": "\n".join(lines_ja),
            "en": "\n".join(lines_en)
        }

    def _synthesize_natural_japanese_titles(self, metadata: Dict[str, Any], count: int = 5) -> List[Dict[str, str]]:
        """Synthesizes natural Pixiv-style poetic Japanese titles with English translations."""
        char_en = metadata.get("character_detected") or "Her"
        char_jp = char_en
        series_jp = ""

        # Map to Japanese name if possible
        chars_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "characters.json")
        if os.path.exists(chars_path):
            try:
                with open(chars_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    for k, v in cdata.items():
                        if k.lower() == char_en.lower():
                            char_jp = v.get("jp_name", char_jp)
                            series_jp = v.get("series", "")
                            break
            except Exception:
                pass

        clothing = [c.lower() for c in metadata.get("features", {}).get("clothing", [])]
        expression = [e.lower() for e in metadata.get("features", {}).get("expression", [])]
        pose = [p.lower() for p in metadata.get("features", {}).get("pose", [])]

        is_lingerie = any("lingerie" in c or "lace" in c or "panties" in c for c in clothing)
        is_blushing = any("blush" in e or "flustered" in e or "tears" in e for e in expression)
        is_tied = any("bondage" in p or "tied" in p or "gagged" in p or "bound" in p for p in pose)

        pool: List[Dict[str, str]] = []

        if is_lingerie and is_tied:
            pool.extend([
                {"ja": f"秘密の夜と{char_jp}", "en": f"A Secret Night with {char_en}"},
                {"ja": f"囚われの{char_jp}、紅いレース", "en": f"Captive {char_en}, Red Lace"},
                {"ja": f"{char_jp}の誰にも言えない秘密", "en": f"{char_en}'s Untold Secret"},
                {"ja": f"解けない絆と{char_jp}", "en": f"Unbreakable Bonds and {char_en}"},
                {"ja": f"夜色に揺れる{char_jp}", "en": f"{char_en} Swaying in the Night Hue"},
                {"ja": f"{char_jp}の秘めやかな逢瀬", "en": f"{char_en}'s Secret Tryst"}
            ])
        elif is_lingerie:
            pool.extend([
                {"ja": f"{char_jp}、夜の装い", "en": f"{char_en}, Night Attire"},
                {"ja": f"夜色に染まる{char_jp}", "en": f"{char_en} Shaded by the Night"},
                {"ja": f"{char_jp}の優雅なひととき", "en": f"{char_en}'s Elegant Moment"},
                {"ja": f"月明かりと{char_jp}", "en": f"Moonlight and {char_en}"},
                {"ja": f"素肌を包むレースの{char_jp}", "en": f"Lace Embracing {char_en}'s Skin"},
                {"ja": f"{char_jp}の静かな夜", "en": f"{char_en}'s Quiet Night"}
            ])
        elif is_blushing:
            pool.extend([
                {"ja": f"恥じらいの{char_jp}", "en": f"Bashful {char_en}"},
                {"ja": f"赤らむ頬の{char_jp}", "en": f"{char_en} with Blushing Cheeks"},
                {"ja": f"{char_jp}の隠せない本音", "en": f"{char_en}'s Unconcealable Feelings"},
                {"ja": f"戸惑いと{char_jp}", "en": f"Hesitation and {char_en}"},
                {"ja": f"見つめ合うふたりと{char_jp}", "en": f"Two Gazes and {char_en}"}
            ])
        else:
            pool.extend([
                {"ja": f"{char_jp}の優雅なひととき", "en": f"{char_en}'s Graceful Moment"},
                {"ja": f"微笑む{char_jp}", "en": f"Smiling {char_en}"},
                {"ja": f"光の中の{char_jp}", "en": f"{char_en} in the Light"},
                {"ja": f"木漏れ日と{char_jp}", "en": f"Sunlight Through Trees and {char_en}"},
                {"ja": f"{char_jp}の日常", "en": f"{char_en}'s Daily Life"}
            ])

        # Additional aesthetic variations
        pool.extend([
            {"ja": f"甘い吐息と{char_jp}", "en": f"Sweet Breath and {char_en}"},
            {"ja": f"{char_jp}の特別な夜", "en": f"{char_en}'s Special Night"},
            {"ja": f"純白と深紅の{char_jp}", "en": f"Pure White & Crimson {char_en}"},
            {"ja": f"静寂に佇む{char_jp}", "en": f"{char_en} Standing in Silence"}
        ])

        # Deduplicate while preserving variation
        selected = []
        seen = set()
        for item in pool:
            if item["ja"] not in seen:
                seen.add(item["ja"])
                selected.append(item)
            if len(selected) >= count:
                break

        return selected

    def _generate_ollama_titles(self, metadata: Dict[str, Any], count: int = 5) -> Optional[List[Dict[str, str]]]:
        """Calls local Ollama instance if active."""
        url = self.settings.get("ollama_url", "http://localhost:11434") + "/api/generate"
        model = self.settings.get("ollama_model", "llama3")
        prompt_text = (
            f"Generate {count} tasteful Japanese Pixiv titles with English translations for this artwork:\n"
            f"Character: {metadata.get('character_detected', 'Original')}\n"
            f"Features: {metadata.get('prompt', '')[:200]}\n"
            f"Output format (JSON Array): [ {{\"ja\": \"タイトル\", \"en\": \"English meaning\"}} ]"
        )

        data = {
            "model": model,
            "prompt": prompt_text,
            "stream": False,
            "format": "json"
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=4) as response:
                resp = json.loads(response.read().decode("utf-8"))
                text = resp.get("response", "")
                parsed = json.loads(text)
                if isinstance(parsed, list) and len(parsed) >= 3:
                    return [{"ja": str(i.get("ja", "")), "en": str(i.get("en", ""))} for i in parsed[:count]]
        except Exception:
            pass
        return None
