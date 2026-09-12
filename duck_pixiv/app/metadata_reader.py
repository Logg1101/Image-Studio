"""Metadata reader for Duck Pixiv Assistant.
Reads embedded PNG chunks (parameters / prompt / negative_prompt) as well as companion
ImageStudio JSON metadata in outputs/metadata/*.json or relative paths.
"""
import os
import json
import re
from typing import Dict, Any, Optional, List
from PIL import Image


class MetadataReader:
    """Extracts image generation metadata safely from PNG images and companion files."""

    @staticmethod
    def read_image_metadata(image_path: str) -> Dict[str, Any]:
        """
        Reads metadata from:
        1. PNG tEXt chunks (parameters / prompt / workflow / metadata)
        2. Companion JSON file next to image or in outputs/metadata/<basename>.json
        """
        result: Dict[str, Any] = {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "model": "",
            "architecture": "",
            "prompt": "",
            "negative_prompt": "",
            "seed": None,
            "steps": None,
            "width": None,
            "height": None,
            "raw_info": {},
            "character_detected": None,
            "features": {
                "character": [],
                "clothing": [],
                "appearance": [],
                "pose": [],
                "expression": [],
                "materials": [],
                "environment": [],
                "general_tags": []
            }
        }

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        # 1. Inspect companion JSON in outputs/metadata/ or same directory
        companion_data = MetadataReader._find_companion_json(image_path)
        if companion_data:
            result.update({
                "model": companion_data.get("model", ""),
                "architecture": companion_data.get("architecture", ""),
                "prompt": companion_data.get("prompt", ""),
                "negative_prompt": companion_data.get("negative_prompt", ""),
                "seed": companion_data.get("seed"),
                "steps": companion_data.get("steps"),
                "width": companion_data.get("width"),
                "height": companion_data.get("height"),
                "raw_info": companion_data
            })

        # 2. Inspect embedded PNG chunks
        try:
            with Image.open(image_path) as img:
                result["width"] = result["width"] or img.width
                result["height"] = result["height"] or img.height

                info = img.info or {}
                # Look for parameters or prompt
                for key in ["parameters", "prompt", "metadata", "Comment"]:
                    val = info.get(key)
                    if not val:
                        continue
                    
                    if isinstance(val, str):
                        try:
                            parsed_json = json.loads(val)
                            if isinstance(parsed_json, dict):
                                if not result["prompt"] and "prompt" in parsed_json:
                                    result["prompt"] = parsed_json.get("prompt", "")
                                if not result["negative_prompt"] and "negative_prompt" in parsed_json:
                                    result["negative_prompt"] = parsed_json.get("negative_prompt", "")
                                if not result["model"] and "model" in parsed_json:
                                    result["model"] = parsed_json.get("model", "")
                                if not result["seed"] and "seed" in parsed_json:
                                    result["seed"] = parsed_json.get("seed")
                                result["raw_info"].update(parsed_json)
                                continue
                        except Exception:
                            pass
                        
                        # Plain text parameter block (e.g. A1111 / WebUI style)
                        if not result["prompt"]:
                            result["prompt"] = val
                    elif isinstance(val, dict):
                        if not result["prompt"] and "prompt" in val:
                            result["prompt"] = val.get("prompt", "")
                        result["raw_info"].update(val)
        except Exception as e:
            # Continue even if image read encounters non-fatal warning
            pass

        # 3. Categorize prompt concepts into structured features
        MetadataReader._categorize_prompt(result)

        return result

    @staticmethod
    def _find_companion_json(image_path: str) -> Optional[Dict[str, Any]]:
        """Checks potential companion metadata json locations."""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.dirname(image_path)

        # Candidates:
        # A) Same dir with .json extension
        same_dir_json = os.path.join(image_dir, f"{base_name}.json")
        if os.path.exists(same_dir_json):
            try:
                with open(same_dir_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # B) Look upwards for outputs/metadata/<base_name>.json
        current = os.path.abspath(image_dir)
        for _ in range(5):
            candidate = os.path.join(current, "outputs", "metadata", f"{base_name}.json")
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass

            meta_direct = os.path.join(current, "metadata", f"{base_name}.json")
            if os.path.exists(meta_direct):
                try:
                    with open(meta_direct, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass

            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

        return None

    @staticmethod
    def _categorize_prompt(meta: Dict[str, Any]) -> None:
        """Parses the raw prompt string into distinct categories."""
        prompt = meta.get("prompt", "")
        if not prompt:
            return

        # Split comma-delimited tags
        tags = [t.strip() for t in prompt.split(",") if t.strip()]

        clothing_keywords = {
            "lingerie", "lace", "bra", "panties", "underwear", "sleepwear", "babydoll", 
            "garter", "stockings", "gloves", "headdress", "ribbons", "kimono", "yukata", 
            "dress", "bikini", "swimsuit", "bunny suit", "skirt", "corset"
        }
        appearance_keywords = {
            "hair", "eyes", "breasts", "skin", "body", "nipples", "cleavage", "navel",
            "collarbone", "thighs", "legs", "shoulders", "twintails", "ponytail"
        }
        pose_keywords = {
            "kneeling", "sitting", "standing", "lying", "tied", "bound", "restrained",
            "hands", "arms", "legs", "posture", "binding", "strappado", "folded"
        }
        expression_keywords = {
            "blush", "blushing", "flustered", "smile", "frown", "panicked", "tears",
            "sweat", "eyes", "crying", "embarrassed", "nervous", "seductive", "gaze"
        }
        materials_keywords = {
            "lace", "leather", "satin", "silk", "embroidery", "straps", "rope", "ropes",
            "chrome", "metal", "hemp", "chain", "duct tape", "cloth"
        }

        for tag in tags:
            tag_clean = tag.strip().lower()
            if tag_clean in ["masterpiece", "best quality", "ultra-detailed", "highly detailed", "1girl"]:
                continue

            # Check character keywords
            if any(char.lower() in tag_clean for char in ["belfast", "taihou", "akeno", "velina"]):
                meta["features"]["character"].append(tag)
                if not meta["character_detected"]:
                    meta["character_detected"] = tag
                continue

            matched = False
            for k in clothing_keywords:
                if k in tag_clean:
                    meta["features"]["clothing"].append(tag)
                    matched = True
                    break
            for k in expression_keywords:
                if k in tag_clean:
                    meta["features"]["expression"].append(tag)
                    matched = True
                    break
            for k in pose_keywords:
                if k in tag_clean:
                    meta["features"]["pose"].append(tag)
                    matched = True
                    break
            for k in appearance_keywords:
                if k in tag_clean:
                    meta["features"]["appearance"].append(tag)
                    matched = True
                    break
            for k in materials_keywords:
                if k in tag_clean:
                    meta["features"]["materials"].append(tag)
                    matched = True
                    break

            if not matched:
                meta["features"]["general_tags"].append(tag)
