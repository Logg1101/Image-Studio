"""Image analyzer fallback for Duck Pixiv Assistant.
Analyzes image dimensions, dominant colors, and provides fallback features when metadata is incomplete.
"""
import os
from typing import Dict, Any, List
from PIL import Image


class ImageAnalyzer:
    """Fallback / supplementary vision analysis tool."""

    @staticmethod
    def analyze_image(image_path: str) -> Dict[str, Any]:
        """Performs quick local image inspection for supplementary metadata."""
        if not os.path.exists(image_path):
            return {"error": "画像が見つかりません", "vision_tags": []}

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                
                # Check aspect ratio / format
                aspect_ratio = round(width / max(height, 1), 2)
                is_portrait = height > width
                
                # Sample colors to detect general mood (warm / cool / dark)
                img_small = img.resize((32, 32)).convert("RGB")
                pixels = list(img_small.getdata())
                avg_r = sum(p[0] for p in pixels) / len(pixels)
                avg_g = sum(p[1] for p in pixels) / len(pixels)
                avg_b = sum(p[2] for p in pixels) / len(pixels)
                
                brightness = (avg_r + avg_g + avg_b) / 3.0
                
                vision_tags = []
                if brightness < 60:
                    vision_tags.append("夜")
                elif avg_r > avg_b + 30 and avg_r > avg_g:
                    vision_tags.append("夕焼け")

                return {
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "aspect_ratio": aspect_ratio,
                    "is_portrait": is_portrait,
                    "brightness": round(brightness, 1),
                    "vision_tags": vision_tags
                }
        except Exception as e:
            return {"error": str(e), "vision_tags": []}
