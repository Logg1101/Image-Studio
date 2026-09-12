"""Post manager for Duck Pixiv Assistant.
Handles preparing post payloads, saving post metadata independently, multi-image orchestration,
and coordinating with the uploader and database.
"""
import os
import json
import time
from typing import Dict, Any, List, Optional
from app.metadata_reader import MetadataReader
from app.tag_generator import TagGenerator
from app.title_generator import TitleGenerator
from app.image_analyzer import ImageAnalyzer
from app.pixiv_uploader import get_uploader, PixivUploadResult
from app.database import record_post

METADATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metadata")


class PostManager:
    """Manages the full lifecycle of preparing and executing a Pixiv post."""

    def __init__(self):
        self.meta_reader = MetadataReader()
        self.tag_gen = TagGenerator()
        self.title_gen = TitleGenerator()
        self.img_analyzer = ImageAnalyzer()
        os.makedirs(METADATA_DIR, exist_ok=True)

    def prepare_post(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Reads metadata from the selected image(s), analyzes tags, and produces title & description candidates.
        Multi-image support: Combines features from all selected images, with the first image as primary.
        """
        if not image_paths:
            raise ValueError("画像が選択されていません。")

        images_info = []
        combined_features: Dict[str, List[str]] = {
            "character": [],
            "clothing": [],
            "appearance": [],
            "pose": [],
            "expression": [],
            "materials": [],
            "environment": [],
            "general_tags": []
        }
        primary_meta = None

        for idx, path in enumerate(image_paths):
            meta = self.meta_reader.read_image_metadata(path)
            if idx == 0:
                primary_meta = meta
            images_info.append(meta)

            # Aggregate features
            for cat, items in meta.get("features", {}).items():
                for item in items:
                    if item not in combined_features[cat]:
                        combined_features[cat].append(item)

        # Supplementary vision analysis on primary image
        vision_info = self.img_analyzer.analyze_image(image_paths[0])
        vision_tags = vision_info.get("vision_tags", [])

        # Generate tags using controlled dictionary pipeline
        tags = self.tag_gen.generate_tags(primary_meta, vision_tags=vision_tags, max_tags=10)

        # Generate 3-5 Japanese title candidates (with English translations)
        title_candidates = self.title_gen.generate_titles(primary_meta, count=5)
        selected_title = title_candidates[0]["ja"] if title_candidates else ""

        # Generate description (both Japanese and English)
        desc_obj = self.title_gen.generate_description(primary_meta, selected_title)

        post_data = {
            "image_paths": image_paths,
            "primary_image": image_paths[0],
            "primary_meta": primary_meta,
            "images_info": images_info,
            "extracted_features": combined_features,
            "title_candidates": title_candidates,
            "selected_title": selected_title,
            "description": desc_obj.get("ja", ""),
            "description_en": desc_obj.get("en", ""),
            "tags": tags,
            "is_ai": True,
            "age_limit": "r18",  # Default based on erotic/bondage/sensual artwork
            "created_at": time.time()
        }

        return post_data

    def save_post_metadata(self, post_data: Dict[str, Any], status: str = "draft", pixiv_id: Optional[str] = None) -> str:
        """
        Saves prepared post metadata to a separate JSON file (e.g. metadata/Belfast_001.json).
        Does not touch or modify the original source images or outputs.
        """
        first_img = post_data.get("primary_image", "")
        base_name = os.path.splitext(os.path.basename(first_img))[0] if first_img else "post"
        
        meta_filename = f"{base_name}.json"
        target_path = os.path.join(METADATA_DIR, meta_filename)

        payload = {
            "source_images": post_data.get("image_paths", []),
            "extracted_metadata": post_data.get("primary_meta", {}),
            "title_candidates": post_data.get("title_candidates", []),
            "selected_title": post_data.get("selected_title", ""),
            "description": post_data.get("description", ""),
            "generated_tags": post_data.get("tags", []),
            "final_edited_tags": post_data.get("tags", []),
            "is_ai": post_data.get("is_ai", True),
            "age_limit": post_data.get("age_limit", "r18"),
            "posting_status": status,
            "pixiv_work_id": pixiv_id or "",
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return target_path

    def execute_post(self, post_data: Dict[str, Any], settings: Optional[Dict[str, Any]] = None) -> PixivUploadResult:
        """
        Sends the prepared post to Pixiv via the isolated PixivUploader,
        records results in SQLite database, and saves metadata file.
        """
        uploader = get_uploader(settings)
        
        image_paths = post_data.get("image_paths", [])
        title = post_data.get("selected_title", "")
        description = post_data.get("description", "")
        tags = post_data.get("tags", [])
        is_ai = post_data.get("is_ai", True)
        age_limit = post_data.get("age_limit", "r18")

        result = uploader.upload(
            image_paths=image_paths,
            title=title,
            description=description,
            tags=tags,
            is_ai=is_ai,
            age_limit=age_limit
        )

        status = "success" if result.success else "failed"
        meta_file = self.save_post_metadata(post_data, status=status, pixiv_id=result.pixiv_id)

        # Record to SQLite database
        record_post(
            title=title,
            description=description,
            tags=tags,
            image_paths=image_paths,
            status=status,
            pixiv_id=result.pixiv_id,
            error_message=result.message if not result.success else None,
            metadata_file=meta_file
        )

        return result
