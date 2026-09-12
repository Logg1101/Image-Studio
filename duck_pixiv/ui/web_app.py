"""Web application router and server for Duck Pixiv Assistant."""
import os
import glob
import json
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.post_manager import PostManager
from app.metadata_reader import MetadataReader
from app.tag_generator import TagGenerator
from app.title_generator import TitleGenerator
from app.database import get_all_posts

app = FastAPI(title="Duck Pixiv Assistant")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "ui", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "ui", "templates")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_TMP = os.path.join(BASE_DIR, "scratch", "uploads")
os.makedirs(UPLOADS_TMP, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

post_manager = PostManager()
metadata_reader = MetadataReader()
tag_generator = TagGenerator()
title_generator = TitleGenerator()


# Request Models
class PreparePostRequest(BaseModel):
    image_paths: List[str]


class RegenerateTitleRequest(BaseModel):
    metadata: Dict[str, Any]


class RegenerateDescRequest(BaseModel):
    metadata: Dict[str, Any]
    selected_title: Optional[str] = None


class RegenerateTagsRequest(BaseModel):
    metadata: Dict[str, Any]


class SubmitPostRequest(BaseModel):
    image_paths: List[str]
    primary_meta: Optional[Dict[str, Any]] = None
    title_candidates: Optional[List[Any]] = None
    selected_title: str
    description: str
    tags: List[str]
    is_ai: bool = True
    age_limit: str = "r18"


@app.get("/", response_class=FileResponse)
def serve_index():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="index.html が見つかりません。")
    return FileResponse(index_file, media_type="text/html")


@app.get("/api/recent_outputs")
def get_recent_outputs():
    """Finds recently generated images in ImageStudio outputs without modifying anything."""
    image_extensions = ("*.png", "*.jpg", "*.webp")
    found_images = []

    # Look for outputs directory in parent
    parent_dir = os.path.dirname(BASE_DIR)
    outputs_dir = os.path.join(parent_dir, "outputs")

    if os.path.exists(outputs_dir):
        for ext in image_extensions:
            pattern = os.path.join(outputs_dir, "**", ext)
            for path in glob.glob(pattern, recursive=True):
                # Ignore metadata directory images if any
                if "metadata" in path.split(os.sep):
                    continue
                mtime = os.path.getmtime(path)
                found_images.append({
                    "path": os.path.abspath(path),
                    "filename": os.path.basename(path),
                    "mtime": mtime
                })

    # Sort descending by modification time
    found_images.sort(key=lambda x: x["mtime"], reverse=True)
    return {"images": found_images[:32]}


@app.get("/api/image_file")
def get_image_file(path: str):
    """Safely serves an image file for thumbnail preview."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path)


@app.get("/api/explore_folder")
def explore_folder(folder_path: Optional[str] = None):
    """
    Lists subfolders and image files with thumbnails in the specified directory.
    Defaults to ImageStudio's outputs directory.
    """
    parent_dir = os.path.dirname(BASE_DIR)
    default_root = os.path.join(parent_dir, "outputs")

    target = os.path.abspath(folder_path) if folder_path else os.path.abspath(default_root)
    if not os.path.exists(target) or not os.path.isdir(target):
        target = os.path.abspath(default_root)

    subfolders = []
    images = []
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}

    # Parent directory for navigation
    parent_target = os.path.dirname(target) if target != os.path.abspath(default_root) else None

    try:
        entries = sorted(os.scandir(target), key=lambda e: (not e.is_dir(), e.name.lower()))
        for e in entries:
            # Skip hidden and metadata folders
            if e.name.startswith(".") or e.name == "metadata":
                continue
            if e.is_dir():
                subfolders.append({
                    "name": e.name,
                    "path": os.path.abspath(e.path)
                })
            elif e.is_file():
                ext = os.path.splitext(e.name)[1].lower()
                if ext in valid_exts:
                    images.append({
                        "name": e.name,
                        "path": os.path.abspath(e.path),
                        "mtime": e.stat().st_mtime
                    })
    except Exception as err:
        return {"error": str(err), "current_path": target, "subfolders": [], "images": []}

    # Sort images by newest first
    images.sort(key=lambda x: x["mtime"], reverse=True)

    return {
        "current_path": target,
        "parent_path": parent_target,
        "subfolders": subfolders,
        "images": images
    }


@app.post("/api/save_settings")
def save_settings(settings: Dict[str, Any] = Body(...)):
    """Saves user settings including Pixiv account details."""
    settings_file = os.path.join(DATA_DIR, "settings.json")
    existing = {}
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    existing.update(settings)
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return {"status": "success", "settings": existing}


@app.post("/api/upload_local_files")
def upload_local_files(files: List[UploadFile] = File(...)):
    """Receives user-selected files from the file picker."""
    saved_paths = []
    for f in files:
        safe_name = os.path.basename(f.filename)
        dest = os.path.join(UPLOADS_TMP, safe_name)
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_paths.append(os.path.abspath(dest))
    return {"paths": saved_paths}


@app.post("/api/prepare_post")
async def prepare_post(req: PreparePostRequest):
    try:
        post_data = post_manager.prepare_post(req.image_paths)
        return JSONResponse(content=post_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/regenerate_titles")
async def regenerate_titles(req: RegenerateTitleRequest):
    titles = title_generator.generate_titles(req.metadata, count=5)
    return {"titles": titles}


@app.post("/api/regenerate_description")
async def regenerate_description(req: RegenerateDescRequest):
    desc = title_generator.generate_description(req.metadata, req.selected_title)
    return {"description": desc}


@app.post("/api/regenerate_tags")
async def regenerate_tags(req: RegenerateTagsRequest):
    tags = tag_generator.generate_tags(req.metadata, max_tags=10)
    return {"tags": tags}


@app.post("/api/submit_post")
async def submit_post(req: SubmitPostRequest):
    try:
        post_data = req.dict()
        post_data["primary_image"] = req.image_paths[0] if req.image_paths else ""
        res = post_manager.execute_post(post_data)
        return JSONResponse(content=res.to_dict())
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"エラー: {str(e)}", "pixiv_id": None})


@app.get("/api/history")
async def get_history():
    posts = get_all_posts(limit=30)
    return JSONResponse(content=posts)


@app.get("/api/settings")
def get_settings():
    settings_file = os.path.join(DATA_DIR, "settings.json")
    if os.path.exists(settings_file):
        with open(settings_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@app.get("/api/get_data_files")
def get_data_files():
    tags_p = os.path.join(DATA_DIR, "tags.json")
    fixed_p = os.path.join(DATA_DIR, "fixed_tags.json")
    chars_p = os.path.join(DATA_DIR, "characters.json")

    tags = {}
    if os.path.exists(tags_p):
        with open(tags_p, "r", encoding="utf-8") as f:
            tags = json.load(f)

    fixed = []
    if os.path.exists(fixed_p):
        with open(fixed_p, "r", encoding="utf-8") as f:
            fixed = json.load(f)

    chars = {}
    if os.path.exists(chars_p):
        with open(chars_p, "r", encoding="utf-8") as f:
            chars = json.load(f)

    return {"tags": tags, "fixed_tags": fixed, "characters": chars}


@app.post("/api/save_data_files")
def save_data_files(payload: Dict[str, Any] = Body(...)):
    tags = payload.get("tags")
    fixed = payload.get("fixed_tags")
    chars = payload.get("characters")

    if tags is not None:
        with open(os.path.join(DATA_DIR, "tags.json"), "w", encoding="utf-8") as f:
            json.dump(tags, f, ensure_ascii=False, indent=2)
    if fixed is not None:
        with open(os.path.join(DATA_DIR, "fixed_tags.json"), "w", encoding="utf-8") as f:
            json.dump(fixed, f, ensure_ascii=False, indent=2)
    if chars is not None:
        with open(os.path.join(DATA_DIR, "characters.json"), "w", encoding="utf-8") as f:
            json.dump(chars, f, ensure_ascii=False, indent=2)

    # Refresh in-memory dictionaries
    tag_generator.refresh()
    return {"status": "success"}
