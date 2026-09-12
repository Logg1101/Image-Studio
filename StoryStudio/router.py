from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from StoryStudio.core.catalogue_manager import CatalogueManager
from StoryStudio.core.project_manager import ProjectManager
from StoryStudio.core.pipeline import StoryStudioPipeline
from StoryStudio.agents.prompt_agent.agent import PromptAgent
from StoryStudio.agents.story_agent.agent import StoryAgent
from core.progress import progress_tracker

router = APIRouter()

catalogue_mgr = CatalogueManager()
project_mgr = ProjectManager()
prompt_agent = PromptAgent()
story_agent = StoryAgent()
pipeline = StoryStudioPipeline(
    catalogue_manager=catalogue_mgr,
    project_manager=project_mgr,
    prompt_agent=prompt_agent,
    story_agent=story_agent
)

@router.get("/catalogues")
def get_catalogues():
    return catalogue_mgr.get_all_catalogues()

@router.get("/characters")
def get_characters():
    return catalogue_mgr.scan_characters()

@router.get("/progress")
def get_story_progress():
    return progress_tracker.get_progress()

@router.post("/prompt_agent")
async def preview_prompt_agent(request: Request):
    data = await request.json()
    return pipeline.prepare_single_prompt(data)

@router.post("/story_agent")
async def preview_story_agent(request: Request):
    data = await request.json()
    char_id = data.get("character_id")
    char_info = catalogue_mgr.get_character_by_id(char_id) if char_id else {}
    return story_agent.interpret_story_line(
        current_line=data.get("current_line", ""),
        line_index=int(data.get("line_index", 0)),
        full_story=data.get("full_story", []),
        continuity_state=data.get("continuity_state", {}),
        character_info=char_info,
        default_settings=data.get("default_settings", {})
    )

from starlette.concurrency import run_in_threadpool

def _update_bridge_progress(step: int, total: int):
    progress_tracker.update(step, total)

@router.post("/generate_single")
async def generate_single(request: Request):
    data = await request.json()
    try:
        steps = int(data.get("steps", 30))
        progress_tracker.reset()
        progress_tracker.set_status("generating", "Starting GPU sampling...", 0)
        progress_tracker.state["total"] = steps
        
        # Execute in background threadpool so the asyncio event loop can answer /api/progress and /api/system_status in real time
        res = await run_in_threadpool(pipeline.generate_single, data, step_callback=_update_bridge_progress)
        progress_tracker.set_status("complete", "Generation completed", 100)
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_tracker.set_status("error", f"Error: {e}", 0)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_batch")
async def generate_batch(request: Request):
    data = await request.json()
    batch_count = int(data.get("batch_count", 1))
    try:
        results = await run_in_threadpool(pipeline.generate_batch, data, batch_count=batch_count)
        return {"count": len(results), "results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upscale_image")
async def upscale_image(request: Request):
    """
    On-demand neural enhancement / upscaling for any image using 4x Real-CUGAN or Tiled SDXL.
    """
    import io
    import base64
    from PIL import Image

    data = await request.json()
    data_url = data.get("image_url", "")
    model_id = str(data.get("model_id", "4x-realcugan")).lower().strip()
    scale = int(data.get("scale", 2))

    if not data_url:
        raise HTTPException(status_code=400, detail="Missing image_url")

    try:
        pil_img = None
        if data_url.startswith("data:") and "base64," in data_url:
            raw_b64 = data_url.split("base64,")[1]
            img_bytes = base64.b64decode(raw_b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif data_url.startswith(("http://", "https://")):
            def _fetch_url(url: str):
                import urllib.request
                with urllib.request.urlopen(url, timeout=30) as resp:
                    return Image.open(io.BytesIO(resp.read())).convert("RGB")
            pil_img = await run_in_threadpool(_fetch_url, data_url)
        else:
            def _load_local_or_b64(raw: str):
                try:
                    if len(raw) < 1024 and Path(raw).is_file():
                        return Image.open(raw).convert("RGB")
                except OSError:
                    pass
                raw_b64 = raw.split("base64,")[1] if "base64," in raw else raw
                img_bytes = base64.b64decode(raw_b64)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")

            pil_img = await run_in_threadpool(_load_local_or_b64, data_url)

        if "tiled" in model_id or "sdxl" in model_id:
            from enhancer.upscaling.adapters.tiled_sdxl import TiledSDXLUpscaler
            tiled_scaler = TiledSDXLUpscaler()
            out_img = await run_in_threadpool(
                tiled_scaler.upscale,
                image=pil_img,
                scale=float(scale),
                prompt=data.get("prompt", ""),
                negative_prompt=data.get("negative_prompt", ""),
                enable_tactile_materials=data.get("enable_tactile_materials", True),
            )
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            b64_out = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {
                "image_url": f"data:image/png;base64,{b64_out}",
                "output_path": "",
                "width": out_img.width,
                "height": out_img.height,
                "scale": scale,
                "model_id": "tiled-sdxl",
                "success": True
            }
        else:
            from enhancer.upscaling.engine import enhancement_engine
            enhanced = await run_in_threadpool(
                enhancement_engine.enhance,
                image=pil_img,
                scale=scale,
                model_id=model_id,
                sharpen=25.0,
                denoise=15.0
            )
            out_res = enhanced.get("output_resolution", [pil_img.width * scale, pil_img.height * scale])
            return {
                "image_url": enhanced.get("image_url", ""),
                "output_path": enhanced.get("output_path", ""),
                "width": out_res[0],
                "height": out_res[1],
                "scale": scale,
                "model_id": model_id,
                "success": True
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upscaling failed: {e}")

@router.post("/decompose_story")
async def decompose_story(request: Request):
    data = await request.json()
    story_text = data.get("story_text", "")
    char_id = data.get("character_id", "")
    defaults = data.get("default_settings", {})
    try:
        scenes = await run_in_threadpool(
            pipeline.decompose_story_to_scenes,
            story_text=story_text,
            character_id=char_id,
            default_settings=defaults
        )
        return {"scenes": scenes, "count": len(scenes)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects")
def list_projects():
    return project_mgr.list_projects()

@router.post("/projects/create")
async def create_project(request: Request):
    data = await request.json()
    name = data.get("name", "Untitled Story")
    lines = data.get("lines", [])
    scenes = data.get("scenes")
    raw_story_text = data.get("story_text", "")
    char_id = data.get("character", "")
    defaults = data.get("default_settings", {})
    return project_mgr.create_or_update_project(
        name=name,
        story_lines=lines,
        character_id=char_id,
        default_settings=defaults,
        scenes_data=scenes,
        raw_story_text=raw_story_text
    )

@router.post("/projects/{project_name}/update_scenes")
async def update_project_scenes(project_name: str, request: Request):
    data = await request.json()
    scenes = data.get("scenes", [])
    updated = project_mgr.update_project_scenes(project_name, scenes)
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    return updated

@router.get("/projects/{project_name}")
def get_project(project_name: str):
    p = project_mgr.get_project(project_name)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    return p

@router.post("/projects/{project_name}/generate_line")
async def generate_project_line(project_name: str, request: Request):
    data = await request.json()
    line_index = int(data.get("line_index", 0))
    try:
        return await run_in_threadpool(
            pipeline.generate_story_line,
            project_name,
            line_index,
            step_callback=_update_bridge_progress
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_name}/images/{image_name}")
def get_project_image(project_name: str, image_name: str):
    p_dir = project_mgr.get_project_dir(project_name)
    img_path = p_dir / "images" / image_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(img_path), media_type="image/png")

@router.post("/cancel")
def cancel_story_generation():
    pipeline.cancel()
    return {"status": "cancelled"}
