import os
import sys
import json
import time
import base64
import io
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

# Add parent ImageStudio root to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import psutil
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

import config.paths as paths
from core.types import GenerationRequest, GenerationResult, ModelInfo, sanitize_seed, MAX_SEED, CharacterRegion
from core.model_manager import ModelManager
from core.lora_manager import LoraManager
from core.generation import GenerationCoordinator
from adapters.registry import adapter_registry

app = FastAPI(title="ImageStudio Tauri AI Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure all project directories (outputs, metadata, images) exist
paths.ensure_directories()

# Mount outputs directory for static serving
if paths.OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(paths.OUTPUTS_DIR)), name="outputs")

# Mount isolated Enhancer subsystem router
try:
    from enhancer.router import router as enhancer_router
    app.include_router(enhancer_router, prefix="/api/enhancer", tags=["enhancer"])
except Exception as e:
    print(f"[ImageStudio Bridge] Enhancer router not loaded: {e}")

# Mount StoryStudio subsystem router
try:
    from StoryStudio.router import router as story_router
    app.include_router(story_router, prefix="/api/story", tags=["story_studio"])
except Exception as e:
    print(f"[ImageStudio Bridge] StoryStudio router not loaded: {e}")

# Retouch Engine (LaMa Erase & Rembg Background Removal)
try:
    from engines.restoration import RestorationEngine
    retouch_engine = RestorationEngine()
except Exception as e:
    print(f"[ImageStudio Bridge] RestorationEngine not initialized: {e}")
    retouch_engine = None

@app.post("/api/retouch/remove-bg")
async def api_remove_bg(request: Request):
    if not retouch_engine:
        raise HTTPException(status_code=500, detail="Restoration engine not available")
    data = await request.json()
    img_b64 = data.get("image")
    if not img_b64:
        raise HTTPException(status_code=400, detail="Missing 'image' parameter")
    try:
        def _process():
            clean = img_b64.split("base64,")[1] if "base64," in img_b64 else img_b64
            pil_img = Image.open(io.BytesIO(base64.b64decode(clean))).convert("RGBA")
            result = retouch_engine.remove_background(pil_img)
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        res_b64 = await asyncio.to_thread(_process)
        print("[ImageStudio Bridge] Background cutout complete -> image sent to UI.")
        return {"success": True, "image": res_b64}
    except Exception as e:
        print(f"[ImageStudio Bridge] Error in remove-bg: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retouch/erase")
async def api_erase_object(request: Request):
    if not retouch_engine:
        raise HTTPException(status_code=500, detail="Restoration engine not available")
    data = await request.json()
    img_b64 = data.get("image")
    mask_b64 = data.get("mask")
    if not img_b64 or not mask_b64:
        raise HTTPException(status_code=400, detail="Missing 'image' or 'mask' parameter")
    try:
        def _process():
            clean_img = img_b64.split("base64,")[1] if "base64," in img_b64 else img_b64
            clean_mask = mask_b64.split("base64,")[1] if "base64," in mask_b64 else mask_b64
            pil_img = Image.open(io.BytesIO(base64.b64decode(clean_img))).convert("RGB")
            pil_mask = Image.open(io.BytesIO(base64.b64decode(clean_mask))).convert("L")
            result = retouch_engine.remove_object(pil_img, pil_mask)
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        res_b64 = await asyncio.to_thread(_process)
        print("[ImageStudio Bridge] Object erase complete -> image sent to UI.")
        return {"success": True, "image": res_b64}
    except Exception as e:
        print(f"[ImageStudio Bridge] Error in erase: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retouch/fix")
async def api_fix_artifacts(request: Request):
    data = await request.json()
    img_b64 = data.get("image")
    mask_b64 = data.get("mask")
    user_prompt = str(data.get("prompt", "") or "").strip()
    if not img_b64 or not mask_b64:
        raise HTTPException(status_code=400, detail="Missing 'image' or 'mask' parameter")

    prompt = user_prompt if user_prompt else "masterpiece, highly detailed, perfect anatomy, correct fingers and hands, anatomically correct limbs, flawless details, sharp focus, natural skin texture, 8k"
    negative_prompt = "bad anatomy, deformed fingers, extra fingers, missing fingers, mutated hands, poorly drawn hands, poorly drawn legs, extra limbs, blurry, distorted, deformed, artifacts, low quality"

    try:
        def _process():
            clean_img = img_b64.split("base64,")[1] if "base64," in img_b64 else img_b64
            clean_mask = mask_b64.split("base64,")[1] if "base64," in mask_b64 else mask_b64
            pil_img = Image.open(io.BytesIO(base64.b64decode(clean_img))).convert("RGB")
            pil_mask = Image.open(io.BytesIO(base64.b64decode(clean_mask))).convert("L")

            # Try generative SDXL inpainting first for realistic anatomy/object reconstruction
            try:
                model_info = resolve_model_info(data.get("model") or data.get("modelId"), preferred_arch="sdxl")
                req_w = (pil_img.width // 8) * 8
                req_h = (pil_img.height // 8) * 8
                if pil_img.size != (req_w, req_h):
                    pil_img = pil_img.resize((req_w, req_h), Image.Resampling.LANCZOS)
                if pil_mask.size != (req_w, req_h):
                    pil_mask = pil_mask.resize((req_w, req_h), Image.Resampling.NEAREST)

                gen_req = GenerationRequest(
                    model=model_info,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=req_w,
                    height=req_h,
                    steps=28,
                    guidance_scale=6.5,
                    sampler="Euler a",
                    scheduler="Normal",
                    seed=int(time.time()) % 2147483647,
                    init_image=pil_img,
                    mask_image=pil_mask,
                    mask_blur=2,
                    denoising_strength=0.85,
                )
                res = coordinator.generate(gen_req)
                result = Image.open(res.image_path)
            except Exception as gen_err:
                print(f"[ImageStudio Bridge] Generative fix fallback to LaMa: {gen_err}")
                if retouch_engine:
                    result = retouch_engine.fix_artifacts(pil_img, pil_mask)
                else:
                    raise gen_err

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        res_b64 = await asyncio.to_thread(_process)
        print("[ImageStudio Bridge] Generative artifact repair complete -> image sent to UI.")
        return {"success": True, "image": res_b64}
    except Exception as e:
        print(f"[ImageStudio Bridge] Error in fix: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# Global Engine Instances
from core.progress import progress_tracker
import ctypes

model_manager = ModelManager()
coordinator = GenerationCoordinator(model_manager)
is_cancelled = False


def get_nvml_status() -> Optional[Dict[str, Any]]:
    """Direct hardware NVML query via ctypes for true VRAM and GPU core load."""
    try:
        nvml = ctypes.CDLL("nvml.dll")
        nvml.nvmlInit_v2()
        handle = ctypes.c_void_p()
        nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))

        class Memory(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)
            ]
        mem = Memory()
        nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))

        class Utilization(ctypes.Structure):
            _fields_ = [
                ("gpu", ctypes.c_uint),
                ("memory", ctypes.c_uint)
            ]
        util = Utilization()
        nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util))

        name_buf = ctypes.create_string_buffer(64)
        nvml.nvmlDeviceGetName(handle, name_buf, 64)
        gpu_name = name_buf.value.decode("utf-8").replace("NVIDIA GeForce ", "")

        return {
            "gpuName": gpu_name,
            "vramUsedGb": round(mem.used / (1024 ** 3), 1),
            "vramTotalGb": round(mem.total / (1024 ** 3), 1),
            "gpuUtilPercent": int(util.gpu)
        }
    except Exception:
        return None


def resolve_model_info(model_key: Optional[str], preferred_arch: Optional[str] = None) -> ModelInfo:
    """Robust resolution of ModelInfo from any ID, stem, or filename."""
    if not model_key:
        # 1. Reuse already-loaded model in VRAM if it matches preferred arch (or none specified)
        if coordinator.active_engine and getattr(coordinator.active_engine, "current_model_info", None):
            cur = coordinator.active_engine.current_model_info
            if not preferred_arch or cur.architecture == preferred_arch:
                return cur
        # 2. Filter by preferred architecture (e.g. "sdxl" for inpaint/retouch)
        if preferred_arch:
            for m in model_manager.available_models.values():
                if m.architecture == preferred_arch:
                    return m
        # 3. Default to first available
        if model_manager.available_models:
            return next(iter(model_manager.available_models.values()))
        raise ValueError("No checkpoints found in models directory.")

    # 1. Exact dictionary key match
    if model_key in model_manager.available_models:
        candidate = model_manager.available_models[model_key]
        if not preferred_arch or candidate.architecture == preferred_arch:
            return candidate

    # 2. Match by stem (e.g. stripping .safetensors / .gguf)
    stem = Path(model_key).stem
    if stem in model_manager.available_models:
        candidate = model_manager.available_models[stem]
        if not preferred_arch or candidate.architecture == preferred_arch:
            return candidate

    # 3. Case-insensitive or filename match
    for k, m in model_manager.available_models.items():
        if k.lower() == model_key.lower() or k.lower() == stem.lower() or (m.transformer_path and Path(m.transformer_path).name.lower() == model_key.lower()):
            if not preferred_arch or m.architecture == preferred_arch:
                return m

    # 4. If preferred arch requested and user requested non-matching or missing, fallback to preferred arch
    if preferred_arch:
        if coordinator.active_engine and getattr(coordinator.active_engine, "current_model_info", None):
            cur = coordinator.active_engine.current_model_info
            if cur.architecture == preferred_arch:
                return cur
        for m in model_manager.available_models.values():
            if m.architecture == preferred_arch:
                return m

    raise ValueError(f"Requested checkpoint '{model_key}' is not available in registered models.")


@app.get("/api/system_status")
def get_system_status():
    nvml_data = get_nvml_status()
    if nvml_data:
        gpu_name = nvml_data["gpuName"]
        vram_used = nvml_data["vramUsedGb"]
        vram_total = nvml_data["vramTotalGb"]
        gpu_util = nvml_data["gpuUtilPercent"]
    elif torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
        vram_used = max(
            torch.cuda.memory_allocated(0),
            torch.cuda.memory_reserved(0)
        ) / (1024 ** 3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_util = 0
    else:
        gpu_name = "N/A"
        vram_used = 0.0
        vram_total = 0.0
        gpu_util = 0

    mem = psutil.virtual_memory()
    ram_used = mem.used / (1024 ** 3)
    ram_total = mem.total / (1024 ** 3)
    cpu_pct = int(psutil.cpu_percent())

    return {
        "gpuName": gpu_name,
        "vramUsedGb": round(vram_used, 1),
        "vramTotalGb": round(vram_total, 1),
        "gpuUtilPercent": gpu_util,
        "ramUsedGb": round(ram_used, 1),
        "ramTotalGb": round(ram_total, 1),
        "cpuPercent": cpu_pct,
    }


@app.get("/api/models")
def get_models():
    models = []
    model_manager.scan_models()
    for k, m in model_manager.available_models.items():
        arch = m.architecture
        size_gb = ""
        if m.transformer_path and Path(m.transformer_path).exists():
            size_gb = f" ? {round(Path(m.transformer_path).stat().st_size / (1024 ** 3), 2)} GB"
        models.append({
            "id": k,
            "name": k,
            "filename": Path(m.transformer_path or k).name,
            "path": (m.transformer_path or "").replace("\\", "/"),
            "architecture": arch,
            "description": f"{arch.upper()} Checkpoint{size_gb}",
        })
    models.sort(key=lambda m: (0 if "hassaku" in m["id"].lower() else (1 if m["architecture"] == "sdxl" else 2), m["id"].lower()))
    return models


@app.get("/api/loras")
def get_loras():
    loras = []
    scanned = adapter_registry.scan_loras()
    for name, p_str in scanned.items():
        p = Path(p_str)
        try:
            rel = p.relative_to(paths.LORAS_DIR)
            category = rel.parts[0] if len(rel.parts) > 1 else "General"
        except Exception:
            category = "General"
        category = category.capitalize()
        loras.append({
            "id": p.name,
            "filename": p.name,
            "displayName": p.stem,
            "category": category,
            "path": str(p).replace("\\", "/"),
            "architecture": "sdxl",
            "sizeBytes": p.stat().st_size if p.exists() else 0,
        })
    loras.sort(key=lambda x: (x["category"], x["displayName"].lower()))
    return loras


@app.get("/api/progress")
def get_progress():
    return progress_tracker.get_progress()


def _step_callback(step: int, total: int):
    progress_tracker.update(step, total)


def _run_generation_thread(req_dict: Dict[str, Any]):
    global is_cancelled
    try:
        requested_steps = int(req_dict.get("steps", 30))
        model_key = req_dict.get("modelId")
        model_info = resolve_model_info(model_key)

        progress_tracker.reset(total=requested_steps)
        progress_tracker.set_status("loading_model", f"Loading '{model_info.id}' into RTX GPU VRAM...", 0)
        progress_tracker.state["total"] = requested_steps
        is_cancelled = False

        upscale_method = req_dict.get("upscaleMethod")
        if upscale_method == "None" or not upscale_method:
            upscale_method = None
        upscale_factor = float(req_dict.get("upscaleFactor", 2.0))

        actual_seed = sanitize_seed(int(req_dict.get("seed", 0)))

        # Parse Init Image and Inpainting Mask Image if provided
        init_img_data = req_dict.get("initImage") or req_dict.get("init_image")
        mask_img_data = req_dict.get("maskImage") or req_dict.get("mask_image")
        init_pil = None
        mask_pil = None

        req_w = int(req_dict.get("width", 1024))
        req_h = int(req_dict.get("height", 1024))

        if init_img_data:
            if "," in init_img_data:
                init_img_data = init_img_data.split(",", 1)[1]
            init_pil = Image.open(io.BytesIO(base64.b64decode(init_img_data))).convert("RGB")
            if not req_dict.get("width"):
                req_w = (init_pil.width // 8) * 8
                req_h = (init_pil.height // 8) * 8
            init_pil = init_pil.resize((req_w, req_h), Image.Resampling.LANCZOS)

        if mask_img_data:
            if "," in mask_img_data:
                mask_img_data = mask_img_data.split(",", 1)[1]
            mask_pil = Image.open(io.BytesIO(base64.b64decode(mask_img_data))).convert("L")
            mask_pil = mask_pil.resize((req_w, req_h), Image.Resampling.NEAREST)

        # Parse Multi-Character regional definitions if provided
        characters = []
        raw_chars = req_dict.get("characters") or []
        for c in raw_chars:
            if isinstance(c, dict):
                c_prompt = c.get("prompt", "")
                c_lora = c.get("lora_name") or c.get("loraName")
                c_strength = float(c.get("lora_strength") or c.get("loraStrength") or 1.0)
                raw_box = c.get("box", [0.0, 0.0, 1.0, 1.0])
                box = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
                feather = float(c.get("feather", 0.04))
                characters.append(CharacterRegion(
                    prompt=c_prompt,
                    lora_name=c_lora,
                    lora_strength=c_strength,
                    box=box,
                    feather=feather
                ))

        req = GenerationRequest(
            model=model_info,
            prompt=req_dict.get("prompt", "") or "masterpiece, high quality",
            negative_prompt=req_dict.get("negativePrompt"),
            characters=characters,
            width=req_w,
            height=req_h,
            steps=requested_steps,
            guidance_scale=float(req_dict.get("cfgScale", 7.0)),
            sampler=req_dict.get("sampler", "Euler a"),
            scheduler=req_dict.get("scheduler", "Normal"),
            seed=actual_seed,
            loras=req_dict.get("loras", {}),
            init_image=init_pil,
            mask_image=mask_pil,
            mask_blur=int(req_dict.get("maskBlur", 4)),
            denoising_strength=float(req_dict.get("denoisingStrength", 0.75)),
            upscale_method=upscale_method,
            upscale_factor=upscale_factor,
            step_callback=_step_callback
        )

        progress_tracker.state["message"] = f"Generating with '{model_info.id}' on GPU..."
        res: GenerationResult = coordinator.generate(req)

        # Convert image to base64 Data URL for instant, lossless frontend delivery
        img_p = Path(res.image_path)
        with open(img_p, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_img}"

        progress_tracker.state["status"] = "complete"
        progress_tracker.state["step"] = requested_steps
        progress_tracker.state["total"] = requested_steps
        progress_tracker.state["percent"] = 100
        progress_tracker.state["message"] = f"Completed with '{model_info.id}' in {res.generation_time_ms / 1000:.2f}s"
        progress_tracker.state["result"] = {
            "imageUrl": data_url,
            "seed": actual_seed,
            "generationTimeMs": res.generation_time_ms,
            "vramPeakGb": res.vram_peak_gb,
            "metadata": req_dict
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_tracker.state["status"] = "error"
        progress_tracker.state["error"] = str(e)
        progress_tracker.state["message"] = f"Error: {e}"


@app.post("/api/generate")
async def generate(request: Request):
    data = await request.json()
    thread = threading.Thread(target=_run_generation_thread, args=(data,), daemon=True)
    thread.start()
    return {"status": "started"}


@app.post("/api/cancel")
def cancel():
    global is_cancelled
    is_cancelled = True
    progress_tracker.set_status("stopped", "Generation cancelled.", 0)
    return {"status": "cancelled"}


# ---------------------------------------------------------------------
# Inpainting Lab: Neural Face Detection & Auto-Lock Helper
# ---------------------------------------------------------------------
@app.post("/api/inpaint/detect_face")
async def inpaint_detect_face(request: Request):
    try:
        import numpy as np
        data = await request.json()
        img_data = data.get("image", "")
        if not img_data:
            raise HTTPException(status_code=400, detail="No image provided")
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        pil_img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGB")

        from enhancer.perception.adapters.face_parser import FaceParserAdapter
        face_parser = FaceParserAdapter()
        res = face_parser.predict(pil_img)
        raw_face = res.get("raw_face")
        if raw_face is None or not np.any(raw_face > 0.3):
            return {"success": False, "error": "No face detected in the image"}

        # Binary face mask (255 where face is present, 0 elsewhere)
        face_mask_arr = (raw_face > 0.3).astype(np.uint8) * 255
        face_mask_pil = Image.fromarray(face_mask_arr, mode="L")

        buf = io.BytesIO()
        face_mask_pil.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "success": True,
            "maskBase64": f"data:image/png;base64,{mask_b64}",
            "coveragePct": res.get("coverage_pct", 0),
            "confidence": res.get("confidence", 1.0),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/output_image")
def get_output_image(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(p), media_type="image/png")


# ---------------------------------------------------------------------
# LLM-Assisted SDXL Prompt/Tag Generator Endpoints
# ---------------------------------------------------------------------
try:
    from prompt_generator import (
        prompt_generator,
        profile_registry,
        provider_registry,
        settings_manager,
        PromptGenerateRequest,
        LLMSettings,
    )

    @app.get("/api/prompt_generator/profiles")
    def get_prompt_profiles():
        return [p.to_dict() for p in profile_registry.list_profiles()]

    @app.get("/api/prompt_generator/settings")
    def get_prompt_generator_settings():
        return settings_manager.get_settings().to_dict()

    @app.post("/api/prompt_generator/settings")
    async def save_prompt_generator_settings(request: Request):
        data = await request.json()
        saved = settings_manager.save_settings(data)
        return {"success": True, "settings": saved.to_dict()}

    @app.post("/api/prompt_generator/test_connection")
    async def test_prompt_generator_connection(request: Request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        settings = LLMSettings.from_dict(data) if data else settings_manager.get_settings()
        provider = provider_registry.get_provider(settings.provider)
        result = await provider.test_connection(settings)
        return result

    @app.post("/api/prompt_generator/generate")
    async def generate_prompt_tags(request: Request):
        data = await request.json()
        custom_settings = None
        if "settings" in data and isinstance(data["settings"], dict):
            custom_settings = LLMSettings.from_dict(data["settings"])

        req = PromptGenerateRequest(
            text=data.get("text", ""),
            profile=data.get("profile", "sdxl_base"),
            style=data.get("style", "General"),
            existing_prompt=data.get("existing_prompt", ""),
            mode=data.get("mode", "append"),
            settings=custom_settings,
        )
        resp = await prompt_generator.generate_prompt(req)
        return resp.to_dict()

except Exception as e:
    print(f"[ImageStudio Bridge] Prompt Generator router not loaded: {e}")



# ---------------------------------------------------------------------
# History & Vault API Endpoints
# ---------------------------------------------------------------------
try:
    from history.database import HistoryDB
    history_db = HistoryDB()

    @app.post("/api/history/sync")
    def sync_history():
        """Force manual synchronization of output images from disk."""
        result = history_db.sync_from_disk(force=True)
        return {"success": True, **result}

    @app.get("/api/history")
    def get_history(limit: int = 60, offset: int = 0, search: str = "", sync: bool = False):
        if sync:
            history_db.sync_from_disk(force=True)
        records = history_db.get_records(limit=limit, offset=offset, search=search)
        total = history_db.get_total_count(search=search)
        
        for r in records:
            img_p = Path(r.get("image_path", ""))
            if img_p.exists():
                r["is_available"] = True
                r["image_url"] = f"/api/output_image?path={img_p.as_posix()}"
            else:
                r["is_available"] = False
                r["image_url"] = ""
                
        return {
            "records": records,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @app.delete("/api/history/{record_id}")
    def delete_history_item(record_id: int, delete_file: bool = True):
        success = history_db.delete_record(record_id, delete_file=delete_file)
        return {"success": success}

except Exception as e:
    print(f"[ImageStudio Bridge] History router not initialized: {e}")


# ---------------------------------------------------------------------
# Composition & Identity Studio API Endpoints
# ---------------------------------------------------------------------
try:
    from pipelines.composition_pipeline import composition_coordinator
    from core.preprocessors import preprocessor_engine

    @app.get("/api/composition/models")
    def get_composition_models():
        return composition_coordinator.list_available_adapters()

    @app.post("/api/composition/preprocess")
    async def preprocess_image(request: Request):
        data = await request.json()
        raw_b64 = data.get("image_data_url", "")
        if "base64," in raw_b64:
            raw_b64 = raw_b64.split("base64,")[1]

        p_type = data.get("preprocessor_type", "openpose")
        img_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(img_bytes))

        processed = preprocessor_engine.preprocess(
            pil_img,
            preprocessor_type=p_type,
            low_threshold=data.get("low_threshold", 100),
            high_threshold=data.get("high_threshold", 200),
        )

        buf = io.BytesIO()
        processed.save(buf, format="PNG")
        out_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return {
            "success": True,
            "preprocessor_type": p_type,
            "preview_data_url": out_b64,
            "width": processed.width,
            "height": processed.height,
        }

    @app.post("/api/composition/generate")
    async def generate_composition(request: Request):
        data = await request.json()

        def _decode_img(b64_str: Optional[str]) -> Optional[Image.Image]:
            if not b64_str:
                return None
            clean = b64_str.split("base64,")[1] if "base64," in b64_str else b64_str
            return Image.open(io.BytesIO(base64.b64decode(clean)))

        identity_img = _decode_img(data.get("identity_image_data_url"))
        comp_img = _decode_img(data.get("composition_image_data_url"))
        prep_img = _decode_img(data.get("preprocessed_control_data_url"))
        style_img = _decode_img(data.get("style_image_data_url"))

        result = composition_coordinator.generate(
            prompt=data.get("prompt", "1girl, highly detailed"),
            negative_prompt=data.get("negative_prompt", "low quality, bad anatomy"),
            architecture=data.get("architecture", "sdxl"),
            model_id=data.get("model_id"),
            identity_image=identity_img,
            identity_engine=data.get("identity_engine", "pulid"),
            identity_strength=float(data.get("identity_strength", 0.85)),
            composition_image=comp_img,
            controlnet_type=data.get("controlnet_type", "openpose"),
            controlnet_strength=float(data.get("controlnet_strength", 0.75)),
            controlnet_preprocessed=prep_img,
            style_image=style_img,
            style_strength=float(data.get("style_strength", 0.60)),
            style_mode=data.get("style_mode", "style_only"),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            steps=int(data.get("steps", 28)),
            cfg_scale=float(data.get("cfg_scale", 6.0)),
            seed=int(data.get("seed", -1)),
        )
        return result

except Exception as e:
    print(f"[ImageStudio Bridge] Composition router not loaded: {e}")


def free_port(port: int = 8188):
    """Terminates any stale background process holding the port before starting."""
    try:
        import psutil
        import os
        current_pid = os.getpid()
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid and conn.pid != current_pid:
                try:
                    p = psutil.Process(conn.pid)
                    print(f"[ImageStudio Bridge] Freeing port {port} by terminating stale process {conn.pid} ({p.name()})...")
                    p.kill()
                except Exception:
                    pass
    except Exception as e:
        print(f"[ImageStudio Bridge] Notice while checking port {port}: {e}")


if __name__ == "__main__":
    free_port(8188)
    print("[ImageStudio Bridge] Starting Python GPU AI Bridge on http://127.0.0.1:8188 ...")
    try:
        uvicorn.run(app, host="127.0.0.1", port=8188, log_level="warning")
    except Exception as e:
        import traceback
        traceback.print_exc()

