import os
import random
import time
import json
import psutil
import torch
import gradio as gr
from PIL import Image, ImageEnhance, ImageFile
from pathlib import Path
import config.paths as paths
from ui.state import coordinator, model_manager, lora_manager
from core.types import GenerationRequest
from core.adapter_manager import adapter_manager
from engines.sdxl.schedulers import AVAILABLE_SAMPLERS, AVAILABLE_SCHEDULERS
from engines.restoration import RestorationEngine
from engines.supir_enhancer import SupirEngine
from core.prompt_enhancer import AIPromptEnhancer
from core.memory import clear_vram

ImageFile.LOAD_TRUNCATED_IMAGES = True
restoration_engine = RestorationEngine()
supir_engine = SupirEngine()
try:
    from engines.cogvideo_engine import AnimationEngine
    animation_engine = AnimationEngine()
except (ImportError, ModuleNotFoundError):
    animation_engine = None

# --- HARDWARE & SYSTEM MONITORS ---

def get_hardware_stats_html():
    cpu_usage = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024 ** 3)
    ram_total = ram.total / (1024 ** 3)
    
    vram_str = "VRAM: N/A"
    gpu_name = "CPU Mode"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        vram_str = f"VRAM: {allocated:.1f}/{total_vram:.1f} GB"
        
    return f"""
    <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
        <span style="display: inline-flex; align-items: center; gap: 6px; background: #151D2C; border: 1px solid #243046; padding: 4px 12px; border-radius: 9999px; color: #E2E8F0;">
            <span style="color: #10B981; font-size: 9px;">●</span>
            <strong>{gpu_name}</strong>
        </span>
        <span style="background: #151D2C; border: 1px solid #243046; padding: 4px 12px; border-radius: 9999px; color: #38BDF8; font-weight: 600;">
            {vram_str}
        </span>
        <span style="background: #151D2C; border: 1px solid #243046; padding: 4px 12px; border-radius: 9999px; color: #94A3B8;">
            RAM: {ram_used:.1f}/{ram_total:.1f} GB
        </span>
        <span style="background: #151D2C; border: 1px solid #243046; padding: 4px 12px; border-radius: 9999px; color: #94A3B8;">
            CPU: {cpu_usage:.0f}%
        </span>
    </div>
    """

def get_bottom_status_html(model_name="None"):
    return f"""
    <div id="status-bar">
        <span style="color: #38BDF8; margin-right: 6px;">⚡</span>
        <strong>ImageStudio</strong> &nbsp;|&nbsp; Active Model: <span style="color: #818CF8; font-weight: 600;">{model_name or 'None'}</span> &nbsp;|&nbsp; 12 GB VRAM Mode &nbsp;|&nbsp; Viewport Optimized
    </div>
    """

def force_unload_vram():
    coordinator.unload_all()
    clear_vram()
    time.sleep(0.5)
    return get_hardware_stats_html()

# --- UTILITIES & PROMPT HELPERS ---

def set_aspect_ratio_radio(ratio_str):
    mapping = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "4:3": (1152, 896),
        "3:4": (896, 1152)
    }
    w, h = mapping.get(ratio_str, (1024, 1024))
    return gr.update(value=w), gr.update(value=h)

def enhance_prompt_text(prompt: str, model_id: str = None) -> str:
    arch = "sdxl"
    if model_id:
        try:
            arch = model_manager.get_model_info(model_id).architecture
        except Exception:
            arch = "sdxl"
    return AIPromptEnhancer.enhance(prompt, architecture=arch)

def get_random_prompt_text(model_id: str = None) -> str:
    arch = "sdxl"
    if model_id:
        try:
            arch = model_manager.get_model_info(model_id).architecture.lower()
        except Exception:
            arch = "sdxl"
    if "flux" in arch:
        prompts = [
            "A cinematic photograph of a serene traveler sitting by a glowing hearth in a mountain cabin during a blizzard, warm ambient shadows, highly detailed.",
            "An intricate digital painting of an ornate cybernetic samurai in a rain-slicked neo-Tokyo street, reflections of vibrant neon signs, ultra-detailed.",
            "A delicate watercolor illustration of a secluded botanical greenhouse brimming with lush ferns and tropical flowers under soft afternoon sunlight.",
            "A striking studio portrait of an expressive model with natural freckles and windswept hair, dramatic chiaroscuro rim lighting, sharp 85mm focus.",
        ]
    else:
        prompts = [
            "1girl, solo, futuristic cyber armor, neon-lit alleyway, cinematic lighting, rain reflections, highly detailed, masterpiece",
            "1girl, solo, elegant silk kimono, cherry blossom grove, golden hour sunlight, ethereal atmosphere, masterpiece",
            "1girl, celestial priestess holding glowing orb, starry night sky, ancient temple ruins, intricate details",
            "1boy, street artist, vibrant urban spray-paint murals, dynamic action pose, twilight glow, rim lighting",
            "1girl, fantasy archer atop misty mountain peak, wind-swept cloak, sunrise rays, panoramic composition",
            "cozy coffee shop interior on a rainy afternoon, warm ambient glow, soft bokeh reflections, nostalgic mood",
            "1girl, vintage steampunk goggles, mechanical wings, brass airship deck, dramatic volumetric clouds",
            "mythical dragon coiled around ancient crystal obelisk, vibrant aurora borealis, epic fantasy concept art",
        ]
    return random.choice(prompts)

def count_tokens_text(text: str) -> str:
    words = len(text.split()) if text else 0
    est_tokens = int(words * 1.3)
    return f"<span class='token-pill'>~{est_tokens} tokens ({len(text)} chars)</span>"

def get_categorized_loras():
    all_loras = lora_manager.scan_loras()
    categorized = {
        "characters": ["None"],
        "concepts": ["None"],
        "poses": ["None"],
        "styles": ["None"]
    }
    for name in all_loras.keys():
        lower_name = name.lower()
        if "character" in lower_name or "char" in lower_name:
            categorized["characters"].append(name)
        elif "concept" in lower_name:
            categorized["concepts"].append(name)
        elif "pose" in lower_name:
            categorized["poses"].append(name)
        elif "style" in lower_name:
            categorized["styles"].append(name)
        else:
            categorized["characters"].append(name)
    return categorized

def get_upscale_models():
    supir_dir = paths.MODELS_DIR / "SUPIR"
    if supir_dir.exists():
        models = [p.name for p in supir_dir.glob("*.safetensors")] + [p.name for p in supir_dir.glob("*.pth")] + [p.name for p in supir_dir.glob("*.pt")] + [p.name for p in supir_dir.glob("*.bin")]
        if models:
            return models
    return ["SUPIR-v0Q_fp16.safetensors"]

def on_model_change(model_id):
    if not model_id:
        return gr.update(), gr.update()
    try:
        arch = model_manager.get_model_info(model_id).architecture.lower()
        if arch == "sdxl":
            return (
                gr.update(value=28, minimum=10, maximum=60, label="Sampling Steps (SDXL: 25-35)"),
                gr.update(value=7.0, minimum=1.0, maximum=15.0, step=0.5, label="CFG Guidance (SDXL: 5-8)")
            )
        else:
            return (
                gr.update(value=8, minimum=1, maximum=50, label="Sampling Steps (Flux: 4-8)"),
                gr.update(value=1.0, minimum=0.0, maximum=10.0, step=0.1, label="CFG Guidance (Flux: 1.0)")
            )
    except Exception:
        return gr.update(), gr.update()

# --- GALLERY & INSPECTOR ---

def get_all_images():
    if not paths.IMAGES_DIR.exists():
        return [], []
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [str(p) for p in paths.IMAGES_DIR.iterdir() if p.suffix.lower() in valid_exts]
    image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return image_files, image_files

def inspect_image_formatted(evt: gr.SelectData, image_list):
    try:
        image_path = image_list[evt.index]
        with Image.open(image_path) as img:
            params = img.info.get("parameters", "")
            if params:
                data = json.loads(params)
                meta_html = f"""
                <div style="background: #0E1420; border: 1px solid #243046; border-radius: 10px; padding: 14px; font-family: 'Inter', sans-serif;">
                    <div style="font-weight: 700; color: #10B981; margin-bottom: 10px; font-size: 12px; letter-spacing: 0.5px;">📋 GENERATION METADATA</div>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin-bottom: 12px;">
                        <div><span style="color: #94A3B8;">Model:</span> <strong style="color: #F8FAFC;">{data.get('model', 'Unknown')}</strong></div>
                        <div><span style="color: #94A3B8;">Resolution:</span> <strong style="color: #F8FAFC;">{data.get('width', 1024)}x{data.get('height', 1024)}</strong></div>
                        <div><span style="color: #94A3B8;">Steps:</span> <strong style="color: #F8FAFC;">{data.get('steps', 28)}</strong></div>
                        <div><span style="color: #94A3B8;">CFG:</span> <strong style="color: #F8FAFC;">{data.get('guidance_scale', 7.0)}</strong></div>
                        <div><span style="color: #94A3B8;">Seed:</span> <strong style="color: #38BDF8;">{data.get('seed', 'N/A')}</strong></div>
                        <div><span style="color: #94A3B8;">Arch:</span> <strong style="color: #818CF8;">{str(data.get('architecture', 'sdxl')).upper()}</strong></div>
                    </div>
                    <div style="border-top: 1px solid #243046; padding-top: 10px;">
                        <div style="font-weight: 600; color: #94A3B8; font-size: 11px; margin-bottom: 4px;">PROMPT</div>
                        <div style="color: #F8FAFC; font-size: 11px; background: #151D2C; padding: 8px 10px; border-radius: 6px; border: 1px solid #243046; max-height: 100px; overflow-y: auto;">{data.get('prompt', '')}</div>
                    </div>
                </div>
                """
                return meta_html, data, image_path
    except Exception as e:
        print(f"Failed to read metadata: {e}")
        
    fallback_html = """
    <div style="background: #0E1420; border: 1px solid #243046; border-radius: 10px; padding: 14px; color: #94A3B8; font-size: 12px;">
        <em>No metadata embedded in image</em>
    </div>
    """
    return fallback_html, None, image_list[evt.index]

def apply_metadata_to_t2i(meta_dict):
    if not meta_dict:
        return [gr.update()] * 9
    return (
        gr.update(value=meta_dict.get("prompt", "")),
        gr.update(value=meta_dict.get("negative_prompt", "")),
        gr.update(value=meta_dict.get("width", 1024)),
        gr.update(value=meta_dict.get("height", 1024)),
        gr.update(value=meta_dict.get("steps", 28)),
        gr.update(value=meta_dict.get("guidance_scale", 7.0)),
        gr.update(value=meta_dict.get("sampler", "Euler a")),
        gr.update(value=meta_dict.get("scheduler", "Normal")),
        gr.update(value=meta_dict.get("seed", 12345))
    )

def apply_remix_to_t2i(meta_dict):
    if not meta_dict:
        return [gr.update()] * 9
    new_seed = random.randint(0, 2**32 - 1)
    return (
        gr.update(value=meta_dict.get("prompt", "")),
        gr.update(value=meta_dict.get("negative_prompt", "")),
        gr.update(value=meta_dict.get("width", 1024)),
        gr.update(value=meta_dict.get("height", 1024)),
        gr.update(value=meta_dict.get("steps", 28)),
        gr.update(value=meta_dict.get("guidance_scale", 7.0)),
        gr.update(value=meta_dict.get("sampler", "Euler a")),
        gr.update(value=meta_dict.get("scheduler", "Normal")),
        gr.update(value=new_seed)
    )

# --- GENERATION WRAPPERS ---

def generate_text2img_ui(model_id, prompt, negative_prompt, width, height, steps, guidance_scale, sampler, scheduler, seed, randomize_seed, 
                         char_lora, char_scale, concept_lora, concept_scale, pose_lora, pose_scale, style_lora, style_scale,
                         upscale_method, upscale_factor, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        return None, "<div class='status-pill status-error'>⚠️ Prompt cannot be empty.</div>", seed
        
    try:
        model_info = model_manager.get_model_info(model_id)
        actual_seed = random.randint(0, 2**32 - 1) if randomize_seed else int(seed)
        
        active_loras = {}
        all_loras = lora_manager.scan_loras()
        
        def add_to_payload(name, weight):
            if name and name != "None" and name in all_loras:
                active_loras[all_loras[name]] = float(weight)
                
        add_to_payload(char_lora, char_scale)
        add_to_payload(concept_lora, concept_scale)
        add_to_payload(pose_lora, pose_scale)
        add_to_payload(style_lora, style_scale)
        
        def update_progress(current_step, total_steps):
            progress(current_step / total_steps, desc=f"Rendering {current_step}/{total_steps}")
            
        request = GenerationRequest(
            model=model_info, prompt=prompt, negative_prompt=negative_prompt if negative_prompt else None,
            width=int(width), height=int(height), steps=int(steps), guidance_scale=float(guidance_scale),
            sampler=sampler, scheduler=scheduler, seed=actual_seed, loras=active_loras,
            upscale_method=upscale_method, upscale_factor=float(upscale_factor),
            step_callback=update_progress
        )
        
        progress(0, desc="Initializing Pipeline...")
        result = coordinator.generate(request)
        status_html = f"<div class='status-pill status-success'>✓ Render Complete ({result.generation_time_ms / 1000:.2f}s) | Seed: {actual_seed}</div>"
        return result.image_path, status_html, actual_seed
    except Exception as e:
        return None, f"<div class='status-pill status-error'>✕ Error: {str(e)}</div>", seed

def generate_img2img_ui(model_id, prompt, negative_prompt, input_image_path, denoising_strength, width, height, steps, guidance_scale, sampler, scheduler, seed, randomize_seed, 
                        char_lora, char_scale, concept_lora, concept_scale, pose_lora, pose_scale, style_lora, style_scale,
                        upscale_method, upscale_factor,
                        controlnet_type="None", controlnet_img_path=None, controlnet_scale=0.8,
                        ip_adapter_name="None", ip_adapter_img_path=None, ip_adapter_scale=0.6,
                        pulid_enabled=False, pulid_img_path=None, pulid_scale=0.8,
                        progress=gr.Progress()):
    if not input_image_path:
        return None, "<div class='status-pill status-error'>⚠️ No input image provided.</div>", seed
        
    try:
        model_info = model_manager.get_model_info(model_id)
        actual_seed = random.randint(0, 2**32 - 1) if randomize_seed else int(seed)
        
        active_loras = {}
        all_loras = lora_manager.scan_loras()
        
        def add_to_payload(name, weight):
            if name and name != "None" and name in all_loras:
                active_loras[all_loras[name]] = float(weight)
                
        add_to_payload(char_lora, char_scale)
        add_to_payload(concept_lora, concept_scale)
        add_to_payload(pose_lora, pose_scale)
        add_to_payload(style_lora, style_scale)
        
        def update_progress(current_step, total_steps):
            progress(current_step / total_steps, desc=f"Rendering {current_step}/{total_steps}")
            
        init_img = Image.open(input_image_path).convert("RGB")
        init_img = init_img.resize((int(width), int(height)), Image.Resampling.LANCZOS)
        
        cnet_img = Image.open(controlnet_img_path).convert("RGB") if controlnet_img_path else None
        ip_img = Image.open(ip_adapter_img_path).convert("RGB") if ip_adapter_img_path else None
        pulid_img = Image.open(pulid_img_path).convert("RGB") if pulid_img_path else None

        request = GenerationRequest(
            model=model_info, prompt=prompt, negative_prompt=negative_prompt if negative_prompt else None,
            width=int(width), height=int(height), steps=int(steps), guidance_scale=float(guidance_scale),
            sampler=sampler, scheduler=scheduler, seed=actual_seed, loras=active_loras,
            upscale_method=upscale_method, upscale_factor=float(upscale_factor),
            step_callback=update_progress,
            init_image=init_img,
            denoising_strength=float(denoising_strength),
            controlnet_type=controlnet_type,
            controlnet_image=cnet_img,
            controlnet_scale=float(controlnet_scale),
            ip_adapter_name=ip_adapter_name,
            ip_adapter_image=ip_img,
            ip_adapter_scale=float(ip_adapter_scale),
            pulid_enabled=pulid_enabled,
            pulid_image=pulid_img,
            pulid_scale=float(pulid_scale),
        )
        
        progress(0, desc="Initializing Transformation Pipeline...")
        result = coordinator.generate(request)
        status_html = f"<div class='status-pill status-success'>✓ Transformation Complete ({result.generation_time_ms / 1000:.2f}s) | Seed: {actual_seed}</div>"
        return result.image_path, status_html, actual_seed
    except Exception as e:
        return None, f"<div class='status-pill status-error'>✕ Error: {str(e)}</div>", seed

def save_output_image(image_obj, target_folder, filename):
    if image_obj is None:
        return "<div class='status-pill status-error'>⚠️ No image available to save.</div>"
    try:
        dest_dir = Path(target_folder) if target_folder and target_folder.strip() else paths.IMAGES_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_filename = filename if filename and filename.strip() else f"restored_{int(time.time())}.png"
        final_path = dest_dir / final_filename
        
        if isinstance(image_obj, str):
            img = Image.open(image_obj)
            img.save(final_path)
        else:
            image_obj.save(final_path)
        return f"<div class='status-pill status-success'>✓ Saved to: {final_path}</div>"
    except Exception as e:
        return f"<div class='status-pill status-error'>✕ Save failed: {str(e)}</div>"

def process_watermark_removal(editor_data, progress=gr.Progress()):
    if not editor_data:
        return None, "<div class='status-pill status-error'>⚠️ No image or mask provided.</div>"
    bg_val = editor_data.get("background")
    bg_path = bg_val.get("path") if isinstance(bg_val, dict) else bg_val
    layers = editor_data.get("layers", [])
    if not bg_path:
        return None, "<div class='status-pill status-error'>⚠️ Missing background image.</div>"
    if not layers:
        return None, "<div class='status-pill status-error'>⚠️ Please paint a mask over the object to erase.</div>"
        
    try:
        progress(0.2, desc="Preparing inpaint mask...")
        base_img = Image.open(bg_path).convert("RGB")
        mask_val = layers[0]
        mask_path = mask_val.get("path") if isinstance(mask_val, dict) else mask_val
        mask_img = Image.open(mask_path).convert("RGBA")
        
        progress(0.5, desc="Running Inpainting...")
        repaired_img = restoration_engine.remove_watermark_cv2(base_img, mask_img)
        progress(1.0, desc="Object Erased!")
        return repaired_img, "<div class='status-pill status-success'>✓ Object erased successfully!</div>"
    except Exception as e:
        return None, f"<div class='status-pill status-error'>✕ Error: {str(e)}</div>"

def process_bg_removal(editor_data, progress=gr.Progress()):
    if not editor_data:
        return None, "<div class='status-pill status-error'>⚠️ No image provided.</div>"
    bg_val = editor_data.get("background")
    bg_path = bg_val.get("path") if isinstance(bg_val, dict) else bg_val
    if not bg_path:
        return None, "<div class='status-pill status-error'>⚠️ Missing base image.</div>"
        
    try:
        progress(0.3, desc="Initializing U-2-Net...")
        img = Image.open(bg_path).convert("RGB")
        progress(0.6, desc="Extracting foreground...")
        transparent_png = restoration_engine.remove_background(img)
        progress(1.0, desc="Background Removed!")
        return transparent_png, "<div class='status-pill status-success'>✓ Background removed successfully!</div>"
    except Exception as e:
        return None, f"<div class='status-pill status-error'>✕ Error: {str(e)}</div>"

def apply_image_enhancements(editor_data, brightness, contrast, saturation, sharpness, progress=gr.Progress()):
    if not editor_data:
        return None, "<div class='status-pill status-error'>⚠️ No image provided.</div>"
    bg_val = editor_data.get("background")
    bg_path = bg_val.get("path") if isinstance(bg_val, dict) else bg_val
    if not bg_path:
        return None, "<div class='status-pill status-error'>⚠️ No base image found.</div>"
        
    try:
        progress(0.5, desc="Applying color adjustments...")
        img = Image.open(bg_path).convert("RGB")
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        if saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(saturation)
        if sharpness != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(sharpness)
        progress(1.0, desc="Adjustments Applied!")
        return img, "<div class='status-pill status-success'>✓ Adjustments applied!</div>"
    except Exception as e:
        return None, f"<div class='status-pill status-error'>✕ Error: {str(e)}</div>"

def run_auto_caption(editor_data, progress=gr.Progress()):
    if not editor_data:
        return "⚠️ No image loaded."
    bg_val = editor_data.get("background")
    bg_path = bg_val.get("path") if isinstance(bg_val, dict) else bg_val
    if not bg_path:
        return "⚠️ No image loaded."
    try:
        progress(0.5, desc="Analyzing image...")
        img = Image.open(bg_path).convert("RGB")
        caption = supir_engine.auto_caption(img)
        progress(1.0, desc="Caption Generated!")
        return caption
    except Exception as e:
        return f"✕ Auto-caption failed: {str(e)}"

def run_supir_enhancement(editor_data, prompt, neg_prompt, upscale_factor, model_name, progress=gr.Progress()):
    if not editor_data:
        return None, "<div class='status-pill status-error'>⚠️ No image provided.</div>"
    bg_val = editor_data.get("background")
    bg_path = bg_val.get("path") if isinstance(bg_val, dict) else bg_val
    if not bg_path:
        return None, "<div class='status-pill status-error'>⚠️ No base image provided.</div>"
    if not prompt or prompt.strip() == "":
        return None, "<div class='status-pill status-error'>⚠️ Please provide an enhancement prompt.</div>"
        
    try:
        progress(0.1, desc="Preparing Upscaler...")
        img = Image.open(bg_path).convert("RGB")
        progress(0.3, desc="Running AI Enhancement...")
        restored_img = supir_engine.enhance_image(img, prompt, neg_prompt, float(upscale_factor), model_name)
        progress(1.0, desc="Enhancement Complete!")
        return restored_img, "<div class='status-pill status-success'>✓ Image enhanced successfully!</div>"
    except Exception as e:
        return None, f"<div class='status-pill status-error'>✕ Error during enhancement: {str(e)}</div>"

def run_video_generation(input_image, prompt, neg_prompt, aspect_ratio, num_steps, num_frames, guidance_scale, fps, ip_scale, progress=gr.Progress()):
    if animation_engine is None:
        return None, "<div class='status-pill status-error'>⚠️ Animation engine (CogVideo) is not installed or available.</div>"

    if input_image is None:
        return None, "<div class='status-pill status-error'>⚠️ Please upload a source image.</div>"
        
    if not prompt or prompt.strip() == "":
        prompt = "smooth motion, high quality, cinematic"

    try:
        progress(0.1, desc="Initializing Animation Engine...")
        output_mp4_path = animation_engine.generate_video(
            image=input_image,
            prompt=prompt,
            negative_prompt=neg_prompt,
            aspect_ratio=aspect_ratio,
            num_inference_steps=int(num_steps),
            num_frames=int(num_frames),
            guidance_scale=float(guidance_scale),
            fps=int(fps),
            ip_scale=float(ip_scale)
        )
        progress(1.0, desc="Video generation complete!")
        return output_mp4_path, "<div class='status-pill status-success'>✓ Animation generated successfully!</div>"
    except Exception as e:
        return None, f"<div class='status-pill status-error'>✕ Error during generation: {str(e)}</div>"

# --- CYBER-INDIGO STYLING SYSTEM ---

CYBER_INDIGO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* --- BASE & BACKGROUND --- */
html, body, .gradio-container {
    background-color: #0A0E17 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #F8FAFC !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 10px 18px !important;
}

/* --- HEADER BAR --- */
#header-row {
    align-items: center;
    margin-bottom: 10px;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 8px;
}

/* --- HIERARCHY LEVEL 2: SETTINGS PANEL --- */
#studio-sidebar {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 16px !important;
    padding: 16px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45) !important;
}

/* --- HIERARCHY LEVEL 1: MAIN WORKSPACE --- */
#studio-workspace {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45) !important;
}

.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #94A3B8;
    margin-top: 12px;
    margin-bottom: 6px;
    text-transform: uppercase;
}

/* --- ASPECT RATIO RADIO PILLS --- */
.ar-radio-group {
    background: #0B0F19 !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    padding: 4px !important;
    margin-bottom: 10px !important;
}

.ar-radio-group .wrap {
    display: flex !important;
    flex-direction: row !important;
    gap: 4px !important;
}

.ar-radio-group label {
    flex: 1 1 0 !important;
    text-align: center !important;
    padding: 6px 2px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    background: transparent !important;
    color: #94A3B8 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

.ar-radio-group label.selected, .ar-radio-group label:has(input:checked) {
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 10px rgba(79, 70, 229, 0.4) !important;
}

/* --- HIERARCHY LEVEL 3: INPUTS, SLIDERS, DROPDOWNS --- */
.gr-form, .gr-input, .gr-box, fieldset {
    background: #0B0F19 !important;
    border-color: #1E293B !important;
    border-radius: 10px !important;
}

input, textarea, select {
    background: #0B0F19 !important;
    color: #F8FAFC !important;
    border-color: #1E293B !important;
}

.prompt-box textarea {
    background: #0B0F19 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}

.prompt-box textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}

.token-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #94A3B8;
    background: #0B0F19;
    border: 1px solid #1E293B;
    padding: 2px 8px;
    border-radius: 6px;
}

/* --- UTILITY BUTTON PILLS (Enhance, Randomize, Clear) --- */
.util-btn-row {
    display: flex !important;
    gap: 8px !important;
    margin: 8px 0 12px 0 !important;
}

.util-btn, .util-btn button {
    height: 34px !important;
    min-height: 34px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    padding: 0 14px !important;
    transition: all 0.15s ease !important;
}

.util-btn:hover, .util-btn button:hover {
    background: #334155 !important;
    border-color: #6366F1 !important;
    color: #FFFFFF !important;
}

.icon-btn {
    height: 40px !important;
    min-height: 40px !important;
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
}

.icon-btn:hover {
    color: #EF4444 !important;
    border-color: #EF4444 !important;
}

/* --- VIBRANT CYBER ACTION BUTTONS --- */
.action-btn-row {
    margin-top: 14px !important;
}

#btn-gen-t2i, #btn-gen-t2i button {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #06B6D4 100%) !important;
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px !important;
    height: 50px !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.45) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

#btn-gen-t2i:hover, #btn-gen-t2i button:hover {
    background: linear-gradient(135deg, #6366F1 0%, #818CF8 50%, #22D3EE 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 26px rgba(99, 102, 241, 0.65) !important;
}

#btn-gen-i2i, #btn-gen-i2i button {
    background: linear-gradient(135deg, #06B6D4 0%, #0EA5E9 50%, #3B82F6 100%) !important;
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    height: 50px !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(6, 182, 212, 0.45) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

#btn-gen-i2i:hover, #btn-gen-i2i button:hover {
    background: linear-gradient(135deg, #22D3EE 0%, #38BDF8 50%, #60A5FA 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 26px rgba(6, 182, 212, 0.65) !important;
}

#btn-gen-restore, #btn-gen-restore button {
    background: linear-gradient(135deg, #A855F7 0%, #C084FC 50%, #EC4899 100%) !important;
    color: #FFFFFF !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    height: 46px !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 18px rgba(168, 85, 247, 0.45) !important;
}

#btn-gen-video, #btn-gen-video button {
    background: linear-gradient(135deg, #F59E0B 0%, #F97316 50%, #E11D48 100%) !important;
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    height: 50px !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(245, 158, 11, 0.45) !important;
}

#btn-stop, #btn-stop button {
    background: #1E1218 !important;
    color: #F87171 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    border: 1px solid #7F1D1D !important;
    border-radius: 12px !important;
    height: 50px !important;
}

.sub-action-btn, .sub-action-btn button {
    font-size: 12px !important;
    font-weight: 600 !important;
    height: 36px !important;
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    margin-top: 6px !important;
}

.sub-action-btn:hover, .sub-action-btn button:hover {
    background: #334155 !important;
    border-color: #6366F1 !important;
    color: #6366F1 !important;
}

/* --- OUTPUT CANVAS --- */
.studio-canvas {
    border-radius: 14px !important;
    border: 1px solid #1E293B !important;
    background: #0B0F19 !important;
    overflow: hidden !important;
}

.status-pill {
    background: #0B0F19;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 6px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #94A3B8;
    margin-top: 6px;
}

.status-success {
    color: #34D399;
    border-color: rgba(52, 211, 153, 0.3);
}

.status-error {
    color: #F87171;
    border-color: rgba(248, 113, 113, 0.3);
}

/* --- TAB THEMED BADGES --- */
.tab-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
}

.badge-t2i {
    background: rgba(99, 102, 241, 0.15);
    color: #818CF8;
    border: 1px solid rgba(99, 102, 241, 0.35);
}

.badge-i2i {
    background: rgba(6, 182, 212, 0.15);
    color: #22D3EE;
    border: 1px solid rgba(6, 182, 212, 0.35);
}

.badge-gal {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.badge-restore {
    background: rgba(236, 72, 153, 0.15);
    color: #F472B6;
    border: 1px solid rgba(236, 72, 153, 0.35);
}

.badge-video {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

/* --- TAB NAVIGATION BAR --- */
.tab-nav {
    border-bottom: 1px solid #1E293B !important;
    margin-bottom: 14px !important;
    gap: 8px !important;
}

.tab-nav button {
    font-size: 13px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 18px !important;
    color: #94A3B8 !important;
    transition: all 0.15s ease !important;
}

.tab-nav button:hover {
    color: #F8FAFC !important;
    background: #1E293B !important;
}

.tab-nav button.selected {
    color: #818CF8 !important;
    border-bottom: 2px solid #818CF8 !important;
    background: rgba(99, 102, 241, 0.12) !important;
}

/* ==========================================================
   STRICT SINGLE PROGRESS BAR ONLY:
   Completely suppress progress wrappers on all components
   except the primary output canvas.
   ========================================================== */
#studio-sidebar .progress-level,
#studio-sidebar .progress-bar-wrap,
#studio-sidebar .meta-text,
#studio-sidebar .meta-text-center,
.hide-progress .progress-level,
.hide-progress .progress-bar-wrap,
.hide-progress .meta-text,
.hide-progress .meta-text-center {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    max-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    pointer-events: none !important;
}

/* --- BOTTOM STATUS BAR --- */
#status-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #111827;
    border-top: 1px solid #1E293B;
    padding: 4px 20px;
    font-size: 11px;
    color: #94A3B8;
    z-index: 999;
    font-family: 'JetBrains Mono', monospace;
}
"""

# --- MAIN UI CREATOR ---

def create_ui():
    if hasattr(model_manager, "available_models"):
        available_models = list(model_manager.available_models.keys())
    elif hasattr(model_manager, "scan_models"):
        available_models = list(model_manager.scan_models().keys())
    else:
        available_models = []

    default_model = available_models[0] if available_models else None
    cat_loras = get_categorized_loras()
    cat_cnets = ["None"] + sorted(list(adapter_manager.scan_controlnets().keys()))
    cat_ip_adapters = ["None"] + sorted(list(adapter_manager.scan_ip_adapters().keys()))

    with gr.Blocks(title="ImageStudio") as app:
        # Direct inline style injection to guarantee instant application in all Gradio versions
        gr.HTML(f"<style>{CYBER_INDIGO_CSS}</style>", visible=False)
        
        gallery_state = gr.State([]) 
        meta_state = gr.State(None)
        selected_image_path = gr.State(None)
        
        # --- HEADER ROW ---
        with gr.Row(elem_id="header-row"):
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 24px; color: #818CF8;">⚡</span>
                    <div>
                        <h1 style="margin: 0; font-size: 19px; font-weight: 800; letter-spacing: -0.5px; color: #FFFFFF; line-height: 1.1;">ImageStudio</h1>
                        <span style="font-size: 11px; color: #94A3B8; font-weight: 500;">Generative AI & Transformation Suite</span>
                    </div>
                </div>
                """)
            with gr.Column(scale=2):
                sys_info = gr.HTML(value=get_hardware_stats_html())
                refresh_timer = gr.Timer(2)
            
        with gr.Row(elem_id="studio-layout"):
            
            # --- SIDEBAR CONTROL PANEL (LEVEL 2 HIERARCHY CARD) ---
            with gr.Column(scale=1, min_width=300, elem_id="studio-sidebar") as sidebar_col:
                
                # MODEL SECTION
                gr.HTML("<div class='section-label'>📦 ACTIVE CHECKPOINT</div>")
                with gr.Row():
                    model_selector = gr.Dropdown(choices=available_models, value=default_model, interactive=True, show_label=False, scale=4)
                    unload_btn = gr.Button("🧹", variant="secondary", scale=1, min_width=38, elem_classes=["icon-btn"])
                
                # GENERATION SECTION
                gr.HTML("<div class='section-label'>📐 ASPECT RATIO</div>")
                ar_radio = gr.Radio(
                    choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                    value="1:1",
                    show_label=False,
                    elem_classes=["ar-radio-group"]
                )
                
                width = gr.Slider(minimum=512, maximum=1536, step=64, value=1024, label="Width (px)")
                height = gr.Slider(minimum=512, maximum=1536, step=64, value=1024, label="Height (px)")
                
                gr.HTML("<div class='section-label'>🎛️ SAMPLING PARAMETERS</div>")
                steps = gr.Slider(minimum=1, maximum=60, step=1, value=28, label="Sampling Steps")
                guidance_scale = gr.Slider(minimum=0.0, maximum=15.0, step=0.1, value=7.0, label="CFG Guidance Scale")
                
                with gr.Row():
                    sampler = gr.Dropdown(label="Sampler", choices=AVAILABLE_SAMPLERS, value="Euler a", interactive=True, scale=1)
                    scheduler = gr.Dropdown(label="Schedule", choices=AVAILABLE_SCHEDULERS, value="Normal", interactive=True, scale=1)
                
                # SEED SECTION
                with gr.Row():
                    seed = gr.Number(label="Seed", value=12345, precision=0, scale=3, elem_id="seed_num", elem_classes=["hide-progress"])
                    randomize_seed = gr.Checkbox(label="🎲 Randomize", value=True, scale=2)
                
                # ADVANCED ACCORDIONS
                with gr.Accordion("🎨 LoRA Loadout (4 Slots)", open=False):
                    char_lora = gr.Dropdown(label="Character", choices=cat_loras["characters"], value="None")
                    char_scale = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="Character Weight")
                    concept_lora = gr.Dropdown(label="Concept", choices=cat_loras["concepts"], value="None")
                    concept_scale = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="Concept Weight")
                    pose_lora = gr.Dropdown(label="Pose", choices=cat_loras["poses"], value="None")
                    pose_scale = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="Pose Weight")
                    style_lora = gr.Dropdown(label="Style", choices=cat_loras["styles"], value="None")
                    style_scale = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="Style Weight")

                with gr.Accordion("🔍 Post-Processing Upscaler", open=False):
                    upscale_method = gr.Dropdown(label="Upscaler", choices=["None", "4x-RealCUGAN", "Tiled SDXL", "Lanczos", "Bicubic"], value="None")
                    upscale_factor = gr.Slider(minimum=1.0, maximum=4.0, step=0.5, value=2.0, label="Scale Factor")

            # --- MAIN WORKSPACE DECK (LEVEL 1 HIERARCHY) ---
            with gr.Column(scale=4, elem_id="studio-workspace"):
                with gr.Tabs(elem_id="main-tabs") as tabs:
                    
                    # ==========================================
                    # TAB 1: ⚡ TEXT STUDIO (CYBER INDIGO THEME)
                    # ==========================================
                    with gr.Tab("⚡ Text Studio", id="tab_t2i", elem_id="tab-t2i"):
                        with gr.Row():
                            with gr.Column(scale=6, elem_id="t2i-left-pane"):
                                gr.HTML("<div class='tab-badge badge-t2i'>⚡ GENERATIVE TEXT STUDIO</div>")
                                
                                with gr.Row():
                                    gr.HTML("<span style='font-size: 12px; font-weight: 700; color: #F8FAFC;'>PROMPT ✨</span>")
                                    token_counter_t2i = gr.HTML("<span class='token-pill'>~0 tokens (0 chars)</span>")
                                
                                prompt_t2i = gr.Textbox(show_label=False, lines=3, placeholder="Describe your masterpiece in detail...", elem_classes=["prompt-box"])
                                
                                with gr.Row(elem_classes=["util-btn-row"]):
                                    enhance_btn_t2i = gr.Button("✨ AI Enhance", elem_classes=["util-btn"])
                                    rand_prompt_btn_t2i = gr.Button("🎲 Randomize", elem_classes=["util-btn"])
                                    clear_prompt_btn_t2i = gr.Button("🧹 Clear", elem_classes=["util-btn"])
                                
                                with gr.Accordion("Negative Prompt (Optional)", open=False):
                                    neg_prompt_t2i = gr.Textbox(show_label=False, lines=2, value="blurry, low quality, distorted, bad anatomy, artifacts")
                                
                                with gr.Row(elem_classes=["action-btn-row"]):
                                    gen_btn_t2i = gr.Button("⚡ Generate Image", elem_id="btn-gen-t2i", scale=4)
                                    stop_btn_t2i = gr.Button("🛑 Stop", elem_id="btn-stop", scale=1)

                            with gr.Column(scale=6, elem_id="t2i-right-pane"):
                                out_image_t2i = gr.Image(label="Output Canvas", type="filepath", interactive=False, height=440, elem_id="out-image-t2i", elem_classes=["studio-canvas"])
                                status_t2i = gr.HTML("<div class='status-pill'>Ready</div>", elem_id="status_t2i")
                                
                                with gr.Row():
                                    send_t2i_to_i2i_btn = gr.Button("↗️ Send to Img2Img Lab", elem_classes=["sub-action-btn"])

                    # ==========================================
                    # TAB 2: 🧪 ADAPTER LAB (CYAN THEME)
                    # ==========================================
                    with gr.Tab("🧪 Adapter Lab", id="tab_i2i", elem_id="tab-i2i"):
                        with gr.Row():
                            with gr.Column(scale=6, elem_id="i2i-left-pane"):
                                gr.HTML("<div class='tab-badge badge-i2i'>🧪 MULTI-ADAPTER TRANSFORMATION LAB</div>")
                                
                                with gr.Row():
                                    input_img_i2i = gr.Image(label="Source Image", type="filepath", interactive=True, height=180, scale=1)
                                    with gr.Column(scale=1):
                                        denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.65, label="Denoising Strength")
                                        token_counter_i2i = gr.HTML("<span class='token-pill'>~0 tokens</span>")
                                
                                prompt_i2i = gr.Textbox(show_label=False, lines=2, placeholder="Transformation prompt or target style...", elem_classes=["prompt-box"])
                                
                                with gr.Row(elem_classes=["util-btn-row"]):
                                    enhance_btn_i2i = gr.Button("✨ AI Enhance", elem_classes=["util-btn"])
                                    clear_prompt_btn_i2i = gr.Button("🧹 Clear", elem_classes=["util-btn"])
                                
                                with gr.Accordion("Negative Prompt", open=False):
                                    neg_prompt_i2i = gr.Textbox(show_label=False, lines=1, value="blurry, low quality, distorted, artifacts")
                                
                                with gr.Accordion("🎛️ ControlNet, IP-Adapter & Face ID", open=False):
                                    with gr.Row():
                                        cnet_type_ui = gr.Dropdown(label="ControlNet Mode", choices=cat_cnets, value="None", scale=2)
                                        cnet_scale_ui = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=0.8, label="Weight", scale=1)
                                    cnet_img_ui = gr.Image(label="Control Guide (Optional)", type="filepath", height=100)
                                    
                                    with gr.Row():
                                        ip_adapter_ui = gr.Dropdown(label="IP-Adapter Model", choices=cat_ip_adapters, value="None", scale=2)
                                        ip_scale_ui = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=0.6, label="Weight", scale=1)
                                    ip_img_ui = gr.Image(label="Style Image", type="filepath", height=100)
                                    
                                    with gr.Row():
                                        pulid_enable_ui = gr.Checkbox(label="Enable PuLID FaceID", value=False)
                                        pulid_scale_ui = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=0.8, label="Weight")
                                    pulid_img_ui = gr.Image(label="Face Photo", type="filepath", height=100)

                                with gr.Row(elem_classes=["action-btn-row"]):
                                    gen_btn_i2i = gr.Button("🧪 Generate Transformation", elem_id="btn-gen-i2i", scale=4)
                                    stop_btn_i2i = gr.Button("🛑 Stop", elem_id="btn-stop", scale=1)

                            with gr.Column(scale=6, elem_id="i2i-right-pane"):
                                out_image_i2i = gr.Image(label="Transformed Canvas", type="filepath", interactive=False, height=440, elem_id="out-image-i2i", elem_classes=["studio-canvas"])
                                status_i2i = gr.HTML("<div class='status-pill'>Ready</div>", elem_id="status_i2i")

                    # ==========================================
                    # TAB 3: ▦ CREATIVE VAULT (EMERALD THEME)
                    # ==========================================
                    with gr.Tab("▦ Creative Vault", id="tab_gal", elem_id="tab-gal"):
                        with gr.Row():
                            with gr.Column(scale=7, elem_id="gal-left-pane"):
                                with gr.Row():
                                    gr.HTML("<div class='tab-badge badge-gal'>▦ EXHIBITION & METADATA VAULT</div>")
                                    refresh_gal_btn = gr.Button("🔄 Refresh", elem_classes=["util-btn"])
                                gallery = gr.Gallery(label="Output History", show_label=False, elem_id="gallery", columns=[4], rows=[3], height=450, object_fit="contain")
                            
                            with gr.Column(scale=5, elem_id="gal-right-pane"):
                                gr.HTML("<div class='tab-badge badge-gal'>🔍 LIGHTBOX INSPECTOR</div>")
                                meta_display_html = gr.HTML(value="<div style='color: #94A3B8; padding: 12px;'>Select an image from the gallery to inspect parameters.</div>")
                                
                                gr.HTML("<div style='font-size: 12px; font-weight: 700; color: #F8FAFC; margin-top: 10px; margin-bottom: 6px;'>REUSE & REMIX</div>")
                                with gr.Row():
                                    send_to_gen_btn = gr.Button("↖️ Send to Text2Img", elem_classes=["sub-action-btn"])
                                    send_to_i2i_btn = gr.Button("↖️ Send to Img2Img", elem_classes=["sub-action-btn"])
                                with gr.Row():
                                    remix_btn = gr.Button("🔀 Remix (New Seed)", elem_classes=["sub-action-btn"])
                                    duplicate_btn = gr.Button("📋 Duplicate (Same Seed)", elem_classes=["sub-action-btn"])

                    # ==========================================
                    # TAB 4: 🪄 ALCHEMY (PURPLE THEME)
                    # ==========================================
                    with gr.Tab("🪄 Alchemy", id="tab_restore", elem_id="tab-restore"):
                        with gr.Row():
                            with gr.Column(scale=6, elem_id="restore-left-pane"):
                                gr.HTML("<div class='tab-badge badge-restore'>🪄 ENHANCEMENT & RESTORATION LAB</div>")
                                restore_editor = gr.ImageEditor(
                                    label="Upload Image & Paint Mask",
                                    type="filepath", 
                                    brush=gr.Brush(colors=["#FFFFFF"], default_size=25),
                                    eraser=gr.Eraser(default_size=25),
                                    layers=False,
                                    height=260
                                )
                                
                                with gr.Row(elem_classes=["util-btn-row"]):
                                    erase_btn = gr.Button("🖌️ Erase Object", elem_classes=["util-btn"])
                                    rembg_btn = gr.Button("✂️ Remove BG", elem_classes=["util-btn"])

                                with gr.Accordion("Lighting Adjustments & SUPIR Upscale", open=False):
                                    with gr.Row():
                                        bright_slider = gr.Slider(minimum=0.2, maximum=2.0, step=0.05, value=1.0, label="Bright")
                                        contrast_slider = gr.Slider(minimum=0.2, maximum=2.0, step=0.05, value=1.0, label="Contrast")
                                        sat_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=1.0, label="Sat")
                                        sharp_slider = gr.Slider(minimum=0.0, maximum=3.0, step=0.1, value=1.0, label="Sharp")
                                    apply_adj_btn = gr.Button("✨ Apply Adjustments", elem_classes=["util-btn"])

                                    gr.Markdown("---")
                                    with gr.Row():
                                        supir_prompt = gr.Textbox(label="Prompt", lines=1, placeholder="Describe desired textures...")
                                        supir_neg_prompt = gr.Textbox(label="Neg", lines=1, value="blurry, distorted")
                                    with gr.Row():
                                        auto_caption_btn = gr.Button("🔍 Auto-Caption", elem_classes=["util-btn"])
                                        supir_checkpoint = gr.Dropdown(label="Model", choices=get_upscale_models(), value=get_upscale_models()[0] if get_upscale_models() else None)
                                        supir_upscale_factor = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Scale")
                                    
                                    supir_enhance_btn = gr.Button("✨ Run SUPIR Enhancer", elem_id="btn-gen-restore")

                            with gr.Column(scale=6, elem_id="restore-right-pane"):
                                restore_output = gr.Image(label="Processed Canvas", type="pil", interactive=False, height=400, elem_classes=["studio-canvas"])
                                restore_status = gr.HTML("<div class='status-pill'>Ready</div>", elem_id="restore_status")
                                
                                with gr.Row():
                                    send_restore_to_i2i = gr.Button("↖️ Send to Img2Img", elem_classes=["sub-action-btn"])
                                
                                with gr.Row():
                                    custom_folder = gr.Textbox(label="Folder", placeholder=r"e.g., D:\AI\Outputs", scale=2)
                                    custom_filename = gr.Textbox(label="Filename", value="restored_output.png", scale=2)
                                    save_file_btn = gr.Button("💾 Save", scale=1, elem_classes=["sub-action-btn"])
                                save_file_status = gr.HTML("<div class='status-pill'>Ready</div>", elem_id="save_file_status")

                    # ==========================================
                    # TAB 5: 🎬 MOTION DECK (AMBER THEME)
                    # ==========================================
                    with gr.Tab("🎬 Motion Deck", id="tab_video", elem_id="tab-video"):
                        with gr.Row():
                            with gr.Column(scale=6, elem_id="video-left-pane"):
                                gr.HTML("<div class='tab-badge badge-video'>🎬 CINEMATIC MOTION & VIDEO DECK</div>")
                                anim_input_img = gr.Image(type="pil", label="Source Character / Frame", height=180)
                                anim_prompt = gr.Textbox(
                                    show_label=False, 
                                    placeholder="e.g., subtle head turn, hair blowing in wind, blinking eyes, cinematic lighting",
                                    lines=2,
                                    elem_classes=["prompt-box"])
                                
                                with gr.Row(elem_classes=["util-btn-row"]):
                                    anim_preset_fast = gr.Button("⚡ Fast (20s, 25f ~2m)", elem_classes=["util-btn"])
                                    anim_preset_hq = gr.Button("💎 High Quality (35s, 49f)", elem_classes=["util-btn"])
                                
                                anim_ar = gr.Radio(
                                    choices=["9:16 (Portrait / Reels / Shorts)", "16:9 (Landscape)", "3:2 (Landscape - Native)", "2:3 (Portrait)", "1:1 (Square)"],
                                    value="9:16 (Portrait / Reels / Shorts)",
                                    label="Aspect Ratio"
                                )
                                
                                with gr.Row():
                                    anim_steps = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Steps", scale=1)
                                    anim_frames = gr.Dropdown(choices=[17, 25, 33, 41, 49], value=25, label="Frames", scale=1)
                                
                                with gr.Row():
                                    anim_guidance = gr.Slider(minimum=2.0, maximum=10.0, value=6.0, step=0.5, label="CFG", scale=1)
                                    anim_fps = gr.Slider(minimum=6, maximum=24, value=8, step=1, label="FPS", scale=1)
                                
                                with gr.Accordion("Negative Prompt", open=False):
                                    anim_neg_prompt = gr.Textbox(
                                        show_label=False, 
                                        value="static, blurry, jittery, distorted, morphing, low quality, artifacts",
                                        lines=1)
                                
                                anim_ip_scale = gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.05, visible=False)
                                anim_generate_btn = gr.Button("🎬 GENERATE VIDEO", elem_id="btn-gen-video")
                            
                            with gr.Column(scale=6, elem_id="video-right-pane"):
                                anim_output_video = gr.Video(label="Rendered Animation (.mp4)", height=450)
                                anim_status = gr.HTML("<div class='status-pill'>Ready</div>", elem_id="anim_status")

        # --- BOTTOM STATUS BAR ---
        status_bar_component = gr.HTML(value=get_bottom_status_html(default_model))

        # --- EVENT BINDINGS (STRICT SINGLE PROGRESS BAR) ---
        
        # Hardware & VRAM
        app.load(fn=get_hardware_stats_html, inputs=None, outputs=sys_info)
        refresh_timer.tick(fn=get_hardware_stats_html, inputs=None, outputs=sys_info)
        unload_btn.click(fn=force_unload_vram, inputs=None, outputs=[sys_info])
        
        # Settings & Model Updates
        model_selector.change(fn=on_model_change, inputs=[model_selector], outputs=[steps, guidance_scale])
        model_selector.change(fn=get_bottom_status_html, inputs=[model_selector], outputs=[status_bar_component])
        ar_radio.change(fn=set_aspect_ratio_radio, inputs=[ar_radio], outputs=[width, height])
        
        # Prompt Utilities
        prompt_t2i.change(fn=count_tokens_text, inputs=[prompt_t2i], outputs=[token_counter_t2i])
        prompt_i2i.change(fn=count_tokens_text, inputs=[prompt_i2i], outputs=[token_counter_i2i])
        enhance_btn_t2i.click(fn=enhance_prompt_text, inputs=[prompt_t2i, model_selector], outputs=[prompt_t2i])
        enhance_btn_i2i.click(fn=enhance_prompt_text, inputs=[prompt_i2i, model_selector], outputs=[prompt_i2i])
        rand_prompt_btn_t2i.click(fn=get_random_prompt_text, inputs=[model_selector], outputs=[prompt_t2i])
        clear_prompt_btn_t2i.click(fn=lambda: "", inputs=None, outputs=[prompt_t2i])
        clear_prompt_btn_i2i.click(fn=lambda: "", inputs=None, outputs=[prompt_i2i])
        
        # Generation Events (Strict Single Progress Bar)
        gen_event_t2i = gen_btn_t2i.click(
            fn=generate_text2img_ui,
            inputs=[model_selector, prompt_t2i, neg_prompt_t2i, width, height, steps, guidance_scale, sampler, scheduler, seed, randomize_seed, char_lora, char_scale, concept_lora, concept_scale, pose_lora, pose_scale, style_lora, style_scale, upscale_method, upscale_factor],
            outputs=[out_image_t2i, status_t2i, seed],
            show_progress="minimal"
        )
        stop_btn_t2i.click(
            fn=lambda: "<div class='status-pill'>⏹️ Generation Stopped</div>",
            inputs=None,
            outputs=[status_t2i],
            cancels=[gen_event_t2i],
            show_progress="hidden"
        )
        
        send_t2i_to_i2i_btn.click(
            fn=lambda img: img, inputs=[out_image_t2i], outputs=[input_img_i2i]
        ).then(fn=lambda: gr.Tabs(selected="tab_i2i"), outputs=tabs)

        gen_event_i2i = gen_btn_i2i.click(
            fn=generate_img2img_ui,
            inputs=[model_selector, prompt_i2i, neg_prompt_i2i, input_img_i2i, denoising_strength, width, height, steps, guidance_scale, sampler, scheduler, seed, randomize_seed, char_lora, char_scale, concept_lora, concept_scale, pose_lora, pose_scale, style_lora, style_scale, upscale_method, upscale_factor, cnet_type_ui, cnet_img_ui, cnet_scale_ui, ip_adapter_ui, ip_img_ui, ip_scale_ui, pulid_enable_ui, pulid_img_ui, pulid_scale_ui],
            outputs=[out_image_i2i, status_i2i, seed],
            show_progress="minimal"
        )
        stop_btn_i2i.click(
            fn=lambda: "<div class='status-pill'>⏹️ Generation Stopped</div>",
            inputs=None,
            outputs=[status_i2i],
            cancels=[gen_event_i2i],
            show_progress="hidden"
        )
        
        # Gallery & Lightbox Inspector
        refresh_gal_btn.click(fn=get_all_images, inputs=None, outputs=[gallery, gallery_state], show_progress="minimal")
        app.load(fn=get_all_images, inputs=None, outputs=[gallery, gallery_state])
        gallery.select(fn=inspect_image_formatted, inputs=[gallery_state], outputs=[meta_display_html, meta_state, selected_image_path])
        
        # Reuse Workflows
        send_to_gen_btn.click(
            fn=apply_metadata_to_t2i,
            inputs=[meta_state],
            outputs=[prompt_t2i, neg_prompt_t2i, width, height, steps, guidance_scale, sampler, scheduler, seed]
        ).then(fn=lambda: gr.Tabs(selected="tab_t2i"), outputs=tabs)
        
        remix_btn.click(
            fn=apply_remix_to_t2i,
            inputs=[meta_state],
            outputs=[prompt_t2i, neg_prompt_t2i, width, height, steps, guidance_scale, sampler, scheduler, seed]
        ).then(fn=lambda: gr.Tabs(selected="tab_t2i"), outputs=tabs)

        duplicate_btn.click(
            fn=apply_metadata_to_t2i,
            inputs=[meta_state],
            outputs=[prompt_t2i, neg_prompt_t2i, width, height, steps, guidance_scale, sampler, scheduler, seed]
        ).then(fn=lambda: gr.Tabs(selected="tab_t2i"), outputs=tabs)
        
        send_to_i2i_btn.click(
            fn=lambda p: p, 
            inputs=[selected_image_path], 
            outputs=[input_img_i2i]
        ).then(fn=lambda: gr.Tabs(selected="tab_i2i"), outputs=tabs)

        # Restoration Event Bindings
        erase_btn.click(
            fn=process_watermark_removal, inputs=[restore_editor], outputs=[restore_output, restore_status], show_progress="minimal"
        )
        
        rembg_btn.click(
            fn=process_bg_removal, inputs=[restore_editor], outputs=[restore_output, restore_status], show_progress="minimal"
        )
        
        apply_adj_btn.click(
            fn=apply_image_enhancements,
            inputs=[restore_editor, bright_slider, contrast_slider, sat_slider, sharp_slider],
            outputs=[restore_output, restore_status],
            show_progress="minimal"
        )
        
        # SUPIR Enhancer Event Bindings
        auto_caption_btn.click(
            fn=run_auto_caption, inputs=[restore_editor], outputs=[supir_prompt], show_progress="minimal"
        )
        
        supir_enhance_btn.click(
            fn=run_supir_enhancement,
            inputs=[restore_editor, supir_prompt, supir_neg_prompt, supir_upscale_factor, supir_checkpoint],
            outputs=[restore_output, restore_status],
            show_progress="minimal"
        )
        
        send_restore_to_i2i.click(
            fn=lambda img: img, inputs=[restore_output], outputs=[input_img_i2i]
        ).then(fn=lambda: gr.Tabs(selected="tab_i2i"), outputs=tabs)

        save_file_btn.click(
            fn=save_output_image, inputs=[restore_output, custom_folder, custom_filename], outputs=[save_file_status], show_progress="minimal"
        )

        anim_preset_fast.click(
            fn=lambda: (20, 25, 6.0, 8),
            inputs=None,
            outputs=[anim_steps, anim_frames, anim_guidance, anim_fps]
        )
        anim_preset_hq.click(
            fn=lambda: (35, 49, 6.0, 8),
            inputs=None,
            outputs=[anim_steps, anim_frames, anim_guidance, anim_fps]
        )

        anim_generate_btn.click(
            fn=run_video_generation,
            inputs=[anim_input_img, anim_prompt, anim_neg_prompt, anim_ar, anim_steps, anim_frames, anim_guidance, anim_fps, anim_ip_scale],
            outputs=[anim_output_video, anim_status],
            show_progress="minimal"
        )

    return app

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=CYBER_INDIGO_CSS
    )
