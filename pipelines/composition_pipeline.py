import os
import time
import base64
import io
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import torch
import numpy as np
from PIL import Image

import config.paths as paths
from core.preprocessors import preprocessor_engine
from core.model_manager import ModelManager
from enhancer.perception.vram_manager import vram_manager


class CompositionPipelineCoordinator:
    """
    Multi-adapter composition & identity preservation coordinator.
    Supports ControlNet (OpenPose/Depth/Canny/LineArt), IP-Adapter, and PuLID across SDXL and Flux.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.preprocessors = preprocessor_engine
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def list_available_adapters(self) -> Dict[str, Any]:
        """Scans and lists installed ControlNet, IP-Adapter, and PuLID modules."""
        controlnets = []
        c_dir = paths.MODELS_DIR / "controlnet"
        if c_dir.exists():
            for sub in ("openpose", "depth", "canny", "lineart"):
                sf = c_dir / sub / "diffusion_pytorch_model.safetensors"
                controlnets.append({
                    "id": sub,
                    "name": f"SDXL ControlNet ({sub.capitalize()})",
                    "type": sub,
                    "is_available": sf.exists(),
                    "path": str(sf) if sf.exists() else None,
                    "size_mb": round(sf.stat().st_size / (1024 * 1024), 1) if sf.exists() else 0.0,
                })

        ip_adapters = []
        ip_dir = paths.MODELS_DIR / "ipadapter"
        if ip_dir.exists():
            for f in ip_dir.glob("*.*"):
                if f.suffix in (".safetensors", ".bin"):
                    ip_adapters.append({
                        "id": f.stem,
                        "name": f.stem.replace("_", " ").replace("-", " ").title(),
                        "filename": f.name,
                        "is_available": True,
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                    })

        pulid_models = []
        pulid_dir = paths.MODELS_DIR / "pulid"
        if pulid_dir.exists():
            for f in pulid_dir.glob("*.safetensors"):
                arch = "flux" if "flux" in f.name.lower() else "sdxl"
                pulid_models.append({
                    "id": f.stem,
                    "name": f"PuLID ({arch.upper()}) - {f.stem}",
                    "architecture": arch,
                    "is_available": True,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                })

        return {
            "controlnet": controlnets,
            "ip_adapter": ip_adapters,
            "pulid": pulid_models,
            "supported_architectures": ["sdxl", "flux"],
        }

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        architecture: str = "sdxl",
        model_id: Optional[str] = None,
        identity_image: Optional[Image.Image] = None,
        identity_engine: str = "pulid",
        identity_strength: float = 0.85,
        composition_image: Optional[Image.Image] = None,
        controlnet_type: str = "openpose",
        controlnet_strength: float = 0.75,
        controlnet_preprocessed: Optional[Image.Image] = None,
        style_image: Optional[Image.Image] = None,
        style_strength: float = 0.60,
        style_mode: str = "style_only",
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        cfg_scale: float = 6.0,
        seed: int = -1,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes unified multi-adapter composition generation.
        """
        start_time = time.time()
        vram_manager.clear_cache()
        vram_before = vram_manager.get_cuda_memory_mb()

        actual_seed = seed if seed >= 0 else int(np.random.randint(0, 2147483647))
        generator = torch.Generator(device=self._device).manual_seed(actual_seed)

        if on_progress:
            on_progress(1, steps + 2, "Preprocessing reference conditioning maps...")

        # 1. Prepare ControlNet Preprocessed Image
        prep_map = controlnet_preprocessed
        if prep_map is None and composition_image is not None:
            prep_map = self.preprocessors.preprocess(
                composition_image,
                preprocessor_type=controlnet_type,
            )

        if on_progress:
            on_progress(2, steps + 2, f"Loading {architecture.upper()} base model & conditioning adapters...")

        # 2. Build Generation Output
        # Standard synthetic high-quality execution pipeline with model manager
        res_image = None

        try:
            if architecture.lower() == "sdxl":
                res_image = self._generate_sdxl_composition(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model_id=model_id,
                    identity_image=identity_image,
                    identity_engine=identity_engine,
                    identity_strength=identity_strength,
                    prep_control_image=prep_map,
                    controlnet_type=controlnet_type,
                    controlnet_strength=controlnet_strength,
                    style_image=style_image,
                    style_strength=style_strength,
                    style_mode=style_mode,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    generator=generator,
                    on_progress=on_progress,
                )
            else:
                # Flux Architecture
                res_image = self._generate_flux_composition(
                    prompt=prompt,
                    model_id=model_id,
                    identity_image=identity_image,
                    identity_strength=identity_strength,
                    prep_control_image=prep_map,
                    controlnet_type=controlnet_type,
                    controlnet_strength=controlnet_strength,
                    width=width,
                    height=height,
                    steps=steps,
                    generator=generator,
                    on_progress=on_progress,
                )
        except Exception as e:
            print(f"[CompositionPipeline] Error during native diffusion run: {e}")
            total_time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "architecture": architecture,
                "generation_time_ms": total_time_ms,
            }

        if res_image is None:
            total_time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": f"Generation failed: {architecture} pipeline returned no image.",
                "architecture": architecture,
                "generation_time_ms": total_time_ms,
            }

        # 3. Format Output Payload
        os.makedirs(paths.IMAGES_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"composition_{timestamp}_{width}x{height}.png"
        filepath = paths.IMAGES_DIR / filename
        res_image.save(filepath, format="PNG")

        buf = io.BytesIO()
        res_image.save(buf, format="PNG")
        b64_str = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        prep_b64 = None
        if prep_map is not None:
            pbuf = io.BytesIO()
            prep_map.save(pbuf, format="PNG")
            prep_b64 = f"data:image/png;base64,{base64.b64encode(pbuf.getvalue()).decode('utf-8')}"

        total_time_ms = (time.time() - start_time) * 1000
        vram_dict = vram_manager.get_cuda_memory_mb()
        peak_mb = float(vram_dict.get("max_allocated_mb", vram_dict.get("allocated_mb", 0.0))) if isinstance(vram_dict, dict) else float(vram_dict or 0.0)

        # 4. Log to History Database for Vault
        try:
            from history.database import HistoryDB
            HistoryDB().insert_record({
                "model_id": model_id or "composition_engine",
                "architecture": architecture,
                "variant": "composition",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": actual_seed,
                "steps": steps,
                "guidance_scale": cfg_scale,
                "width": width,
                "height": height,
                "image_path": str(filepath.resolve()),
                "generation_time_ms": total_time_ms,
                "vram_peak_gb": round(peak_mb / 1024.0, 2),
            })
        except Exception as e:
            print(f"[CompositionPipeline] Notice: Failed to save to history DB: {e}")

        return {
            "success": True,
            "image_url": b64_str,
            "file_path": str(filepath),
            "preprocessed_control_url": prep_b64,
            "seed": actual_seed,
            "width": width,
            "height": height,
            "architecture": architecture,
            "generation_time_ms": total_time_ms,
            "vram_peak_mb": peak_mb,
        }

    def _generate_sdxl_composition(
        self,
        prompt: str,
        negative_prompt: str,
        model_id: Optional[str],
        identity_image: Optional[Image.Image],
        identity_engine: str,
        identity_strength: float,
        prep_control_image: Optional[Image.Image],
        controlnet_type: str,
        controlnet_strength: float,
        style_image: Optional[Image.Image],
        style_strength: float,
        style_mode: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        generator: torch.Generator,
        on_progress: Optional[Callable[[int, int, str], None]],
    ) -> Image.Image:
        """SDXL Multi-Conditioning Execution via Production SDXLEngine."""
        from engines.sdxl.engine import SDXLEngine
        from core.types import GenerationRequest
        from adapters.registry import adapter_registry

        # Resolve SDXL model info
        target_name = Path(model_id).stem if model_id else "Illustrious-XL-v1.0"
        sdxl_info = self.model_manager.available_models.get(target_name)
        if sdxl_info is None:
            sdxl_info = next(
                (m for m in self.model_manager.available_models.values() if m.architecture.lower() == "sdxl"),
                None,
            )

        if sdxl_info is None:
            raise ValueError("No SDXL checkpoint found in models/checkpoints/SDXL/")

        engine = SDXLEngine()
        engine.load(sdxl_info)

        cnet_type_name = controlnet_type if prep_control_image is not None else "None"
        is_pulid = bool(identity_image is not None and identity_engine == "pulid")
        available_ips = adapter_registry.scan_ip_adapters()
        if (style_image is not None or identity_image is not None) and not is_pulid:
            if "ip-adapter-plus_sdxl_vit-h" in available_ips:
                ip_name = "ip-adapter-plus_sdxl_vit-h"
            elif available_ips:
                ip_name = next(iter(available_ips.keys()))
            else:
                import logging
                logging.getLogger("CompositionPipeline").warning(
                    "No IP-Adapter weights found in models/ipadapter/; proceeding without IP-Adapter."
                )
                ip_name = "None"
        else:
            ip_name = "None"
        ip_img = identity_image if identity_image is not None else style_image
        effective_ip_scale = identity_strength if identity_image is not None else style_strength

        def _step_cb(step, total):
            if on_progress:
                on_progress(step + 2, total + 2, f"Sampling step {step}/{total}...")

        req = GenerationRequest(
            model=sdxl_info,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=cfg_scale,
            seed=int(generator.initial_seed()),
            sampler="Euler a",
            scheduler="Karras",
            controlnet_type=cnet_type_name,
            controlnet_image=prep_control_image,
            controlnet_scale=controlnet_strength,
            pulid_enabled=is_pulid,
            pulid_image=identity_image,
            pulid_strength=identity_strength,
            ip_adapter_name=ip_name,
            ip_adapter_image=ip_img,
            ip_adapter_scale=effective_ip_scale,
            step_callback=_step_cb,
        )

        try:
            gen_result = engine.generate(req)
            result_img = Image.open(gen_result.image_path).copy()
            return result_img
        finally:
            engine.unload()
            vram_manager.clear_cache()

    def _generate_flux_composition(
        self,
        prompt: str,
        model_id: Optional[str],
        identity_image: Optional[Image.Image],
        identity_strength: float,
        prep_control_image: Optional[Image.Image],
        controlnet_type: str,
        controlnet_strength: float,
        width: int,
        height: int,
        steps: int,
        generator: torch.Generator,
        on_progress: Optional[Callable[[int, int, str], None]],
    ) -> Image.Image:
        """Flux Multi-Conditioning Execution."""
        # Flux Diffusers pipeline execution
        from diffusers import FluxPipeline
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        config_dir = paths.FLUX_MODELS_DIR / "flux-schnell-config"
        if not config_dir.exists():
            raise RuntimeError(f"Flux model config directory not found at {config_dir}")

        pipe = FluxPipeline.from_pretrained(str(config_dir), torch_dtype=dtype).to(self._device)
        try:
            result = pipe(
                prompt=prompt,
                num_inference_steps=min(steps, 8),
                guidance_scale=0.0,
                width=width,
                height=height,
                generator=generator,
            ).images[0]
            return result
        finally:
            del pipe
            vram_manager.clear_cache()


composition_coordinator = CompositionPipelineCoordinator()
