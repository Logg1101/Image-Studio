import logging
from pathlib import Path
from typing import Dict, Any, List
import safetensors

logger = logging.getLogger(__name__)

class ModelArchitectureDetector:
    """
    Inspects model checkpoint weights, tensor shapes, and metadata to reliably identify
    architecture (SDXL, Flux, Pony, Illustrious, SD 1.5) without relying purely on filenames.
    """

    @staticmethod
    def detect(checkpoint_path: Path) -> Dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.exists():
            return {"architecture": "unknown", "variant": "base", "format": "unknown"}

        fmt = "gguf" if path.suffix.lower() == ".gguf" else "safetensors"

        if fmt == "gguf":
            try:
                import gguf
                reader = gguf.GGUFReader(str(path))
                arch_name = ""
                arch_field = reader.fields.get("general.architecture")
                if arch_field and getattr(arch_field, "parts", None):
                    arch_name = bytes(arch_field.parts[-1]).decode("utf-8", errors="ignore").lower().strip()

                tensor_names = [t.name.lower() for t in reader.tensors]

                has_flux = arch_name == "flux" or any(
                    "double_blocks" in tn or "single_blocks" in tn or "img_in" in tn or "txt_in" in tn
                    for tn in tensor_names
                )
                if has_flux:
                    has_guidance = any("guidance" in tn for tn in tensor_names)
                    variant = "dev" if (has_guidance or "dev" in path.stem.lower()) else "schnell"
                    return {"architecture": "flux", "variant": variant, "format": "gguf"}

                has_sdxl = (
                    arch_name in ("sdxl", "stable-diffusion-xl")
                    or any("conditioner.embedders.1" in tn or "label_emb" in tn for tn in tensor_names)
                )
                if has_sdxl:
                    return {"architecture": "sdxl", "variant": "base", "format": "gguf"}

                has_sd15 = (
                    arch_name in ("sd1", "sd15", "stable-diffusion")
                    or any("cond_stage_model" in tn for tn in tensor_names)
                )
                if has_sd15:
                    return {"architecture": "sd15", "variant": "base", "format": "gguf"}

            except (ImportError, OSError, ValueError, KeyError, AttributeError) as e:
                logger.debug(f"[ModelDetector] GGUFReader inspection failed for {path}: {e}")

            # Fallback based on filename hints if GGUFReader unavailable or fails
            stem_lower = path.stem.lower()
            if "flux" in stem_lower:
                variant = "dev" if "dev" in stem_lower else "schnell"
                return {"architecture": "flux", "variant": variant, "format": "gguf"}
            if "sdxl" in stem_lower or "xl" in stem_lower:
                return {"architecture": "sdxl", "variant": "base", "format": "gguf"}
            if "sd15" in stem_lower or "1.5" in stem_lower:
                return {"architecture": "sd15", "variant": "base", "format": "gguf"}

            return {"architecture": "flux", "variant": "schnell", "format": "gguf"}

        metadata: Dict[str, str] = {}
        all_keys: List[str] = []

        try:
            with safetensors.safe_open(str(path), framework="pt", device="cpu") as sf:
                metadata = sf.metadata() or {}
                all_keys = list(sf.keys())
        except (safetensors.SafetensorError, OSError, ValueError, KeyError) as e:
            logger.warning(f"[ModelDetector] Failed to parse safetensors for {path}: {e}")
            return {"architecture": "sdxl", "variant": "base", "format": fmt}

        # 1. Metadata checks
        if "modelspec.architecture" in metadata:
            arch = metadata["modelspec.architecture"].lower()
            if "flux" in arch:
                has_guidance = any("guidance" in k.lower() for k in all_keys)
                variant = "dev" if (has_guidance or "dev" in arch or "dev" in path.stem.lower()) else "schnell"
                return {"architecture": "flux", "variant": variant, "format": fmt}
            if "sdxl" in arch:
                return {"architecture": "sdxl", "variant": "base", "format": fmt}
            if "sd1" in arch:
                return {"architecture": "sd15", "variant": "base", "format": fmt}

        # 2. Tensor topology checks across all keys
        has_flux_tensors = any(
            "double_blocks" in kl or "single_blocks" in kl or "img_in" in kl or "txt_in" in kl
            for kl in (k.lower() for k in all_keys)
        )
        if has_flux_tensors:
            has_guidance = any("guidance" in kl for kl in (k.lower() for k in all_keys))
            variant = "dev" if (has_guidance or "dev" in path.stem.lower()) else "schnell"
            return {"architecture": "flux", "variant": variant, "format": fmt}

        has_sdxl_tensors = any(
            "conditioner.embedders.1" in kl or "model.diffusion_model.input_blocks.7" in kl or "label_emb" in kl
            for kl in (k.lower() for k in all_keys)
        )
        if has_sdxl_tensors:
            return {"architecture": "sdxl", "variant": "base", "format": fmt}

        has_sd15_tensors = any(
            "cond_stage_model" in kl or ("model.diffusion_model.input_blocks" in kl and "conditioner" not in kl)
            for kl in (k.lower() for k in all_keys)
        )
        if has_sd15_tensors:
            return {"architecture": "sd15", "variant": "base", "format": fmt}

        # Check filename fallback
        stem_lower = path.stem.lower()
        if "flux" in stem_lower:
            variant = "dev" if "dev" in stem_lower else "schnell"
            return {"architecture": "flux", "variant": variant, "format": fmt}
        if "sd15" in stem_lower or "1.5" in stem_lower:
            return {"architecture": "sd15", "variant": "base", "format": fmt}

        return {"architecture": "sdxl", "variant": "base", "format": fmt}
