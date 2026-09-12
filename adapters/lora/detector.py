import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Set
import safetensors
import torch
from adapters.base import AdapterType, AdapterInfo

class LoRAFormatDetector:
    """
    Model-agnostic format, architecture, and network topology detector for LoRA checkpoints.
    Does not rely on crude string checks or filename heuristics.
    """

    @staticmethod
    def detect(file_path: Path) -> AdapterInfo:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {path}")

        metadata: Dict[str, str] = {}
        keys: list[str] = []
        sample_shapes: Dict[str, Tuple[int, ...]] = {}

        if path.suffix.lower() == ".safetensors":
            with safetensors.safe_open(str(path), framework="pt", device="cpu") as sf:
                metadata = sf.metadata() or {}
                keys = list(sf.keys())
                for k in keys[:30]:
                    sample_shapes[k] = tuple(sf.get_tensor(k).shape)
        else:
            state_dict = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict):
                keys = list(state_dict.keys())
                for k in keys[:30]:
                    if isinstance(state_dict[k], torch.Tensor):
                        sample_shapes[k] = tuple(state_dict[k].shape)
            metadata = {}

        # 1. Inspect metadata for explicit architecture declarations
        arch = "unknown"
        if "modelspec.architecture" in metadata:
            raw_arch = metadata["modelspec.architecture"].lower()
            if "sdxl" in raw_arch:
                arch = "sdxl"
            elif "flux" in raw_arch:
                arch = "flux"
            elif "sd1" in raw_arch or "sd-v1" in raw_arch:
                arch = "sd15"
            elif "sd3" in raw_arch:
                arch = "sd3"
        elif "ss_base_model_version" in metadata:
            raw_base = metadata["ss_base_model_version"].lower()
            if "sdxl" in raw_base:
                arch = "sdxl"
            elif "flux" in raw_base:
                arch = "flux"
            elif "v1" in raw_base or "1.5" in raw_base:
                arch = "sd15"
            elif "v2" in raw_base or "2.1" in raw_base:
                arch = "sd21"

        # 2. Inspect metadata for network type (LyCORIS vs Kohya vs Standard)
        adapter_type = AdapterType.LORA
        network_type = "lora"
        if "ss_network_module" in metadata:
            mod = metadata["ss_network_module"].lower()
            if "lycoris" in mod:
                adapter_type = AdapterType.LYCORIS
                network_type = metadata.get("ss_network_type", "lycoris_lora")
        if "ss_network_args" in metadata:
            try:
                args = json.loads(metadata["ss_network_args"]) if isinstance(metadata["ss_network_args"], str) else metadata["ss_network_args"]
                if isinstance(args, dict) and "network_type" in args:
                    network_type = args["network_type"]
            except Exception:
                pass

        # 3. Infer architecture and network type from tensor key topology if metadata missing
        if arch == "unknown":
            arch = LoRAFormatDetector._infer_architecture_from_keys(keys)

        # 4. Extract rank (dim) and alpha
        rank = None
        alpha = None
        if "ss_network_dim" in metadata:
            try:
                rank = int(metadata["ss_network_dim"])
            except Exception:
                pass
        if "ss_network_alpha" in metadata:
            try:
                alpha = float(metadata["ss_network_alpha"])
            except Exception:
                pass

        if rank is None or alpha is None:
            inferred_rank, inferred_alpha = LoRAFormatDetector._infer_rank_alpha_from_keys(keys, path)
            rank = rank or inferred_rank
            alpha = alpha or inferred_alpha

        title = metadata.get("modelspec.title", path.name)
        description = metadata.get("modelspec.description", metadata.get("ss_training_comment", ""))

        return AdapterInfo(
            name=path.stem,
            file_path=path,
            adapter_type=adapter_type,
            architecture=arch,
            network_type=network_type,
            rank=rank,
            alpha=alpha,
            tensor_count=len(keys),
            metadata=metadata,
            description=description
        )

    @staticmethod
    def _infer_architecture_from_keys(keys: list[str]) -> str:
        for k in keys:
            kl = k.lower()
            # SDXL specific modules
            if "input_blocks_4" in kl or "input_blocks_5" in kl or "output_blocks_6" in kl or "output_blocks_7" in kl:
                return "sd15"
            if "input_blocks_7" in kl or "input_blocks_8" in kl or "output_blocks_5" in kl:
                return "sdxl"
            if "add_embedding" in kl or "add.embedding" in kl:
                return "sdxl"
            if "te1" in kl or "te2" in kl or "text_encoder_2" in kl:
                return "sdxl"
            if "double_blocks" in kl or "single_blocks" in kl or "img_attn" in kl or "txt_attn" in kl:
                return "flux"
        return "sdxl"

    @staticmethod
    def _infer_rank_alpha_from_keys(keys: list[str], file_path: Path) -> Tuple[Optional[int], Optional[float]]:
        rank = None
        alpha = None
        if file_path.suffix.lower() == ".safetensors":
            try:
                with safetensors.safe_open(str(file_path), framework="pt", device="cpu") as sf:
                    for k in keys:
                        if rank is None and ("lora_down.weight" in k or "lora_A.weight" in k or "down.weight" in k):
                            t = sf.get_tensor(k)
                            if t.ndim >= 2:
                                rank = int(t.shape[0])
                        if alpha is None and (".alpha" in k or "_alpha" in k):
                            t = sf.get_tensor(k)
                            alpha = float(t.item()) if t.numel() == 1 else None
                        if rank is not None and alpha is not None:
                            break
            except Exception:
                pass
        return rank, alpha
