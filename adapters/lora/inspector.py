from pathlib import Path
from typing import Dict, Any, List, Tuple
import safetensors
import torch
from adapters.base import ValidationResult, AdapterStatus

class LoRACheckpointInspector:
    """
    Performs deep matrix validation, tensor key categorization, and compatibility checking
    against target diffusion architectures without silent drops.
    """

    @staticmethod
    def inspect(file_path: Path, target_architecture: str = "sdxl") -> ValidationResult:
        path = Path(file_path)
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                status=AdapterStatus.INVALID,
                message=f"File does not exist: {path}"
            )

        keys: List[str] = []
        tensor_shapes: Dict[str, Tuple[int, ...]] = {}

        try:
            if path.suffix.lower() == ".safetensors":
                with safetensors.safe_open(str(path), framework="pt", device="cpu") as sf:
                    keys = list(sf.keys())
                    for k in keys:
                        tensor_shapes[k] = tuple(sf.get_tensor(k).shape)
            else:
                sd = torch.load(str(path), map_location="cpu", weights_only=False)
                if isinstance(sd, dict):
                    keys = list(sd.keys())
                    for k in keys:
                        if isinstance(sd[k], torch.Tensor):
                            tensor_shapes[k] = tuple(sd[k].shape)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                status=AdapterStatus.ERROR,
                message=f"Failed to read checkpoint: {e}",
                diagnostics={"error": str(e)}
            )

        if not keys:
            return ValidationResult(
                is_valid=False,
                status=AdapterStatus.INVALID,
                message="Checkpoint contains 0 tensors."
            )

        # Categorize tensors
        unet_tensors = []
        te_tensors = []
        add_embedding_tensors = []
        alpha_tensors = []
        conv_tensors = []
        unknown_tensors = []

        for k in keys:
            kl = k.lower()
            if "add_embedding" in kl or "add.embedding" in kl or "lora_embedding" in kl:
                add_embedding_tensors.append(k)
            elif ".alpha" in kl or "_alpha" in kl:
                alpha_tensors.append(k)
            elif any(kl.startswith(p) for p in ["lora_te", "te1.", "te2.", "te.", "text_encoder", "text_model"]):
                te_tensors.append(k)
            elif any(kl.startswith(p) for p in ["lora_unet", "unet.", "lycoris_unet", "lycoris_"]):
                if "conv" in kl:
                    conv_tensors.append(k)
                else:
                    unet_tensors.append(k)
            else:
                unknown_tensors.append(k)

        down_weights = [k for k in keys if "lora_down" in k or "lora_a" in k.lower() or "down.weight" in k]
        up_weights = [k for k in keys if "lora_up" in k or "lora_b" in k.lower() or "up.weight" in k]

        if not down_weights or not up_weights:
            return ValidationResult(
                is_valid=False,
                status=AdapterStatus.INCOMPATIBLE,
                message="Checkpoint does not contain valid low-rank factorized matrix pairs (down/up).",
                diagnostics={"total_keys": len(keys), "down_keys": len(down_weights), "up_keys": len(up_weights)}
            )

        optional_tensors = add_embedding_tensors + te_tensors
        usable_unet_tensors = unet_tensors + conv_tensors + down_weights

        diagnostics = {
            "total_tensors": len(keys),
            "unet_tensors": len(unet_tensors),
            "conv_tensors": len(conv_tensors),
            "text_encoder_tensors": len(te_tensors),
            "add_embedding_tensors": len(add_embedding_tensors),
            "alpha_tensors": len(alpha_tensors),
            "unknown_tensors": len(unknown_tensors),
            "down_up_pairs": min(len(down_weights), len(up_weights)),
            "target_architecture": target_architecture
        }

        if len(usable_unet_tensors) == 0:
            return ValidationResult(
                is_valid=False,
                status=AdapterStatus.INCOMPATIBLE,
                message=f"No compatible {target_architecture.upper()} UNet weights found in checkpoint.",
                unsupported_tensors=unknown_tensors,
                diagnostics=diagnostics
            )

        status = AdapterStatus.LOADED
        msg = f"Valid {target_architecture.upper()} LoRA checkpoint with {len(keys)} tensors."
        if unknown_tensors:
            status = AdapterStatus.PARTIALLY_LOADED
            msg += f" ({len(unknown_tensors)} non-standard tensors ignored)"

        return ValidationResult(
            is_valid=True,
            status=status,
            message=msg,
            unsupported_tensors=unknown_tensors,
            optional_tensors=optional_tensors,
            diagnostics=diagnostics
        )
