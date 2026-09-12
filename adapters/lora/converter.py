from typing import Dict, Tuple, Any, List
import torch

class LoRAKeyConverter:
    """
    Universal key and tensor mapping layer for LoRA/LyCORIS adapters.
    Converts Kohya, LyCORIS, OneTrainer, and custom formats into normalized diffusers representations.
    """

    @staticmethod
    def normalize_state_dict(state_dict: Dict[str, torch.Tensor], target_architecture: str = "sdxl") -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        normalized: Dict[str, torch.Tensor] = {}
        filtered_out: List[str] = []
        converted_count = 0

        # Prefixes to strip or identify
        text_prefixes = (
            "lora_te1_", "lora_te2_", "lora_te_",
            "te1.", "te2.", "te.",
            "text_encoder.", "text_encoder_1.", "text_encoder_2.",
            "text_encoder1.", "text_encoder2.",
            "lycoris_te1_", "lycoris_te2_", "lycoris_te_"
        )

        for key, tensor in state_dict.items():
            key_lower = key.lower()

            # 1. Handle add_embedding (time/crop projection linear layers in SDXL)
            # These 6 tensors are composite module weights not hookable by standard PEFT UNet Linear hooks
            if "add.embedding" in key_lower or "add_embedding" in key_lower or "lora_embedding" in key_lower:
                filtered_out.append(key)
                continue

            # 2. Handle Text Encoder weights for SDXL Compel pipelines
            # In SDXL Compel pipelines, prompt conditioning is computed independently; 
            # legacy text encoder weights with incompatible projection shapes are safely separated
            if key_lower.startswith(text_prefixes) or "text_model.encoder" in key_lower or "text_model.embeddings" in key_lower:
                filtered_out.append(key)
                continue

            # 3. Normalize LyCORIS prefix ('lycoris_' -> 'lora_unet_')
            normalized_key = key
            if key_lower.startswith("lycoris_"):
                if not key_lower.startswith("lycoris_unet_"):
                    # Map lycoris_xxx -> lora_unet_xxx
                    normalized_key = "lora_unet_" + key[8:]
                else:
                    normalized_key = "lora_" + key[8:]

            normalized[normalized_key] = tensor
            converted_count += 1

        stats = {
            "total_input_tensors": len(state_dict),
            "normalized_tensors": converted_count,
            "filtered_optional_tensors": len(filtered_out),
            "target_architecture": target_architecture
        }

        return normalized, stats
