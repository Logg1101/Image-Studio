from pathlib import Path
from typing import Dict, Any, Optional
import torch
import safetensors.torch

from adapters.base import BaseAdapter, AdapterType, AdapterStatus, AdapterInfo, ValidationResult, AdapterLoadResult
from adapters.lora.detector import LoRAFormatDetector
from adapters.lora.inspector import LoRACheckpointInspector
from adapters.lora.converter import LoRAKeyConverter

class UniversalLoRAHandler(BaseAdapter):
    """
    Universal LoRA & LyCORIS adapter handler conforming to BaseAdapter.
    Manages format detection, inspection, conversion, diffusers/PEFT injection,
    in-place strength updates, and complete memory-safe detachment.
    """
    def __init__(self, name: str, file_path: Path):
        super().__init__(name=name, file_path=file_path, adapter_type=AdapterType.LORA)
        self.info: AdapterInfo = LoRAFormatDetector.detect(file_path)
        self.adapter_type = self.info.adapter_type
        self.is_attached = False

    def validate(self, target_architecture: str = "sdxl") -> ValidationResult:
        return LoRACheckpointInspector.inspect(self.file_path, target_architecture)

    def load(self, pipeline: Any, **kwargs) -> AdapterLoadResult:
        if pipeline is None:
            return AdapterLoadResult(
                status=AdapterStatus.ERROR,
                adapter_name=self.name,
                adapter_type=self.adapter_type,
                tensor_count=self.info.tensor_count,
                matched_tensor_count=0,
                error_message="Target pipeline is None."
            )

        target_arch = kwargs.get("architecture", self.info.architecture)
        validation = self.validate(target_arch)
        if not validation.is_valid:
            return AdapterLoadResult(
                status=validation.status,
                adapter_name=self.name,
                adapter_type=self.adapter_type,
                tensor_count=self.info.tensor_count,
                matched_tensor_count=0,
                diagnostics=validation.diagnostics,
                error_message=validation.message
            )

        # 1. Attempt Native Diffusers Load
        try:
            pipeline.load_lora_weights(str(self.file_path), adapter_name=self.name)
            self.status = AdapterStatus.LOADED
            self.is_attached = True
            self.diagnostics = {"load_mode": "native", "target_architecture": target_arch}
            return AdapterLoadResult(
                status=AdapterStatus.LOADED,
                adapter_name=self.name,
                adapter_type=self.adapter_type,
                tensor_count=self.info.tensor_count,
                matched_tensor_count=self.info.tensor_count,
                rank=self.info.rank,
                alpha=self.info.alpha,
                architecture=target_arch,
                diagnostics=self.diagnostics
            )
        except Exception as native_err:
            self.remove(pipeline)

        # 2. Universal Normalized Fallback Conversion
        try:
            if self.file_path.suffix.lower() == ".safetensors":
                raw_sd = safetensors.torch.load_file(str(self.file_path), device="cpu")
            else:
                raw_sd = torch.load(str(self.file_path), map_location="cpu", weights_only=False)

            normalized_sd, conv_stats = LoRAKeyConverter.normalize_state_dict(raw_sd, target_architecture=target_arch)
            if not normalized_sd:
                raise RuntimeError("No usable weights remained after universal normalization.")

            pipeline.load_lora_weights(normalized_sd, adapter_name=self.name)
            self.status = AdapterStatus.LOADED
            self.is_attached = True
            self.diagnostics = {
                "load_mode": "normalized_fallback",
                "target_architecture": target_arch,
                **conv_stats
            }

            return AdapterLoadResult(
                status=AdapterStatus.LOADED,
                adapter_name=self.name,
                adapter_type=self.adapter_type,
                tensor_count=self.info.tensor_count,
                matched_tensor_count=len(normalized_sd),
                rank=self.info.rank,
                alpha=self.info.alpha,
                architecture=target_arch,
                diagnostics=self.diagnostics
            )
        except Exception as fallback_err:
            self.remove(pipeline)
            self.status = AdapterStatus.ERROR
            self.is_attached = False
            return AdapterLoadResult(
                status=AdapterStatus.ERROR,
                adapter_name=self.name,
                adapter_type=self.adapter_type,
                tensor_count=self.info.tensor_count,
                matched_tensor_count=0,
                error_message=f"Universal LoRA loading failed: {fallback_err}"
            )

    def set_strength(self, strength: float, pipeline: Any = None) -> None:
        self.strength = float(strength)
        if pipeline is not None and self.is_attached:
            try:
                if hasattr(pipeline, "set_adapters"):
                    from adapters.registry import adapter_registry

                    if adapter_registry.active_loras:
                        names = list(adapter_registry.active_loras.keys())
                        if self.name not in names:
                            names.append(self.name)
                        weights = [
                            self.strength if n == self.name else adapter_registry.active_loras[n].strength
                            for n in names
                        ]
                        pipeline.set_adapters(names, adapter_weights=weights)
                    else:
                        active = None
                        for mod_name in ["unet", "transformer"]:
                            mod = getattr(pipeline, mod_name, None)
                            if mod is not None and hasattr(mod, "active_adapters"):
                                try:
                                    active = mod.active_adapters() if callable(mod.active_adapters) else mod.active_adapters
                                    break
                                except (RuntimeError, ValueError, TypeError, AttributeError):
                                    pass
                        if active and isinstance(active, (list, tuple)) and len(active) > 1 and self.name in active:
                            names = list(active)
                            weights = [self.strength if n == self.name else 1.0 for n in names]
                            pipeline.set_adapters(names, adapter_weights=weights)
                        else:
                            pipeline.set_adapters([self.name], adapter_weights=[self.strength])
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                import logging
                logging.getLogger("UniversalLoRAHandler").warning(
                    "[UniversalLoRAHandler] set_adapters failed during set_strength: %s", e
                )

    def remove(self, pipeline: Any = None) -> None:
        if pipeline is not None:
            try:
                if hasattr(pipeline, "delete_adapters"):
                    pipeline.delete_adapters(self.name)
            except Exception:
                pass
            
            # Clean PEFT configs on submodules
            for mod_name in ["unet", "text_encoder", "text_encoder_2", "transformer"]:
                mod = getattr(pipeline, mod_name, None)
                if mod is not None and hasattr(mod, "peft_config"):
                    if isinstance(mod.peft_config, dict) and self.name in mod.peft_config:
                        try:
                            del mod.peft_config[self.name]
                        except Exception:
                            pass
        self.is_attached = False
        self.status = AdapterStatus.UNLOADED

    def unload(self) -> None:
        self.status = AdapterStatus.UNLOADED
        self.is_attached = False

    def get_info(self) -> AdapterInfo:
        return self.info
