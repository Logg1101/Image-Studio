from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import config.paths as paths
from adapters.base import BaseAdapter, AdapterType, AdapterStatus, AdapterLoadResult
from adapters.lora.handler import UniversalLoRAHandler
from core.memory import clear_vram
from core.device import get_torch_device

SUPPORTED_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx"}

class AdapterRegistry:
    """
    Central manager for discovery, registration, synchronization,
    and memory-safe teardown of LoRA adapter modules across base models.
    """
    def __init__(self):
        self.active_loras: Dict[str, UniversalLoRAHandler] = {}
        self.active_controlnet = None
        self.active_ip_adapter = None
        self.active_pulid = None

    def scan_loras(self) -> Dict[str, Path]:
        loras = {}
        if paths.LORAS_DIR.exists():
            for p in paths.LORAS_DIR.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".safetensors", ".pt", ".bin", ".ckpt"}:
                    loras[p.name] = p
        return loras

    def scan_controlnets(self) -> Dict[str, Path]:
        cnets = {}
        if paths.CONTROLNET_DIR.exists():
            for p in paths.CONTROLNET_DIR.iterdir():
                if p.is_dir():
                    cnets[p.name.lower()] = p
                elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    cnets[p.stem.lower()] = p
        return cnets

    def scan_ip_adapters(self) -> Dict[str, Path]:
        ips = {}
        if paths.IPADAPTER_DIR.exists():
            for p in paths.IPADAPTER_DIR.rglob("*"):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    ips[p.name.lower()] = p
                    ips[p.stem.lower()] = p
        return ips

    def sync_loras(self, pipeline: Any, requested_loras: Dict[str, float], target_architecture: str = "sdxl") -> List[AdapterLoadResult]:
        """
        Synchronizes requested LoRA loadout with the diffusion pipeline.
        Reuses existing loaded adapters if file selection is unchanged.
        Updates weights in-place when weights change.
        """
        results: List[AdapterLoadResult] = []
        if pipeline is None:
            return results

        # 1. Map requested payload to {name: (path, weight)}
        req_map: Dict[str, Tuple[Path, float]] = {}
        for path_str, weight in requested_loras.items():
            p = Path(path_str)
            if not p.is_absolute():
                if (paths.LORAS_DIR / p).exists():
                    p = paths.LORAS_DIR / p
                else:
                    found = list(paths.LORAS_DIR.rglob(p.name))
                    if not found:
                        found = list(paths.LORAS_DIR.rglob(f"{p.stem}.*"))
                    if found:
                        p = found[0]
            if not p.exists():
                print(f"[AdapterRegistry] Warning: LoRA checkpoint not found on disk: {p}, skipping.")
                continue
            req_map[p.stem] = (p, float(weight))

        current_names = set(self.active_loras.keys())
        requested_names = set(req_map.keys())

        # 2. If LoRA files changed, unload old and load new
        if current_names != requested_names:
            if current_names:
                self.unload_all_loras(pipeline)

            for name, (path, weight) in req_map.items():
                handler = UniversalLoRAHandler(name=name, file_path=path)
                load_res = handler.load(pipeline, architecture=target_architecture)
                results.append(load_res)
                if load_res.status in (AdapterStatus.LOADED, AdapterStatus.PARTIALLY_LOADED):
                    handler.strength = weight
                    self.active_loras[name] = handler

            # Activate all loaded LoRAs simultaneously
            if self.active_loras:
                names = list(self.active_loras.keys())
                weights = [self.active_loras[n].strength for n in names]
                if hasattr(pipeline, "set_adapters"):
                    pipeline.set_adapters(names, adapter_weights=weights)
                if hasattr(pipeline, "set_lora_device"):
                    dev = getattr(pipeline, "device", None) or get_torch_device()
                    pipeline.set_lora_device(adapter_names=names, device=str(dev))
        else:
            # 3. Same LoRAs: check if weights changed and update in-place
            weights_changed = False
            for name, (path, weight) in req_map.items():
                if self.active_loras[name].strength != weight:
                    weights_changed = True
                    self.active_loras[name].strength = weight

            if weights_changed and self.active_loras:
                names = list(self.active_loras.keys())
                weights = [self.active_loras[n].strength for n in names]
                if hasattr(pipeline, "set_adapters"):
                    pipeline.set_adapters(names, adapter_weights=weights)

        return results

    def set_adapter_strength(self, name: str, strength: float, pipeline: Any = None) -> bool:
        """
        Updates the strength of an active LoRA while keeping all other active LoRAs
        intact on the pipeline.
        """
        if name not in self.active_loras:
            return False
        handler = self.active_loras[name]
        handler.strength = float(strength)
        if pipeline is not None and hasattr(pipeline, "set_adapters"):
            names = list(self.active_loras.keys())
            weights = [self.active_loras[n].strength for n in names]
            try:
                pipeline.set_adapters(names, adapter_weights=weights)
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                import logging
                logging.getLogger("AdapterRegistry").warning(
                    "[AdapterRegistry] set_adapters failed during set_adapter_strength: %s", e
                )
        return True

    def unload_all_loras(self, pipeline: Any = None) -> None:
        """Safely detaches all active LoRAs, clears PEFT configs, and frees VRAM."""
        for name, handler in list(self.active_loras.items()):
            handler.remove(pipeline)
            handler.unload()

        if pipeline is not None and hasattr(pipeline, "unload_lora_weights"):
            try:
                pipeline.unload_lora_weights()
            except Exception:
                pass

        self.active_loras.clear()

    def unload_all(self, pipeline: Any = None) -> None:
        """Complete teardown of all active adapters."""
        self.unload_all_loras(pipeline)
        if self.active_controlnet:
            self.active_controlnet.remove(pipeline)
            self.active_controlnet = None
        if self.active_ip_adapter:
            self.active_ip_adapter.remove(pipeline)
            self.active_ip_adapter = None
        if self.active_pulid:
            self.active_pulid.remove(pipeline)
            self.active_pulid = None
        clear_vram()

adapter_registry = AdapterRegistry()
