from typing import Dict, List, Optional
from enhancer.upscaling.base import BaseUpscalerAdapter
from enhancer.upscaling.adapters.realcugan import RealCUGANAdapter
from enhancer.upscaling.adapters.tiled_sdxl import TiledSDXLAdapter
from enhancer.upscaling.adapters.lanczos import LanczosAdapter


class UpscalerRegistry:
    """
    Registry for neural and mathematical upscaler adapters.
    """

    def __init__(self):
        self._adapters: Dict[str, BaseUpscalerAdapter] = {}
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        self.register(RealCUGANAdapter())
        self.register(TiledSDXLAdapter())
        self.register(LanczosAdapter())

    def register(self, adapter: BaseUpscalerAdapter) -> None:
        self._adapters[adapter.model_id] = adapter

    def get(self, model_id: str) -> Optional[BaseUpscalerAdapter]:
        if not model_id:
            return self._adapters.get("4x-realcugan") or next(iter(self._adapters.values()), None)

        clean_id = model_id.lower().strip()

        if clean_id in self._adapters:
            return self._adapters[clean_id]

        # Search by exact key or name match
        for k, v in self._adapters.items():
            if k.lower() == clean_id or v.name.lower() == clean_id:
                return v

        # Intelligent routing & backward-compatible aliases
        if "tiled" in clean_id or "sdxl" in clean_id:
            return self._adapters.get("tiled-sdxl")
        if "cugan" in clean_id or "realcugan" in clean_id:
            return self._adapters.get("4x-realcugan")
        if "lanczos" in clean_id or "bicubic" in clean_id:
            return self._adapters.get("lanczos-native")

        return None

    def list_available(self) -> List[Dict[str, any]]:
        return [adapter.metadata() for adapter in self._adapters.values()]


upscaler_registry = UpscalerRegistry()
