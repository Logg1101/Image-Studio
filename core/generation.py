import threading
from typing import Optional
from core.types import GenerationRequest, GenerationResult
from engines.base import GenerationEngine
from engines.flux.engine import FluxEngine
from engines.sdxl.engine import SDXLEngine
from history.manager import HistoryManager

class GenerationCoordinator:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.engines: list[GenerationEngine] = [FluxEngine(), SDXLEngine()]
        self.active_engine: Optional[GenerationEngine] = None
        self.history_manager = HistoryManager()
        self._lock = threading.Lock()
        
    def generate(self, request: GenerationRequest) -> GenerationResult:
        with self._lock:
            model_info = request.model
            
            # Determine which engine supports the model architecture
            target_engine = None
            for engine in self.engines:
                if engine.supports(model_info):
                    target_engine = engine
                    break
                    
            if not target_engine:
                raise ValueError(f"No engine supports architecture: {model_info.architecture}")
                
            # If switching architectures (e.g., Flux to SDXL), unload the old one to free VRAM
            if self.active_engine and self.active_engine != target_engine:
                print(f"Switching engine: unloading {self.active_engine.__class__.__name__}")
                self.active_engine.unload()
                
            # Load the model into the target engine if it isn't already loaded
            if not target_engine.is_loaded() or getattr(target_engine, 'current_model_info', None) != model_info:
                print(f"Loading model: {model_info.id} into {target_engine.__class__.__name__}")
                target_engine.load(model_info)
                
            self.active_engine = target_engine
            
            try:
                # Execute the generation loop
                result = self.active_engine.generate(request)
                
                # Log to History (JSON file and SQLite DB)
                try:
                    meta_path = self.history_manager.save_generation(request, result)
                    result.metadata_path = meta_path
                except Exception as e:
                    print(f"Warning: Failed to save generation history: {e}")
                    
                return result
            finally:
                from core.memory import clear_vram
                clear_vram()