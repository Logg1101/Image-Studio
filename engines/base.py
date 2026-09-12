from abc import ABC, abstractmethod
from core.types import GenerationRequest, GenerationResult, ModelInfo

class GenerationEngine(ABC):
    """
    Abstract base class for all generation engines.
    Engines must implement these methods to be managed by the Core Application.
    """
    
    @abstractmethod
    def load(self, model_info: ModelInfo) -> None:
        """Load the model weights into memory."""
        pass
        
    @abstractmethod
    def unload(self) -> None:
        """Remove the model from memory and clear VRAM."""
        pass
        
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute the generation pipeline."""
        pass
        
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded in the engine."""
        pass
        
    @abstractmethod
    def supports(self, model_info: ModelInfo) -> bool:
        """Check if this engine supports the provided model architecture."""
        pass