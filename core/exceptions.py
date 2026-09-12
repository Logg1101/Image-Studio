class ImageStudioError(Exception):
    """Base exception for all ImageStudio errors."""
    pass

class ModelNotFoundError(ImageStudioError):
    """Raised when a requested model file cannot be located on disk."""
    pass

class ModelLoadError(ImageStudioError):
    """Raised when a model fails to load into memory."""
    pass

class UnsupportedModelError(ImageStudioError):
    """Raised when an engine is asked to load an incompatible model architecture."""
    pass

class DeviceError(ImageStudioError):
    """Raised when there is a hardware or VRAM allocation issue."""
    pass

class GenerationError(ImageStudioError):
    """Raised when the generation pipeline fails during execution."""
    pass