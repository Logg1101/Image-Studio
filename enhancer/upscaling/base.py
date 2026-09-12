from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class BaseUpscalerAdapter(ABC):
    """
    Abstract contract for neural upscaler and restoration model adapters.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the upscaler."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier or relative path."""
        pass

    @property
    @abstractmethod
    def native_scale(self) -> int:
        """Native upscale factor (e.g. 4 for 4x ESRGAN)."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True if model weights exist locally and can be loaded."""
        pass

    @abstractmethod
    def supports_scale(self, scale: int) -> bool:
        """Returns True if the adapter can produce the requested scale (1x, 2x, 4x)."""
        pass

    @abstractmethod
    def estimated_vram_mb(self) -> float:
        """Estimated peak VRAM consumption during inference in MB."""
        pass

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Loads model weights to the target device."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unloads model weights from memory."""
        pass

    @abstractmethod
    def upscale(self, image_tensor: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Upscales an input image tensor [1, 3, H, W] in [0.0, 1.0].

        Returns:
            torch.Tensor: Upscaled image tensor [1, 3, H*scale, W*scale] in [0.0, 1.0].
        """
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Returns adapter telemetry and capabilities."""
        pass
