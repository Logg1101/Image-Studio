from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image


class BasePerceptionAdapter(ABC):
    """
    Abstract adapter for modular perception models.
    Each adapter encapsulates loading, inference, and normalization
    behind a unified contract without exposing PyTorch/CUDA details.
    """

    @property
    @abstractmethod
    def adapter_name(self) -> str:
        """Identifier for the perception adapter."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the underlying model/pipeline is ready for inference."""
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Executes prediction on the input PIL image.

        Args:
            image (Image.Image): Input source image (RGB).

        Returns:
            Dict[str, Any]: Dictionary containing raw float mask arrays [H, W] in [0.0, 1.0]
            and metadata.
        """
        pass
