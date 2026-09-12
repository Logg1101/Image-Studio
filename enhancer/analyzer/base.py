from abc import ABC, abstractmethod
from PIL import Image
from enhancer.types import SemanticAnalysisResult


class BaseSemanticAnalyzer(ABC):
    """
    Abstract interface for semantic image analyzers.
    Any concrete segmentation, matting, or vision-model parser implements this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name/Identifier of the analyzer implementation."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string."""
        pass

    @abstractmethod
    def analyze(self, image: Image.Image) -> SemanticAnalysisResult:
        """
        Executes hierarchical semantic analysis on the input PIL Image.

        Args:
            image (Image.Image): Input source image (RGB or RGBA).

        Returns:
            SemanticAnalysisResult: Structured hierarchy containing soft masks,
            confidence maps, boundary transitions, and face protection metadata.
        """
        pass
