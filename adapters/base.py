from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

class AdapterType(str, Enum):
    LORA = "lora"
    LYCORIS = "lycoris"
    IP_ADAPTER = "ip_adapter"
    CONTROLNET = "controlnet"
    PULID = "pulid"
    TEXTUAL_INVERSION = "textual_inversion"

class AdapterStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADED = "loaded"
    PARTIALLY_LOADED = "partially_loaded"
    INCOMPATIBLE = "incompatible"
    INVALID = "invalid"
    ERROR = "error"

@dataclass
class AdapterInfo:
    name: str
    file_path: Path
    adapter_type: AdapterType
    architecture: str  # e.g., "sdxl", "flux", "sd15", "unknown"
    network_type: str = "lora"  # e.g., "lora", "locon", "loha", "ia3"
    rank: Optional[int] = None
    alpha: Optional[float] = None
    tensor_count: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)
    description: str = ""

@dataclass
class ValidationResult:
    is_valid: bool
    status: AdapterStatus
    message: str
    unsupported_tensors: List[str] = field(default_factory=list)
    optional_tensors: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdapterLoadResult:
    status: AdapterStatus
    adapter_name: str
    adapter_type: AdapterType
    tensor_count: int
    matched_tensor_count: int
    rank: Optional[int] = None
    alpha: Optional[float] = None
    architecture: str = "unknown"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class BaseAdapter(ABC):
    """
    Normalized universal lifecycle interface for all adapters (LoRA, LyCORIS, IP-Adapter, ControlNet, PuLID).
    Guarantees deterministic lifecycle, zero state leaks, and standardized diagnostic introspection.
    """
    def __init__(self, name: str, file_path: Path, adapter_type: AdapterType):
        self.name = name
        self.file_path = file_path
        self.adapter_type = adapter_type
        self.status = AdapterStatus.UNLOADED
        self.strength: float = 1.0
        self.diagnostics: Dict[str, Any] = {}

    @abstractmethod
    def validate(self, target_architecture: str) -> ValidationResult:
        """Inspects and validates the adapter against the target model architecture before loading."""
        pass

    @abstractmethod
    def load(self, pipeline: Any, **kwargs) -> AdapterLoadResult:
        """Loads and attaches the adapter to the diffusion pipeline."""
        pass

    @abstractmethod
    def set_strength(self, strength: float, pipeline: Any = None) -> None:
        """Updates the adapter strength/weight in-place without reloading."""
        pass

    @abstractmethod
    def remove(self, pipeline: Any = None) -> None:
        """Deactivates and detaches adapter hooks and clears PEFT/diffusers parameters."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Completely frees adapter tensors from CPU/GPU memory."""
        pass

    @abstractmethod
    def get_info(self) -> AdapterInfo:
        """Returns structured metadata and parameter info about the adapter."""
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns real-time diagnostics on adapter status, hooks, and memory placement."""
        return {
            "name": self.name,
            "file_path": str(self.file_path),
            "type": self.adapter_type.value,
            "status": self.status.value,
            "strength": self.strength,
            **self.diagnostics
        }
