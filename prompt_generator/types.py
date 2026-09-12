from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

@dataclass
class LLMSettings:
    enabled: bool = True
    provider: str = "openai_compatible"
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = ""
    model: str = "joycaption"
    temperature: float = 0.5
    max_tokens: int = 512
    timeout: int = 60
    default_profile: str = "joycaption"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMSettings":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

@dataclass
class PromptGenerateRequest:
    text: str
    profile: str = "joycaption"
    style: str = "General"
    existing_prompt: str = ""
    mode: str = "append"  # "replace" or "append"
    settings: Optional[LLMSettings] = None

@dataclass
class PromptGenerateResponse:
    success: bool
    prompt: str
    negative_prompt: str = ""
    raw_tags: str = ""
    parsed_tags: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ModelProfileInfo:
    id: str
    name: str
    description: str
    negative_prompt: str = ""
    tag_style: str = "visual_tags"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
