from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class MaterialClass(str, Enum):
    ORGANIC_SKIN = "organic_skin"
    ANISOTROPIC_HAIR = "anisotropic_hair"
    WOVEN_FABRIC = "woven_fabric"
    SHEER_TRANSLUCENT = "sheer_translucent"
    LACE_OPENWEAVE = "lace_openweave"
    METALLIC_HARDWARE = "metallic_hardware"
    ARCHITECTURAL_DIFFUSE = "architectural_diffuse"
    SURFACE_MODIFIER = "surface_modifier"


class ProtectionPriority(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SemanticRegion:
    name: str
    mask_b64: str                         # PNG Base64 encoded soft mask (0-255 uint8)
    coverage_pct: float                   # Percentage of total image area (0.0 - 100.0)
    mean_confidence: float                # Reliability score (0.0 - 1.0)
    material_type: MaterialClass
    is_protected: bool = False
    protection_priority: ProtectionPriority = ProtectionPriority.NONE
    bounding_box: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax]
    sub_regions: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryTransition:
    name: str                             # e.g., "skin_clothing", "subject_background"
    mask_b64: str                         # PNG Base64 encoded boundary transition mask
    transition_width_px: float            # Average transition zone thickness
    uncertainty_score: float              # 0.0 (crisp/certain) to 1.0 (highly ambiguous)
    primary_region: str
    secondary_region: str
    description: str = ""


@dataclass
class SurfaceModifier:
    name: str                             # e.g., "water_droplets_moisture"
    mask_b64: str                         # PNG Base64 mask
    modifier_type: str                    # "specular_refractive", "dust", "dirt"
    intensity: float                      # 0.0 to 1.0
    affected_regions: List[str] = field(default_factory=list)


@dataclass
class ImageMetadata:
    width: int
    height: int
    aspect_ratio: str
    channels: int
    has_alpha: bool
    color_space: str = "sRGB"


@dataclass
class SemanticAnalysisResult:
    image: ImageMetadata
    regions: Dict[str, SemanticRegion]
    boundaries: Dict[str, BoundaryTransition]
    surface_modifiers: Dict[str, SurfaceModifier]
    global_confidence: float
    confidence_map_b64: str                # Spatial confidence tensor as PNG Base64
    composite_preview_b64: str            # Color-coded semantic segmentation visualization
    analysis_time_ms: float
    analyzer_name: str
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": {
                "width": self.image.width,
                "height": self.image.height,
                "aspect_ratio": self.image.aspect_ratio,
                "channels": self.image.channels,
                "has_alpha": self.image.has_alpha,
                "color_space": self.image.color_space,
            },
            "regions": {
                k: {
                    "name": v.name,
                    "mask": v.mask_b64,
                    "coverage_pct": round(v.coverage_pct, 2),
                    "confidence": round(v.mean_confidence, 3),
                    "material_type": v.material_type.value,
                    "is_protected": v.is_protected,
                    "protection_priority": v.protection_priority.value,
                    "bounding_box": v.bounding_box,
                    "sub_regions": v.sub_regions,
                    "attributes": v.attributes,
                }
                for k, v in self.regions.items()
            },
            "boundaries": {
                k: {
                    "name": v.name,
                    "mask": v.mask_b64,
                    "transition_width_px": round(v.transition_width_px, 1),
                    "uncertainty_score": round(v.uncertainty_score, 3),
                    "primary_region": v.primary_region,
                    "secondary_region": v.secondary_region,
                    "description": v.description,
                }
                for k, v in self.boundaries.items()
            },
            "surface_modifiers": {
                k: {
                    "name": v.name,
                    "mask": v.mask_b64,
                    "modifier_type": v.modifier_type,
                    "intensity": round(v.intensity, 3),
                    "affected_regions": v.affected_regions,
                }
                for k, v in self.surface_modifiers.items()
            },
            "global_confidence": round(self.global_confidence, 3),
            "confidence_map": self.confidence_map_b64,
            "composite_preview": self.composite_preview_b64,
            "analysis_time_ms": round(self.analysis_time_ms, 2),
            "analyzer_name": self.analyzer_name,
            "version": self.version,
        }
