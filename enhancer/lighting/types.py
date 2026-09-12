from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class RegionLightingMetrics:
    """Lighting metrics specific to a single semantic material region."""
    region_name: str
    mean_luminance: float
    highlight_coverage_pct: float
    shadow_coverage_pct: float
    specular_intensity: float
    diffuse_intensity: float
    estimated_reflectance: str


@dataclass
class LightingProfile:
    """
    Structured parametric representation of an image's estimated lighting state.
    Independent from the downstream relighting shader implementation.
    """
    direction_angle_deg: float          # 0 = Top, 90 = Right, 180 = Bottom, 270 = Left
    direction_vector: List[float]        # [vx, vy, vz], normalized unit vector
    light_source_position: List[float]   # [x, y] in normalized image coords [0.0, 1.0]
    key_intensity: float                 # Estimated direct key-light energy [0.0, 1.0]
    ambient_intensity: float             # Estimated ambient diffuse fill energy [0.0, 1.0]
    shadow_strength: float               # Shadow depth/contrast [0.0, 1.0]
    shadow_direction_deg: float          # Direction cast by shadows (opposite to light)
    color_temperature_k: float           # Estimated color temperature in Kelvin (e.g. 5800K)
    color_bias: str                      # 'warm', 'cool', 'neutral'
    color_tint_rgb: List[float]          # [R, G, B] normalized tint multiplier
    global_confidence: float             # Reliability score of the estimation [0.0, 1.0]
    material_lighting: Dict[str, RegionLightingMetrics] = field(default_factory=dict)


@dataclass
class LightingMaps:
    """Base64 data URLs for 2D spatial lighting decomposition maps."""
    luminance_b64: str
    shadows_b64: str
    highlights_b64: str
    ambient_b64: str
    direction_vis_b64: str
    confidence_b64: str
    composite_overlay_b64: str


@dataclass
class LightingAnalysisResult:
    """Complete response payload for lighting analysis."""
    profile: LightingProfile
    maps: LightingMaps
    analysis_time_ms: float
    version: str = "4.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes LightingAnalysisResult to JSON-compliant dictionary."""
        return {
            "profile": {
                "direction_angle_deg": round(self.profile.direction_angle_deg, 1),
                "direction_vector": [round(v, 4) for v in self.profile.direction_vector],
                "light_source_position": [round(p, 4) for p in self.profile.light_source_position],
                "key_intensity": round(self.profile.key_intensity, 3),
                "ambient_intensity": round(self.profile.ambient_intensity, 3),
                "shadow_strength": round(self.profile.shadow_strength, 3),
                "shadow_direction_deg": round(self.profile.shadow_direction_deg, 1),
                "color_temperature_k": round(self.profile.color_temperature_k, 0),
                "color_bias": self.profile.color_bias,
                "color_tint_rgb": [round(c, 3) for c in self.profile.color_tint_rgb],
                "global_confidence": round(self.profile.global_confidence, 3),
                "material_lighting": {
                    k: asdict(v) for k, v in self.profile.material_lighting.items()
                },
            },
            "maps": asdict(self.maps),
            "analysis_time_ms": round(self.analysis_time_ms, 2),
            "version": self.version,
        }
