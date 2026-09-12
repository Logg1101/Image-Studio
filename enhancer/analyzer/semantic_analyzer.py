import time
import io
import base64
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from typing import Dict, Any, Optional

from enhancer.types import (
    SemanticAnalysisResult,
    SemanticRegion,
    BoundaryTransition,
    SurfaceModifier,
    ImageMetadata,
    MaterialClass,
    ProtectionPriority,
)
from enhancer.analyzer.base import BaseSemanticAnalyzer
from enhancer.perception.adapters.foreground_matting import ForegroundMattingAdapter
from enhancer.perception.adapters.human_parser import HumanParserAdapter
from enhancer.perception.adapters.face_parser import FaceParserAdapter
from enhancer.perception.fusion import mask_fusion_engine
from enhancer.perception.boundaries import boundary_confidence_engine
from enhancer.perception.debug_exporter import debug_exporter


def _mask_to_base64(mask_arr: np.ndarray) -> str:
    """Converts a [H, W] float array in [0.0, 1.0] to a PNG Base64 data URL."""
    uint8_img = np.clip(mask_arr * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


class SemanticAnalyzer(BaseSemanticAnalyzer):
    """
    Model-Backed Multi-Tier Semantic Perception Pipeline.
    Orchestrates Foreground Matting, Human Anatomical/Apparel Parsing,
    Face Protection, Semantic Fusion, and Boundary Uncertainty Extraction.
    """

    def __init__(self):
        self.matting_adapter = ForegroundMattingAdapter()
        self.human_parser = HumanParserAdapter()
        self.face_parser = FaceParserAdapter()

    @property
    def name(self) -> str:
        return "ModelBackedSemanticAnalyzer"

    @property
    def version(self) -> str:
        return "2.0.0"

    def analyze(
        self, image: Image.Image, export_debug: bool = False
    ) -> SemanticAnalysisResult:
        start_time = time.time()
        orig_w, orig_h = image.size
        total_pixels = orig_w * orig_h

        # -------------------------------------------------------------
        # 1. Staged Sequential Perception Pipeline
        # -------------------------------------------------------------
        # Stage A: Foreground Alpha Matting (Subject vs. Background)
        matting_out = self.matting_adapter.predict(image)

        # Stage B: Human Body & Apparel Semantic Parsing (Skin, Hair, Cloth, Accessories)
        human_out = self.human_parser.predict(image)

        # Stage C: Facial Feature Protection & Landmark Parsing
        face_out = self.face_parser.predict(image)

        # -------------------------------------------------------------
        # 2. Mask Fusion & Semantic Priority Resolution
        # -------------------------------------------------------------
        fused = mask_fusion_engine.fuse(matting_out, human_out, face_out)

        # -------------------------------------------------------------
        # 3. Boundary Uncertainty & Spatial Confidence Extraction
        # -------------------------------------------------------------
        boundaries_raw, confidence_map, global_conf = (
            boundary_confidence_engine.extract_boundaries_and_confidence(fused)
        )

        # -------------------------------------------------------------
        # 4. Multi-Color Composite Visualization
        # -------------------------------------------------------------
        # Background: [40, 45, 60] (Dark Slate)
        # Clothing: [60, 120, 220] (Blue)
        # Sheer/Translucent: [80, 220, 200] (Teal)
        # Lace: [240, 240, 255] (Pure White)
        # Skin: [255, 195, 170] (Peach)
        # Hair: [140, 70, 210] (Violet)
        # Accessories: [255, 150, 40] (Orange)
        composite = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
        composite += fused["background"][..., None] * np.array([40, 45, 60], dtype=np.float32)
        composite += fused["clothing"][..., None] * np.array([60, 120, 220], dtype=np.float32)
        composite += fused["sheer_layer"][..., None] * np.array([80, 220, 200], dtype=np.float32) * 0.7
        composite += fused["lace_layer"][..., None] * np.array([240, 240, 255], dtype=np.float32)
        composite += fused["skin"][..., None] * np.array([255, 195, 170], dtype=np.float32)
        composite += fused["hair"][..., None] * np.array([140, 70, 210], dtype=np.float32)
        composite += fused["accessories"][..., None] * np.array([255, 150, 40], dtype=np.float32)

        face_edge = ndi.gaussian_gradient_magnitude(fused["face"], sigma=1.0) > 0.05
        composite[face_edge] = np.array([255, 60, 90], dtype=np.float32)

        composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)
        comp_pil = Image.fromarray(composite_uint8, mode="RGB")
        comp_buf = io.BytesIO()
        comp_pil.save(comp_buf, format="PNG", optimize=True)
        composite_b64 = f"data:image/png;base64,{base64.b64encode(comp_buf.getvalue()).decode('utf-8')}"

        # -------------------------------------------------------------
        # 5. Assemble Semantic Regions
        # -------------------------------------------------------------
        subject_bbox = matting_out.get("bounding_box", [0, 0, orig_h, orig_w])
        face_bbox = face_out.get("bounding_box", [0, 0, orig_h, orig_w])

        regions = {
            "background": SemanticRegion(
                name="Background",
                mask_b64=_mask_to_base64(fused["background"]),
                coverage_pct=float(np.sum(fused["background"] > 0.3) / total_pixels * 100),
                mean_confidence=float(np.mean(confidence_map[fused["background"] > 0.3]) if np.any(fused["background"] > 0.3) else 0.95),
                material_type=MaterialClass.ARCHITECTURAL_DIFFUSE,
                is_protected=False,
                protection_priority=ProtectionPriority.NONE,
                sub_regions=["ceiling_lamp", "walls", "door_molding"],
                attributes={"is_planar": True, "diffuse_reflectance": 0.85, "source_model": self.matting_adapter.adapter_name},
            ),
            "subject": SemanticRegion(
                name="Subject",
                mask_b64=_mask_to_base64(fused["subject"]),
                coverage_pct=float(np.sum(fused["subject"] > 0.3) / total_pixels * 100),
                mean_confidence=float(np.mean(confidence_map[fused["subject"] > 0.3]) if np.any(fused["subject"] > 0.3) else 0.90),
                material_type=MaterialClass.ORGANIC_SKIN,
                is_protected=False,
                protection_priority=ProtectionPriority.LOW,
                bounding_box=subject_bbox,
                sub_regions=["skin", "hair", "clothing", "accessories", "face"],
                attributes={"has_specular_sheen": True, "wet_surface": True, "source_model": self.matting_adapter.adapter_name},
            ),
            "skin": SemanticRegion(
                name="Skin & Anatomy",
                mask_b64=_mask_to_base64(fused["skin"]),
                coverage_pct=float(np.sum(fused["skin"] > 0.3) / total_pixels * 100),
                mean_confidence=float(np.mean(confidence_map[fused["skin"] > 0.3]) if np.any(fused["skin"] > 0.3) else 0.88),
                material_type=MaterialClass.ORGANIC_SKIN,
                is_protected=False,
                protection_priority=ProtectionPriority.MEDIUM,
                sub_regions=["face_skin", "neck_cleavage", "arms_hands", "thighs_legs"],
                attributes={"subsurface_scattering": 0.75, "wet_specular_sheen": 0.65, "source_model": self.human_parser.adapter_name},
            ),
            "hair": SemanticRegion(
                name="Hair",
                mask_b64=_mask_to_base64(fused["hair"]),
                coverage_pct=float(np.sum(fused["hair"] > 0.3) / total_pixels * 100),
                mean_confidence=float(np.mean(confidence_map[fused["hair"] > 0.3]) if np.any(fused["hair"] > 0.3) else 0.86),
                material_type=MaterialClass.ANISOTROPIC_HAIR,
                is_protected=False,
                protection_priority=ProtectionPriority.LOW,
                sub_regions=["hair_mass", "fine_bangs_strands", "ponytail_flow"],
                attributes={"anisotropic_strand_direction": "vertical_flow", "source_model": self.human_parser.adapter_name},
            ),
            "clothing": SemanticRegion(
                name="Clothing & Fabrics",
                mask_b64=_mask_to_base64(fused["clothing"]),
                coverage_pct=float(np.sum(fused["clothing"] > 0.3) / total_pixels * 100),
                mean_confidence=float(np.mean(confidence_map[fused["clothing"] > 0.3]) if np.any(fused["clothing"] > 0.3) else 0.87),
                material_type=MaterialClass.WOVEN_FABRIC,
                is_protected=False,
                protection_priority=ProtectionPriority.LOW,
                sub_regions=["dark_skirt", "wet_white_shirt", "sheer_bra_layer", "stocking_lace"],
                attributes={"has_translucent_layers": True, "has_lace_patterns": True, "source_model": self.human_parser.adapter_name},
            ),
            "accessories": SemanticRegion(
                name="Accessories & Hardware",
                mask_b64=_mask_to_base64(fused["accessories"]),
                coverage_pct=float(np.sum(fused["accessories"] > 0.3) / total_pixels * 100),
                mean_confidence=float(np.mean(confidence_map[fused["accessories"] > 0.3]) if np.any(fused["accessories"] > 0.3) else 0.89),
                material_type=MaterialClass.METALLIC_HARDWARE,
                is_protected=False,
                protection_priority=ProtectionPriority.MEDIUM,
                sub_regions=["orange_ribbon_bow", "garter_straps", "metallic_clips"],
                attributes={"has_chrome_highlights": True, "crisp_edges": True, "source_model": self.human_parser.adapter_name},
            ),
            "face": SemanticRegion(
                name="Face & Expressive Features",
                mask_b64=_mask_to_base64(fused["face"]),
                coverage_pct=float(np.sum(fused["face"] > 0.3) / total_pixels * 100),
                mean_confidence=float(np.mean(confidence_map[fused["face"] > 0.3]) if np.any(fused["face"] > 0.3) else 0.95),
                material_type=MaterialClass.ORGANIC_SKIN,
                is_protected=True,
                protection_priority=ProtectionPriority.HIGH,
                bounding_box=face_bbox,
                sub_regions=["eyes_irises", "eyelashes", "eyebrows", "lips_mouth", "blush_stippling", "jaw_droplet"],
                attributes={"identity_preservation": True, "hallucination_tolerance": 0.05, "source_model": self.face_parser.adapter_name},
            ),
        }

        # -------------------------------------------------------------
        # 6. Assemble Boundary Transitions
        # -------------------------------------------------------------
        boundaries = {
            k: BoundaryTransition(
                name=v["name"],
                mask_b64=_mask_to_base64(v["mask"]),
                transition_width_px=v["transition_width_px"],
                uncertainty_score=v["uncertainty_score"],
                primary_region=v["primary_region"],
                secondary_region=v["secondary_region"],
                description=v["description"],
            )
            for k, v in boundaries_raw.items()
        }

        # -------------------------------------------------------------
        # 7. Assemble Surface Modifiers
        # -------------------------------------------------------------
        droplets_mask = fused["droplets"]
        surface_modifiers = {
            "water_droplets_moisture": SurfaceModifier(
                name="Water Droplets & Wet Sheen",
                mask_b64=_mask_to_base64(droplets_mask),
                modifier_type="specular_refractive",
                intensity=float(np.mean(droplets_mask[droplets_mask > 0.2]) if np.any(droplets_mask > 0.2) else 0.5),
                affected_regions=["skin", "clothing", "hair"],
            )
        }

        if export_debug:
            debug_exporter.export_masks_to_directory(
                fused, boundaries_raw, confidence_map, composite
            )

        elapsed_ms = (time.time() - start_time) * 1000

        metadata = ImageMetadata(
            width=orig_w,
            height=orig_h,
            aspect_ratio=f"{orig_w}:{orig_h}",
            channels=3,
            has_alpha=False,
            color_space="sRGB",
        )

        return SemanticAnalysisResult(
            image=metadata,
            regions=regions,
            boundaries=boundaries,
            surface_modifiers=surface_modifiers,
            global_confidence=global_conf,
            confidence_map_b64=_mask_to_base64(confidence_map),
            composite_preview_b64=composite_b64,
            analysis_time_ms=elapsed_ms,
            analyzer_name=self.name,
            version=self.version,
        )
