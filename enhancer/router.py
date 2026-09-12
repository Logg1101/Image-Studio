import io
import base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from typing import Optional, Dict, Any, List

from enhancer.analyzer.semantic_analyzer import SemanticAnalyzer
from enhancer.upscaling.engine import enhancement_engine
from enhancer.upscaling.registry import upscaler_registry
from enhancer.lighting.analyzer import lighting_analyzer
from enhancer.lighting.relighting.engine import relighting_engine

router = APIRouter()
analyzer = SemanticAnalyzer()


class AnalyzeImageRequest(BaseModel):
    image_data_url: str
    export_debug: Optional[bool] = False


class AnalyzeLightingRequest(BaseModel):
    image_data_url: str
    export_debug: Optional[bool] = True


class PreviewLightingRequest(BaseModel):
    image_data_url: str
    light_direction_angle: Optional[float] = 287.5
    light_intensity: Optional[float] = 100.0
    light_temperature: Optional[float] = 0.0
    shadow_depth: Optional[float] = 40.0
    specular_strength: Optional[float] = 50.0
    ambient_light: Optional[float] = 30.0
    contrast: Optional[float] = 0.0
    micro_relief_strength: Optional[float] = 50.0
    rim_light_enabled: Optional[bool] = False
    rim_light_angle: Optional[float] = 45.0
    rim_light_intensity: Optional[float] = 50.0
    rim_light_color: Optional[str] = "#35D6C5"
    fill_light_enabled: Optional[bool] = False
    fill_light_angle: Optional[float] = 225.0
    fill_light_intensity: Optional[float] = 35.0
    fill_light_color: Optional[str] = "#4F9CFF"


class ProcessImageRequest(BaseModel):
    image_data_url: str
    scale: Optional[int] = Field(default=2, ge=1, le=4)
    model_id: Optional[str] = "4x-realcugan"
    detail_recovery: Optional[float] = 50.0
    texture_synthesis: Optional[float] = 40.0
    sharpen: Optional[float] = 30.0
    denoise: Optional[float] = 20.0
    face_restoration: Optional[float] = 75.0
    relighting_enabled: Optional[bool] = False
    light_direction_angle: Optional[float] = 287.5
    light_intensity: Optional[float] = 100.0
    light_temperature: Optional[float] = 0.0
    shadow_recovery: Optional[float] = 25.0
    highlight_recovery: Optional[float] = 50.0
    ambient_light: Optional[float] = 30.0
    contrast: Optional[float] = 0.0
    micro_relief_strength: Optional[float] = 50.0
    shadow_depth: Optional[float] = 40.0
    specular_strength: Optional[float] = 50.0
    material_response: Optional[float] = 100.0
    rim_light_enabled: Optional[bool] = False
    rim_light_angle: Optional[float] = 45.0
    rim_light_intensity: Optional[float] = 50.0
    rim_light_color: Optional[str] = "#35D6C5"
    fill_light_enabled: Optional[bool] = False
    fill_light_angle: Optional[float] = 225.0
    fill_light_intensity: Optional[float] = 35.0
    fill_light_color: Optional[str] = "#4F9CFF"
    output_format: Optional[str] = "PNG"
    output_dir: Optional[str] = "outputs/upscale"
    export_debug: Optional[bool] = False


@router.get("/health")
def get_enhancer_health():
    return {
        "status": "ready",
        "analyzer": analyzer.name,
        "lighting_analyzer": "SemanticLightingAnalyzer",
        "version": "8.0.0",
        "perception_status": {
            "foreground_matting": analyzer.matting_adapter.is_available,
            "human_parsing": analyzer.human_parser.is_available,
            "face_parsing": analyzer.face_parser.is_available,
        },
        "available_upscalers": upscaler_registry.list_available(),
        "supported_features": [
            "hierarchical_segmentation",
            "soft_probabilistic_masks",
            "boundary_transitions",
            "face_protection",
            "surface_modifiers",
            "confidence_estimation",
            "vram_staged_inference",
            "neural_upscaling_1x_2x_4x",
            "tiled_inference",
            "semantic_aware_blending",
            "anti_bleed_defense",
            "light_direction_estimation",
            "color_temperature_estimation",
            "lighting_decomposition",
            "material_photometrics",
            "surface_depth_reconstruction",
            "surface_normal_fields",
            "micro_relief_highlight_shadows",
            "directional_cast_shadows",
            "multi_light_rigging_key_rim_fill",
            "fast_interactive_preview",
        ],
    }


@router.get("/models")
def get_available_upscalers():
    return upscaler_registry.list_available()


@router.post("/analyze")
def analyze_image(req: AnalyzeImageRequest):
    try:
        data_str = req.image_data_url
        if "base64," in data_str:
            data_str = data_str.split("base64,")[1]

        image_bytes = base64.b64decode(data_str)
        pil_img = Image.open(io.BytesIO(image_bytes))

        result = analyzer.analyze(pil_img, export_debug=req.export_debug or False)
        return result.to_dict()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Semantic analysis failed: {str(e)}")


@router.post("/lighting/analyze")
def analyze_lighting(req: AnalyzeLightingRequest):
    try:
        data_str = req.image_data_url
        if "base64," in data_str:
            data_str = data_str.split("base64,")[1]

        image_bytes = base64.b64decode(data_str)
        pil_img = Image.open(io.BytesIO(image_bytes))

        result = lighting_analyzer.analyze(pil_img, export_debug=req.export_debug if req.export_debug is not None else True)
        return result.to_dict()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Lighting analysis failed: {str(e)}")


@router.post("/preview")
def preview_lighting(req: PreviewLightingRequest):
    """Fast (<30ms) interactive relighting preview."""
    try:
        data_str = req.image_data_url
        if "base64," in data_str:
            data_str = data_str.split("base64,")[1]

        image_bytes = base64.b64decode(data_str)
        pil_img = Image.open(io.BytesIO(image_bytes))

        preview_img = relighting_engine.fast_relight_preview(
            image=pil_img,
            light_direction_angle=req.light_direction_angle if req.light_direction_angle is not None else 287.5,
            light_intensity=req.light_intensity if req.light_intensity is not None else 100.0,
            light_temperature=req.light_temperature if req.light_temperature is not None else 0.0,
            shadow_depth=req.shadow_depth if req.shadow_depth is not None else 40.0,
            specular_strength=req.specular_strength if req.specular_strength is not None else 50.0,
            ambient_light=req.ambient_light if req.ambient_light is not None else 30.0,
            contrast=req.contrast if req.contrast is not None else 0.0,
            micro_relief_strength=req.micro_relief_strength if req.micro_relief_strength is not None else 50.0,
            rim_light_enabled=req.rim_light_enabled or False,
            rim_light_angle=req.rim_light_angle if req.rim_light_angle is not None else 45.0,
            rim_light_intensity=req.rim_light_intensity if req.rim_light_intensity is not None else 50.0,
            rim_light_color=req.rim_light_color or "#35D6C5",
            fill_light_enabled=req.fill_light_enabled or False,
            fill_light_angle=req.fill_light_angle if req.fill_light_angle is not None else 225.0,
            fill_light_intensity=req.fill_light_intensity if req.fill_light_intensity is not None else 35.0,
            fill_light_color=req.fill_light_color or "#4F9CFF",
        )

        buffered = io.BytesIO()
        preview_img.save(buffered, format="JPEG", quality=85)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {
            "success": True,
            "preview_data_url": f"data:image/jpeg;base64,{b64_str}",
            "width": preview_img.width,
            "height": preview_img.height,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Preview failed: {str(e)}")


@router.post("/process")
def process_image(req: ProcessImageRequest):
    try:
        data_str = req.image_data_url
        if "base64," in data_str:
            data_str = data_str.split("base64,")[1]

        image_bytes = base64.b64decode(data_str)
        pil_img = Image.open(io.BytesIO(image_bytes))

        result = enhancement_engine.enhance(
            image=pil_img,
            scale=req.scale,
            model_id=req.model_id,
            detail_recovery=req.detail_recovery,
            texture_synthesis=req.texture_synthesis,
            sharpen=req.sharpen,
            denoise=req.denoise,
            face_restoration=req.face_restoration,
            relighting_enabled=req.relighting_enabled or False,
            light_direction_angle=req.light_direction_angle if req.light_direction_angle is not None else 287.5,
            light_intensity=req.light_intensity if req.light_intensity is not None else 100.0,
            light_temperature=req.light_temperature if req.light_temperature is not None else 0.0,
            shadow_recovery=req.shadow_recovery if req.shadow_recovery is not None else 25.0,
            highlight_recovery=req.highlight_recovery if req.highlight_recovery is not None else 50.0,
            ambient_light=req.ambient_light if req.ambient_light is not None else 30.0,
            contrast=req.contrast if req.contrast is not None else 0.0,
            micro_relief_strength=req.micro_relief_strength if req.micro_relief_strength is not None else 50.0,
            shadow_depth=req.shadow_depth if req.shadow_depth is not None else 40.0,
            specular_strength=req.specular_strength if req.specular_strength is not None else 50.0,
            material_response=req.material_response if req.material_response is not None else 100.0,
            rim_light_enabled=req.rim_light_enabled or False,
            rim_light_angle=req.rim_light_angle if req.rim_light_angle is not None else 45.0,
            rim_light_intensity=req.rim_light_intensity if req.rim_light_intensity is not None else 50.0,
            rim_light_color=req.rim_light_color or "#35D6C5",
            fill_light_enabled=req.fill_light_enabled or False,
            fill_light_angle=req.fill_light_angle if req.fill_light_angle is not None else 225.0,
            fill_light_intensity=req.fill_light_intensity if req.fill_light_intensity is not None else 35.0,
            fill_light_color=req.fill_light_color or "#4F9CFF",
            output_format=req.output_format,
            output_dir=req.output_dir,
            export_debug=req.export_debug or False,
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Image enhancement failed: {str(e)}")
