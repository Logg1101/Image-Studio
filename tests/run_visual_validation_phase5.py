import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancer.upscaling.engine import enhancement_engine

def run_phase5_visual_validation():
    ref_img_path = "tests/assets/reference_target_belfast.jpg"
    ref_img = Image.open(ref_img_path)
    w, h = ref_img.size

    artifact_dir = r"C:\Users\ankit\.gemini\antigravity\brain\70ac2ed5-1a7b-480d-85a4-fd3b8f0d0405\visual_validation_phase5"
    os.makedirs(artifact_dir, exist_ok=True)

    print("Running Phase 5 Material-Aware Physical Lighting (Scale=2)...")
    res = enhancement_engine.enhance(
        image=ref_img,
        scale=2,
        model_id="4x-ultrasharp",
        detail_recovery=50.0,
        texture_synthesis=45.0,
        sharpen=35.0,
        face_restoration=80.0,
        relighting_enabled=True,
        light_direction_angle=305.0,  # Top-Left window daylight
        light_intensity=125.0,
        light_temperature=20.0,      # Natural daylight room bounce
        micro_relief_strength=70.0,  # Fine floral lace & embroidery depth
        shadow_depth=50.0,           # Contact occlusion under straps/underwire
        specular_strength=65.0,      # Hair ribbons, sheer highlights, brass handle
        material_response=120.0,
        output_dir="outputs/validation_phase5",
        export_debug=True,
    )

    out_img = Image.open(res["output_path"]).convert("RGB")
    orig_up = ref_img.resize(out_img.size, Image.Resampling.BICUBIC).convert("RGB")

    crops = {
        "crop_1_face_eyes": (350, 150, 750, 500),
        "crop_2_hair_strands_ribbons": (250, 50, 680, 420),
        "crop_3_bra_embroidery_lace": (200, 750, 950, 1350),
        "crop_4_sheer_negligee_wet_folds": (300, 1150, 850, 1600),
        "crop_5_garter_straps_metal_clips": (400, 1400, 900, 1900),
        "crop_6_stocking_floral_lace": (450, 1750, 1050, 2040),
        "crop_7_door_handle_metal_specular": (800, 750, 1140, 1300),
        "crop_8_skin_subsurface_gradients": (300, 450, 850, 850),
        "crop_9_subject_background_separation": (50, 200, 400, 800),
    }

    for name, (x1, y1, x2, y2) in crops.items():
        # Clamp crop bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out_img.width, x2), min(out_img.height, y2)

        c_orig = orig_up.crop((x1, y1, x2, y2))
        c_out = out_img.crop((x1, y1, x2, y2))

        cw, ch = c_orig.size
        strip = Image.new("RGB", (cw * 2 + 20, ch + 40), color=(15, 19, 27))
        draw = ImageDraw.Draw(strip)

        draw.text((10, 10), "Original Input (2x Bicubic)", fill=(200, 200, 200))
        draw.text((cw + 20, 10), "Phase 5 Material-Aware Lighting + Micro-Relief", fill=(53, 214, 197))

        strip.paste(c_orig, (10, 35))
        strip.paste(c_out, (cw + 20, 35))

        out_crop_path = os.path.join(artifact_dir, f"{name}.png")
        strip.save(out_crop_path)
        print(f"Saved {name} comparison to {out_crop_path}")

    # Full side-by-side
    full_comp = Image.new("RGB", (out_img.width * 2 + 20, out_img.height + 40), color=(15, 19, 27))
    d = ImageDraw.Draw(full_comp)
    d.text((10, 10), "Original Input (2x)", fill=(200, 200, 200))
    d.text((out_img.width + 20, 10), "Phase 5 Enhanced (Material BRDF + Micro-Relief + Contact Shadows)", fill=(53, 214, 197))

    full_comp.paste(orig_up, (10, 35))
    full_comp.paste(out_img, (out_img.width + 20, 35))

    full_path = os.path.join(artifact_dir, "validation_full_comparison_phase5.png")
    full_comp.save(full_path)
    print(f"Saved full comparison to {full_path}")


if __name__ == "__main__":
    run_phase5_visual_validation()
