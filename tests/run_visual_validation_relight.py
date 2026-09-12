import os
import sys
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancer.upscaling.engine import enhancement_engine

def run_visual_validation():
    ref_img_path = "tests/assets/reference_test_image.jpg"
    ref_img = Image.open(ref_img_path)
    w, h = ref_img.size

    artifact_dir = r"C:\Users\ankit\.gemini\antigravity\brain\70ac2ed5-1a7b-480d-85a4-fd3b8f0d0405\visual_validation_relight"
    os.makedirs(artifact_dir, exist_ok=True)

    print("Running Phase 3 Base Upscale (Scale=2)...")
    res_phase3 = enhancement_engine.enhance(
        image=ref_img,
        scale=2,
        model_id="4x-ultrasharp",
        relighting_enabled=False,
        output_dir="outputs/validation_relight",
    )

    print("Running Phase 4B Relighting (Top-Right 45 deg, Warm +40K, Scale=2)...")
    res_relit = enhancement_engine.enhance(
        image=ref_img,
        scale=2,
        model_id="4x-ultrasharp",
        relighting_enabled=True,
        light_direction_angle=45.0,
        light_intensity=120.0,
        light_temperature=40.0,
        shadow_recovery=40.0,
        highlight_recovery=65.0,
        output_dir="outputs/validation_relight",
        export_debug=True,
    )

    p3_img = Image.open(res_phase3["output_path"]).convert("RGB")
    relit_img = Image.open(res_relit["output_path"]).convert("RGB")
    orig_up = ref_img.resize(p3_img.size, Image.Resampling.BICUBIC).convert("RGB")

    crops = {
        "crop_1_face": (480, 200, 720, 440),
        "crop_2_skin_gradients": (420, 700, 700, 980),
        "crop_3_hair_highlights": (350, 100, 650, 400),
        "crop_4_clothing_folds": (380, 500, 750, 850),
        "crop_5_embroidery_lace": (440, 1100, 720, 1400),
        "crop_6_wet_translucent_clothing": (400, 360, 750, 680),
        "crop_7_skirt": (360, 720, 780, 1100),
        "crop_8_subject_background_boundary": (150, 400, 450, 750),
    }

    for name, (x1, y1, x2, y2) in crops.items():
        c_orig = orig_up.crop((x1, y1, x2, y2))
        c_p3 = p3_img.crop((x1, y1, x2, y2))
        c_relit = relit_img.crop((x1, y1, x2, y2))

        cw, ch = c_orig.size
        strip = Image.new("RGB", (cw * 3 + 20, ch + 40), color=(15, 19, 27))
        draw = ImageDraw.Draw(strip)

        draw.text((10, 10), "Original (Bicubic)", fill=(200, 200, 200))
        draw.text((cw + 20, 10), "Phase 3 (Enhanced)", fill=(100, 220, 150))
        draw.text((cw * 2 + 30, 10), "Phase 4B (Relit 45 deg Warm)", fill=(245, 158, 11))

        strip.paste(c_orig, (10, 35))
        strip.paste(c_p3, (cw + 20, 35))
        strip.paste(c_relit, (cw * 2 + 30, 35))

        out_crop_path = os.path.join(artifact_dir, f"{name}.png")
        strip.save(out_crop_path)
        print(f"Saved {name} comparison to {out_crop_path}")

    # Full side-by-side
    full_comp = Image.new("RGB", (p3_img.width * 3 + 20, p3_img.height + 40), color=(15, 19, 27))
    d = ImageDraw.Draw(full_comp)
    d.text((10, 10), "Original Input (2x)", fill=(200, 200, 200))
    d.text((p3_img.width + 20, 10), "Phase 3 Enhanced (4x-UltraSharp)", fill=(100, 220, 150))
    d.text((p3_img.width * 2 + 30, 10), "Phase 4B Controlled Relighting (45 deg, Warm)", fill=(245, 158, 11))

    full_comp.paste(orig_up, (10, 35))
    full_comp.paste(p3_img, (p3_img.width + 20, 35))
    full_comp.paste(relit_img, (p3_img.width * 2 + 30, 35))

    full_path = os.path.join(artifact_dir, "validation_full_comparison_relit.png")
    full_comp.save(full_path)
    print(f"Saved full comparison to {full_path}")


if __name__ == "__main__":
    run_visual_validation()
