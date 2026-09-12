import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancer.upscaling.engine import enhancement_engine

def run_phase8_validation():
    img_path = "tests/assets/reference_flat_test_image.jpg"
    flat_img = Image.open(img_path)
    w, h = flat_img.size

    artifact_dir = r"C:\Users\ankit\.gemini\antigravity\brain\70ac2ed5-1a7b-480d-85a4-fd3b8f0d0405\phase8_shadow_realism"
    os.makedirs(artifact_dir, exist_ok=True)

    print("================================================================")
    print("PHASE 8 — LIGHT REALISM & RAISED-DETAIL SHADOWS VALIDATION")
    print("================================================================")

    directions = [
        ("right_90", 90.0, "Right Directional (90 deg)"),
        ("left_270", 270.0, "Left Directional (270 deg)"),
        ("bottom_180", 180.0, "Bottom Directional (180 deg)"),
    ]

    results = {}

    for key, angle_deg, label in directions:
        print(f"\n--> Rendering {label} (Scale 2)...")
        debug_dir = f"outputs/phase8_{key}"
        res = enhancement_engine.enhance(
            image=flat_img,
            scale=2,
            model_id="4x-ultrasharp",
            detail_recovery=50.0,
            texture_synthesis=45.0,
            sharpen=35.0,
            face_restoration=80.0,
            relighting_enabled=True,
            light_direction_angle=angle_deg,
            light_intensity=135.0,
            light_temperature=8.0,
            micro_relief_strength=80.0,
            shadow_depth=65.0,
            specular_strength=65.0,
            material_response=130.0,
            output_dir=debug_dir,
            export_debug=True,
        )

        out_img = Image.open(res["output_path"]).convert("RGB")
        results[key] = {
            "image": out_img,
            "label": label,
            "angle": angle_deg,
            "debug_dir": debug_dir,
        }

        # Save single directional full render
        out_single_path = os.path.join(artifact_dir, f"directional_{key}.png")
        out_img.save(out_single_path)
        print(f"Saved {out_single_path}")

    # Copy primary debug maps from default debug export
    primary_debug_dir = "outputs/enhancer_debug/relighting"
    debug_files = [
        "micro_relief_highlight_map.png",
        "micro_relief_shadow_map.png",
        "directional_cast_shadow_map.png",
        "depth_map.png",
        "refined_depth_map.png",
        "normal_map_rgb.png",
        "diffuse_ndotl_map.png",
        "final_relit_image.png",
        "amplified_difference_map.png",
    ]

    for fname in debug_files:
        src = os.path.join(primary_debug_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(artifact_dir, fname)
            Image.open(src).save(dst)
            print(f"Saved debug map {fname} to {dst}")

    # Base 2x upscaled original
    orig_up = flat_img.resize(results["right_90"]["image"].size, Image.Resampling.BICUBIC).convert("RGB")

    # Crops for direct inspection of raised detail highlights, self-shadows, and cast shadows
    crops = {
        "crop_1_bra_rose_embroidery": (150, 520, 1000, 1050),
        "crop_2_panties_garter_lace": (150, 1050, 1000, 1650),
        "crop_3_white_shirt_folds": (650, 500, 1100, 1300),
        "crop_4_hair_bangs_twintails": (50, 200, 450, 950),
        "crop_5_face_blush": (350, 50, 800, 480),
        "crop_6_body_thighs": (200, 1300, 950, 1950),
    }

    print("\n--> Generating 4-Panel Crop Comparisons (Original, Right, Left, Bottom)...")
    for name, (x1, y1, x2, y2) in crops.items():
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_up.width, x2), min(orig_up.height, y2)

        c_orig = orig_up.crop((x1, y1, x2, y2))
        c_right = results["right_90"]["image"].crop((x1, y1, x2, y2))
        c_left = results["left_270"]["image"].crop((x1, y1, x2, y2))
        c_bottom = results["bottom_180"]["image"].crop((x1, y1, x2, y2))

        cw, ch = c_orig.size
        strip = Image.new("RGB", (cw * 4 + 40, ch + 45), color=(15, 19, 27))
        draw = ImageDraw.Draw(strip)

        draw.text((10, 10), "Original Flat Input", fill=(180, 180, 180))
        draw.text((cw + 20, 10), "1. Right Light (90 deg)", fill=(245, 158, 11))
        draw.text((cw * 2 + 30, 10), "2. Left Light (270 deg)", fill=(53, 214, 197))
        draw.text((cw * 3 + 40, 10), "3. Bottom Light (180 deg)", fill=(255, 92, 108))

        strip.paste(c_orig, (10, 35))
        strip.paste(c_right, (cw + 20, 35))
        strip.paste(c_left, (cw * 2 + 30, 35))
        strip.paste(c_bottom, (cw * 3 + 40, 35))

        out_crop_path = os.path.join(artifact_dir, f"{name}.png")
        strip.save(out_crop_path)
        print(f"Saved {name} to {out_crop_path}")

    # Full 4-Panel Overview
    print("\n--> Generating Full 4-Panel Overview Strip...")
    full_w, full_h = orig_up.size
    full_strip = Image.new("RGB", (full_w * 4 + 40, full_h + 45), color=(15, 19, 27))
    d = ImageDraw.Draw(full_strip)

    d.text((10, 10), "Original Flat Input (2x)", fill=(180, 180, 180))
    d.text((full_w + 20, 10), "1. Right Light (90 deg)", fill=(245, 158, 11))
    d.text((full_w * 2 + 30, 10), "2. Left Light (270 deg)", fill=(53, 214, 197))
    d.text((full_w * 3 + 40, 10), "3. Bottom Light (180 deg)", fill=(255, 92, 108))

    full_strip.paste(orig_up, (10, 35))
    full_strip.paste(results["right_90"]["image"], (full_w + 20, 35))
    full_strip.paste(results["left_270"]["image"], (full_w * 2 + 30, 35))
    full_strip.paste(results["bottom_180"]["image"], (full_w * 3 + 40, 35))

    full_path = os.path.join(artifact_dir, "phase8_directional_shadow_full_comparison.png")
    full_strip.save(full_path)
    print(f"Saved full Phase 8 comparison to {full_path}")


if __name__ == "__main__":
    run_phase8_validation()
