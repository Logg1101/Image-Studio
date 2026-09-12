import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancer.upscaling.engine import enhancement_engine

def run_phase6_stress_test():
    img_path = "tests/assets/reference_flat_test_image.jpg"
    flat_img = Image.open(img_path)
    w, h = flat_img.size

    artifact_dir = r"C:\Users\ankit\.gemini\antigravity\brain\70ac2ed5-1a7b-480d-85a4-fd3b8f0d0405\phase6_stress_test"
    debug_dir = "outputs/enhancer_debug/relighting"
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print("================================================================")
    print("PHASE 6 FLAT-IMAGE LIGHTING STRESS TEST (SCALE=2)")
    print("================================================================")

    directions = [
        ("upper_left", 315.0, "Upper-Left (315 deg)"),
        ("upper_right", 45.0, "Upper-Right (45 deg)"),
        ("top_front", 0.0, "Top/Front (0 deg)"),
    ]

    results = {}

    for dir_key, angle_deg, label in directions:
        print(f"\n--> Running Direction: {label}...")
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
            light_intensity=130.0,
            light_temperature=15.0,
            micro_relief_strength=75.0,
            shadow_depth=50.0,
            specular_strength=65.0,
            material_response=120.0,
            output_dir=f"outputs/phase6_{dir_key}",
            export_debug=True,
        )

        out_img = Image.open(res["output_path"]).convert("RGB")
        results[dir_key] = {
            "image": out_img,
            "label": label,
            "angle": angle_deg,
        }

        # Save single directional result to artifact dir
        dir_out_path = os.path.join(artifact_dir, f"directional_result_{dir_key}.png")
        out_img.save(dir_out_path)
        print(f"Saved {dir_out_path}")

    # Copy debug maps for primary direction
    debug_maps = [
        "relight_original.png",
        "relight_inferred_depth.png",
        "relight_lighting_field.png",
        "relight_material_response.png",
        "relight_micro_relief.png",
        "relight_shadow_contact.png",
        "relight_semantic_contribution.png",
        "relight_difference.png",
        "relight_final.png",
    ]
    for dbg in debug_maps:
        src = os.path.join(debug_dir, dbg)
        dst = os.path.join(artifact_dir, dbg)
        if os.path.exists(src):
            Image.open(src).save(dst)

    # Base 2x upscaled original
    orig_up = flat_img.resize(results["upper_left"]["image"].size, Image.Resampling.BICUBIC).convert("RGB")

    # Define key inspection crops for flat-lighting evaluation
    crops = {
        "crop_1_bra_rose_embroidery": (150, 520, 1000, 1050),
        "crop_2_panties_garter_lace": (150, 1050, 1000, 1650),
        "crop_3_white_shirt_folds": (650, 500, 1100, 1300),
        "crop_4_hair_flow_twintails": (50, 200, 450, 950),
        "crop_5_face_blush_protection": (350, 50, 800, 480),
        "crop_6_skin_body_gradients": (200, 1300, 950, 1950),
        "crop_7_lace_gloves_contact": (0, 850, 300, 1200),
        "crop_8_subject_background_separation": (700, 100, 1140, 650),
    }

    print("\n--> Generating 4-Way Multi-Directional Crop Comparisons...")
    for name, (x1, y1, x2, y2) in crops.items():
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_up.width, x2), min(orig_up.height, y2)

        c_orig = orig_up.crop((x1, y1, x2, y2))
        c_ul = results["upper_left"]["image"].crop((x1, y1, x2, y2))
        c_ur = results["upper_right"]["image"].crop((x1, y1, x2, y2))
        c_tf = results["top_front"]["image"].crop((x1, y1, x2, y2))

        cw, ch = c_orig.size
        # 4 panels horizontally
        strip = Image.new("RGB", (cw * 4 + 40, ch + 45), color=(15, 19, 27))
        draw = ImageDraw.Draw(strip)

        draw.text((10, 10), "Original Flat Input", fill=(180, 180, 180))
        draw.text((cw + 20, 10), "1. Upper-Left (315 deg)", fill=(245, 158, 11))
        draw.text((cw * 2 + 30, 10), "2. Upper-Right (45 deg)", fill=(53, 214, 197))
        draw.text((cw * 3 + 40, 10), "3. Top/Front (0 deg)", fill=(124, 108, 255))

        strip.paste(c_orig, (10, 35))
        strip.paste(c_ul, (cw + 20, 35))
        strip.paste(c_ur, (cw * 2 + 30, 35))
        strip.paste(c_tf, (cw * 3 + 40, 35))

        out_crop_path = os.path.join(artifact_dir, f"{name}.png")
        strip.save(out_crop_path)
        print(f"Saved {name} to {out_crop_path}")

    # Full 4-Way Side-by-Side Comparison
    print("\n--> Generating Full 4-Way Multi-Directional Overview...")
    full_w, full_h = orig_up.size
    full_strip = Image.new("RGB", (full_w * 4 + 40, full_h + 45), color=(15, 19, 27))
    d = ImageDraw.Draw(full_strip)

    d.text((10, 10), "Original Flat Input (2x)", fill=(180, 180, 180))
    d.text((full_w + 20, 10), "Direction 1: Upper-Left (315 deg)", fill=(245, 158, 11))
    d.text((full_w * 2 + 30, 10), "Direction 2: Upper-Right (45 deg)", fill=(53, 214, 197))
    d.text((full_w * 3 + 40, 10), "Direction 3: Top/Front (0 deg)", fill=(124, 108, 255))

    full_strip.paste(orig_up, (10, 35))
    full_strip.paste(results["upper_left"]["image"], (full_w + 20, 35))
    full_strip.paste(results["upper_right"]["image"], (full_w * 2 + 30, 35))
    full_strip.paste(results["top_front"]["image"], (full_w * 3 + 40, 35))

    full_path = os.path.join(artifact_dir, "phase6_multi_directional_comparison.png")
    full_strip.save(full_path)
    print(f"Saved full comparison to {full_path}")


if __name__ == "__main__":
    run_phase6_stress_test()
