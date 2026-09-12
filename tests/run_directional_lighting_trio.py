import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancer.upscaling.engine import enhancement_engine

def run_directional_trio():
    img_path = "tests/assets/reference_flat_test_image.jpg"
    flat_img = Image.open(img_path)
    w, h = flat_img.size

    artifact_dir = r"C:\Users\ankit\.gemini\antigravity\brain\70ac2ed5-1a7b-480d-85a4-fd3b8f0d0405\directional_trio_lighting"
    os.makedirs(artifact_dir, exist_ok=True)

    print("================================================================")
    print("RUNNING DIRECTIONAL LIGHTING TRIO: RIGHT, LEFT, BOTTOM (SCALE=2)")
    print("================================================================")

    trio = [
        ("1_right", 90.0, "1. Right Directional (90 deg)"),
        ("2_left", 270.0, "2. Left Directional (270 deg)"),
        ("3_bottom", 180.0, "3. Bottom Directional (180 deg)"),
    ]

    results = {}

    for key, angle_deg, label in trio:
        print(f"\n--> Rendering {label}...")
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
            light_intensity=140.0,
            light_temperature=10.0,
            micro_relief_strength=80.0,
            shadow_depth=60.0,
            specular_strength=70.0,
            material_response=130.0,
            output_dir=f"outputs/trio_{key}",
            export_debug=True,
        )

        out_img = Image.open(res["output_path"]).convert("RGB")
        results[key] = {
            "image": out_img,
            "label": label,
            "angle": angle_deg,
        }

        # Save single directional full render
        out_single_path = os.path.join(artifact_dir, f"directional_{key}.png")
        out_img.save(out_single_path)
        print(f"Saved {out_single_path}")

    # Base 2x upscaled original
    orig_up = flat_img.resize(results["1_right"]["image"].size, Image.Resampling.BICUBIC).convert("RGB")

    # Crops for direct inspection
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
        c_right = results["1_right"]["image"].crop((x1, y1, x2, y2))
        c_left = results["2_left"]["image"].crop((x1, y1, x2, y2))
        c_bottom = results["3_bottom"]["image"].crop((x1, y1, x2, y2))

        cw, ch = c_orig.size
        strip = Image.new("RGB", (cw * 4 + 40, ch + 45), color=(15, 19, 27))
        draw = ImageDraw.Draw(strip)

        draw.text((10, 10), "Original Flat Input", fill=(180, 180, 180))
        draw.text((cw + 20, 10), "1. Right Directional (90 deg)", fill=(245, 158, 11))
        draw.text((cw * 2 + 30, 10), "2. Left Directional (270 deg)", fill=(53, 214, 197))
        draw.text((cw * 3 + 40, 10), "3. Bottom Directional (180 deg)", fill=(255, 92, 108))

        strip.paste(c_orig, (10, 35))
        strip.paste(c_right, (cw + 20, 35))
        strip.paste(c_left, (cw * 2 + 30, 35))
        strip.paste(c_bottom, (cw * 3 + 40, 35))

        out_crop_path = os.path.join(artifact_dir, f"{name}.png")
        strip.save(out_crop_path)
        print(f"Saved {name} to {out_crop_path}")

    # Full 4-Panel Comparison
    print("\n--> Generating Full 4-Panel Overview Strip...")
    full_w, full_h = orig_up.size
    full_strip = Image.new("RGB", (full_w * 4 + 40, full_h + 45), color=(15, 19, 27))
    d = ImageDraw.Draw(full_strip)

    d.text((10, 10), "Original Flat Input (2x)", fill=(180, 180, 180))
    d.text((full_w + 20, 10), "1. Right Light (90 deg)", fill=(245, 158, 11))
    d.text((full_w * 2 + 30, 10), "2. Left Light (270 deg)", fill=(53, 214, 197))
    d.text((full_w * 3 + 40, 10), "3. Bottom Light (180 deg)", fill=(255, 92, 108))

    full_strip.paste(orig_up, (10, 35))
    full_strip.paste(results["1_right"]["image"], (full_w + 20, 35))
    full_strip.paste(results["2_left"]["image"], (full_w * 2 + 30, 35))
    full_strip.paste(results["3_bottom"]["image"], (full_w * 3 + 40, 35))

    full_path = os.path.join(artifact_dir, "directional_trio_full_comparison.png")
    full_strip.save(full_path)
    print(f"Saved full trio comparison to {full_path}")


if __name__ == "__main__":
    run_directional_trio()
