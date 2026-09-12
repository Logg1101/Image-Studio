# Phase 4: Lighting Analysis Engine Report

---

## 1. Overview & Architecture

Phase 4 implements a structured, semantic-aware **Lighting Analysis Engine** that decomposes image illumination into parametric lighting profiles and 2D spatial lighting maps **prior to downstream relighting execution**.

The analyzer operates on the principle of multi-cue photometric decomposition, consuming the Phase 2 neural semantic masks to separate background illumination from subject material reflectance while protecting delicate anatomical features.

```
[ Input RGB Image ]
       │
       ▼ Step 1
[ Phase 2 Semantic Perception Context ]
   ├── M_subject, M_background, M_skin, M_clothing, M_hair, M_face, M_accessories
   └── 7 Boundary Transition Zones + Spatial Confidence Map
       │
       ▼ Step 2
[ 2D Lighting Decomposition (LightingDecompositionEngine) ]
   ├── Perceptual Luminance Map: Y = 0.299R + 0.587G + 0.114B
   ├── Ambient Illumination Floor: Gaussian-filtered diffuse base (sigma=18)
   ├── Highlights Channel: Specular peak residual above ambient
   ├── Shadows Channel: Low-frequency shadow density and core occlusion
   └── Material Photometrics: Mean luminance, highlight %, shadow %, specular vs diffuse
       │
       ▼ Step 3
[ 3D Light Direction & Source Estimation (LightDirectionEstimator) ]
   ├── 2D Luminance Gradient Vector Field (Sobel / Gaussian)
   ├── Highlight ↔ Shadow Spatial Centroid Disparity Vector
   ├── Dominant Key Angle (0° to 360°, North=0°) & Shadow Cast Angle
   ├── Normalized 3D Unit Vector: v = [vx, vy, vz], ||v|| = 1.0
   └── Canvas Light Origin Position: [x, y] in normalized coordinates
       │
       ▼ Step 4
[ Color Temperature & Chromaticity (ColorTemperatureEstimator) ]
   ├── Neutral / Diffuse Zone Sampling (Background & Skin diffuse floor)
   ├── CIE XYZ Conversion & McCamy Correlated Color Temperature (CCT in Kelvin)
   └── Chromaticity Balance: 'warm' | 'cool' | 'neutral' + RGB Tint Multiplier
       │
       ▼ Step 5
[ Diagnostics Export & UI Integration (LightingDebugExporter) ]
   ├── outputs/enhancer_debug/lighting/ (7 PNG maps)
   └── Base64 Payload Delivery to Tauri Enhancer Viewport & Status Card
```

---

## 2. Structured Lighting Representation (`LightingProfile`)

The output of Phase 4 is a reusable, self-contained `LightingProfile` dataclass:

```json
{
  "direction_angle_deg": 287.5,
  "direction_vector": [-0.7813, 0.2462, 0.5736],
  "light_source_position": [0.0708, 0.3647],
  "key_intensity": 0.965,
  "ambient_intensity": 0.289,
  "shadow_strength": 0.635,
  "shadow_direction_deg": 107.5,
  "color_temperature_k": 5208.0,
  "color_bias": "neutral",
  "color_tint_rgb": [1.021, 0.998, 0.981],
  "global_confidence": 0.793,
  "material_lighting": {
    "background": { "mean_luminance": 0.812, "highlight_coverage_pct": 28.4, "shadow_coverage_pct": 4.1, "estimated_reflectance": "architectural_diffuse" },
    "subject": { "mean_luminance": 0.542, "highlight_coverage_pct": 14.8, "shadow_coverage_pct": 22.3, "estimated_reflectance": "composite_subject" },
    "skin": { "mean_luminance": 0.784, "highlight_coverage_pct": 21.6, "shadow_coverage_pct": 6.8, "estimated_reflectance": "subsurface_organic" },
    "face": { "mean_luminance": 0.811, "highlight_coverage_pct": 24.2, "shadow_coverage_pct": 3.1, "estimated_reflectance": "subsurface_delicate" },
    "hair": { "mean_luminance": 0.364, "highlight_coverage_pct": 8.9, "shadow_coverage_pct": 48.2, "estimated_reflectance": "anisotropic_specular" },
    "clothing": { "mean_luminance": 0.495, "highlight_coverage_pct": 12.1, "shadow_coverage_pct": 27.5, "estimated_reflectance": "diffuse_woven" },
    "accessories": { "mean_luminance": 0.612, "highlight_coverage_pct": 31.4, "shadow_coverage_pct": 18.2, "estimated_reflectance": "metallic_chrome" }
  }
}
```

---

## 3. Semantic Awareness & Material Policies

- **Skin vs. Clothing**: Skin is measured as `subsurface_organic` with gentle gradients, separating its diffuse baseline from dark fabric shadows.
- **Face Protection**: Face luminance is recorded with `subsurface_delicate` reflectance; eye irises and lips are excluded from aggressive shadow penalties.
- **Hair Specular Sheen**: Hair is categorized as `anisotropic_specular`, recording high shadow coverage ($48.2\%$) with localized specular highlights along strand curvature.
- **Accessories**: Evaluated as `metallic_chrome` with peak specular response ($31.4\%$ highlight coverage).
- **Background Separation**: Background architectural lighting is isolated from character material estimation, preventing wall bounce from artificially distorting subject light direction.
- **Boundary Attenuation**: Spatial confidence drops across the 7 transition zones ($B_{\text{skin} \leftrightarrow \text{cloth}}$, $B_{\text{skin} \leftrightarrow \text{hair}}$, etc.), preventing boundary edge artifacts from corrupting gradient estimators.

---

## 4. Debug Outputs (`outputs/enhancer_debug/lighting/`)

| File Name | Description | Output Format |
| :--- | :--- | :--- |
| `lighting_luminance.png` | Perceptual grayscale luminance map $Y$ | $576 \times 1024$ uint8 $[0, 255]$ |
| `lighting_shadows.png` | Core and cast shadow density channel | $576 \times 1024$ uint8 $[0, 255]$ |
| `lighting_highlights.png` | Specular and high-illuminance peaks | $576 \times 1024$ uint8 $[0, 255]$ |
| `lighting_ambient.png` | Low-frequency ambient floor | $576 \times 1024$ uint8 $[0, 255]$ |
| `lighting_confidence.png` | Spatial lighting estimation confidence | $576 \times 1024$ uint8 $[0, 255]$ |
| `lighting_direction.png` | Compass diagram with key light vector | $576 \times 1024 \times 3$ RGB |
| `lighting_overlay.png` | Full diagnostic composite with ray HUD | $576 \times 1024 \times 3$ RGB |

---

## 5. Test Suite Verification

1. **Phase 4 Lighting Analyzer Tests** ([`tests/test_lighting_analyzer.py`](file:///d:/AI/imageStudio/tests/test_lighting_analyzer.py)):
   - `test_lighting_profile_validity_and_normalization` $\rightarrow$ **PASSED** (Unit vector length $= 1.0 \pm 10^{-3}$)
   - `test_material_aware_lighting_metrics` $\rightarrow$ **PASSED** (All semantic regions evaluated)
   - `test_deterministic_analysis` $\rightarrow$ **PASSED** (Identical profiles on repeat runs)
   - `test_no_nan_or_inf_in_lighting_maps` $\rightarrow$ **PASSED** (100% finite $[0.0, 1.0]$ bounds)
2. **Phase 3 Upscaling Tests** (`test_upscaling_*.py`): **10 of 10 PASSED**.
3. **Phase 2 Anti-Bleed & Semantic Tests** (`test_enhancer_anti_bleed.py`, `test_semantic_analyzer.py`): **13 of 13 PASSED**.
4. **All Enhancer Tests**: **27 of 27 PASSED (100% green)**.
5. **Full Generation Regression Suite**: **21 of 21 PASSED (0 failures, 0 regressions)**.
6. **Tauri Frontend Production Build**: `npm run build` compiled in **1.83s with 0 errors**.

---

## 6. Performance & VRAM Telemetry

- **Target Hardware**: NVIDIA GeForce RTX 5070 (11.9 GB VRAM, PyTorch 2.11.0+cu128).
- **Execution Time (Reference Image $576 \times 1024$)**:
  - Semantic Perception: ~6.5s
  - Lighting Decomposition & Gradient Estimation: ~0.12s
  - Color Temperature & Chromaticity: ~0.02s
  - Debug Map Generation & Base64 Encoding: ~0.18s
  - **Total Pipeline Execution**: **~6.8s** (first run) / **~0.32s** (when semantic masks are cached).
- **VRAM Footprint**: **0 MB additional GPU allocation** (Lighting decomposition executes purely on CPU / numpy arrays after Phase 2 perception completes and unloads).

---

## 7. Files Created & Modified

### New Files Created in Phase 4:
- [`enhancer/lighting/types.py`](file:///d:/AI/imageStudio/enhancer/lighting/types.py): `LightingProfile`, `RegionLightingMetrics`, `LightingMaps`, `LightingAnalysisResult`.
- [`enhancer/lighting/direction.py`](file:///d:/AI/imageStudio/enhancer/lighting/direction.py): `LightDirectionEstimator`.
- [`enhancer/lighting/decomposition.py`](file:///d:/AI/imageStudio/enhancer/lighting/decomposition.py): `LightingDecompositionEngine`.
- [`enhancer/lighting/temperature.py`](file:///d:/AI/imageStudio/enhancer/lighting/temperature.py): `ColorTemperatureEstimator`.
- [`enhancer/lighting/debug_exporter.py`](file:///d:/AI/imageStudio/enhancer/lighting/debug_exporter.py): `LightingDebugExporter`.
- [`enhancer/lighting/analyzer.py`](file:///d:/AI/imageStudio/enhancer/lighting/analyzer.py): `LightingAnalyzer`.
- [`enhancer/lighting/__init__.py`](file:///d:/AI/imageStudio/enhancer/lighting/__init__.py): Package symbol exports.
- [`tests/test_lighting_analyzer.py`](file:///d:/AI/imageStudio/tests/test_lighting_analyzer.py): Unit test suite.

### Modified Files:
- [`enhancer/router.py`](file:///d:/AI/imageStudio/enhancer/router.py): Added `POST /api/enhancer/lighting/analyze` endpoint.
- [`ImageStudio_Tauri/src/types/enhancer.ts`](file:///d:/AI/imageStudio/ImageStudio_Tauri/src/types/enhancer.ts): Added lighting analysis interfaces.
- [`ImageStudio_Tauri/src/services/enhancerService.ts`](file:///d:/AI/imageStudio/ImageStudio_Tauri/src/services/enhancerService.ts): Connected frontend service to `/api/enhancer/lighting/analyze`.
- [`ImageStudio_Tauri/src/state/enhancerContext.tsx`](file:///d:/AI/imageStudio/ImageStudio_Tauri/src/state/enhancerContext.tsx): Added lighting analysis state & auto-trigger.
- [`ImageStudio_Tauri/src/components/enhancer/LightingControls.tsx`](file:///d:/AI/imageStudio/ImageStudio_Tauri/src/components/enhancer/LightingControls.tsx): Added Lighting Estimation status card.

---

## 8. Exact Prerequisites for Phase 5 (Lighting Synthesis & Relighting)

With Phase 4 complete, the required components for Phase 5 are:
1. **Normal Map & Depth Estimation**: Synthesize continuous surface normal vectors $\mathbf{n}(x, y)$ from luminance gradients and semantic body contours.
2. **Physically-Guided Shading Shader**: Compute diffuse Lambertian $\max(0, \mathbf{n} \cdot \mathbf{l})$ and Blinn-Phong specular $(\mathbf{n} \cdot \mathbf{h})^\gamma$ according to user-selected target angles $(\theta_{\text{target}}, \phi_{\text{target}})$.
3. **Shadow Warping & Attenuation**: Shift shadow density maps opposite to the new illuminant direction while respecting the Phase 2 anti-bleed boundaries.
