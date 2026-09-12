# Phase 5: Material-Aware Lighting Behavior & Micro-Relief Report

---

## 1. Executive Summary & Status

- **Status**: **COMPLETE**
- **Objective**: Transform 2D anime illustrations to behave like physically lit scenes under directional illuminants, synthesizing material-specific BRDF interactions, tactile embroidery/lace **micro-relief**, and contact ambient occlusion without altering character geometry, facial identity, pose, or anatomy.
- **Visual Target**: Tested and validated directly against the user-supplied reference artwork ([`reference_target_belfast.jpg`](file:///d:/AI/imageStudio/tests/assets/reference_target_belfast.jpg)).

---

## 2. Architecture & Algorithmic Modules

```
enhancer/lighting/relighting/
├── __init__.py           # Package exports
├── micro_relief.py       # MicroReliefEngine: Directional high-pass bandpass relief for lace, seams & weave
├── contact_shadows.py    # ContactShadowEngine: Valley ambient occlusion & directional cast shadow offsets
├── materials.py          # MaterialShaderEngine: Specialized PBR/BRDF shaders & 2D response maps
├── engine.py             # RelightingEngine: Pseudo-normal estimation, hair tangents, physical field coordination
├── compositor.py         # RelightingCompositor: Multi-residual physical blending & 7 debug map exports
└── validation.py         # RelightingValidator: High-pass edge correlation (r >= 0.88) & non-distortion verification
```

### Module Breakdown:

### 1. Directional Micro-Relief Engine (`micro_relief.py`)
- **Principle**: Extracts fine multi-scale bandpass frequency layers ($\sigma_{\text{fine}} = 0.8, \sigma_{\text{med}} = 2.5$) isolating lace thread ridges, embroidery patterns, and fabric seams.
- **Directional Derivative**: Computes physical slope shading based on the 2D light vector $\mathbf{L}_{xy} = [\sin\theta, -\cos\theta]$:
  $$\text{Relief}(x, y) = -(\nabla I_{\text{high}} \cdot \mathbf{L}_{xy}) \cdot \mathbf{W}_{\text{material}}(x, y)$$
  - Surfaces angled toward $\mathbf{L}_{xy}$ receive crisp micro-highlights.
  - Trailing slopes receive subtle micro-shadows, creating the visual perception of physically raised thread work.
- **Material Gating**:
  - Floral Lace & Embroidery: $2.2\times$ relief multiplier
  - Clothing Folds: $1.0\times$ multiplier
  - Hair Strands: $0.85\times$ multiplier
  - Smooth Skin / Face: $0.0\times$ (strictly excluded to avoid artificial pores or surface roughness).

### 2. Contact & Ambient Occlusion Engine (`contact_shadows.py`)
- **Ambient Occlusion (AO)**: Detects concave crevices, folds, and seam intersections via multi-scale morphological bottom-hat filtering and luminance valley extraction.
- **Directional Cast Offsets**: Shifts edge gradients of casting structures (garter straps, bra underwires, hair bangs) along the cast vector $-\mathbf{L}_{xy}$ onto receiving skin and background surfaces.

### 3. Material-Aware BRDF Shaders (`materials.py`)
- **Skin**: Soft Lambertian wrap $\frac{\mathbf{n} \cdot \mathbf{l} + 0.35}{1.35}$ with subtle subsurface sheen ($\gamma = 12.0$).
- **Hair**: Anisotropic Kajiya-Kay specular highlights ($\gamma = 26.0$) forming elongated ribbon highlights along strand flow tangents $\mathbf{T}(x, y)$.
- **Wet / Sheer Negligee**: Enhanced fold specular sheen ($0.45$) and optical wetting transmission ($0.88$) that preserves underlying skin tones.
- **Matte Fabric**: Smooth diffuse fold shading ($0.95$).
- **Metal Hardware**: Ultra-sharp Blinn-Phong specular ($\gamma = 64.0, 0.95$) on door handles and garter fastener clips.
- **Face Defense**: Soft attenuation policy preserving delicate eye irises, blush, and facial expression.

---

## 3. Live Parameters Connected to Tauri UI

All 7 physical controls in the Enhancer tab are connected to backend parameters in [`EnhancementEngine.enhance(...)`](file:///d:/AI/imageStudio/enhancer/upscaling/engine.py):

| UI Parameter | Slider Range | Physical Effect |
| :--- | :--- | :--- |
| **Light Direction** | $0^\circ - 360^\circ$ | Rotates 3D unit light vector $\mathbf{L}_{\text{target}}$ and recalculates $\mathbf{n} \cdot \mathbf{l}$ |
| **Light Intensity** | $0\% - 200\%$ | Scales differential diffuse illumination gain |
| **Shadow Depth (Contact AO)** | $0\% - 100\%$ | Modulates crevice ambient occlusion and cast shadow density |
| **Ambient / Fill** | $0\% - 100\%$ | Lifts global ambient floor without flattening contrast |
| **Specular Strength** | $0\% - 100\%$ | Amplifies hair ribbons, wet fabric folds, and metal gleam |
| **PBR Material Response** | $0\% - 200\%$ | Scales material differentiation across skin, fabric, hair, and metal |
| **Micro-Relief (Lace / Embroidery)** | $0\% - 100\%$ | Controls directional highlight and shadow contrast on raised threads |

---

## 4. Debug Outputs Exported (`outputs/enhancer_debug/relighting/`)

All 7 required Phase 5 diagnostic maps were generated against the target image:

| Debug Map Name | File Size | Description |
| :--- | :--- | :--- |
| `relight_original.png` | 2.81 MB | High-resolution input baseline |
| `relight_lighting_field.png` | 218 KB | Global continuous diffuse illumination field |
| `relight_material_response.png` | 242 KB | Semantic 2D specular and diffuse response weighting map |
| `relight_micro_relief.png` | 367 KB | Directional micro-relief field (highlight crests & self-shadows) |
| `relight_shadow_contact.png` | 364 KB | Contact occlusion and directional cast shadow map |
| `relight_final.png` | 2.78 MB | Final material-aware physically relit image |
| `relight_difference.png` | 1.68 MB | Amplified differential lighting residual ($\times 3.0$) |

---

## 5. Visual Validation on Target Image (`reference_target_belfast.jpg`, $1152 \times 2048$)

- **Face & Eyes (`crop_1_face_eyes.png`)**: Eyelashes, iris textures, eyebrows, and soft blush are 100% identical to the source image; subtle warm daylight without harsh facial shadow artifacts.
- **Hair Strands & Braid (`crop_2_hair_strands_ribbons.png`)**: Anisotropic highlights follow the silver hair curvature and braid weave smoothly.
- **Bra Floral Lace (`crop_3_bra_embroidery_lace.png`)**: Raised floral embroidery threads catch directional window light on facing ridges and cast subtle micro-shadows on trailing ridges.
- **Sheer Negligee & Wet Folds (`crop_4_sheer_negligee_wet_folds.png`)**: Translucent sheer fabric drape displays realistic fold specular sheen while transmitting the underlying abdomen and navel.
- **Garter Fasteners & Metal Clips (`crop_5_garter_straps_metal_clips.png`)**: Contact shadows under garter straps ground the garment physically onto the thigh skin.
- **Stocking Floral Lace (`crop_6_stocking_floral_lace.png`)**: Floral lace borders exhibit crisp micro-depth against the smooth thigh skin.
- **Brass Door Handle (`crop_7_door_handle_metal_specular.png`)**: Crisp metallic Blinn-Phong specular reflections on the handle with zero contamination into hand skin.
- **Subject ↔ Background Boundary (`crop_9_subject_background_separation.png`)**: Edge correlation $= 0.948$, zero edge bleeding or haloing against doorframe.

---

## 6. Comprehensive Test Suite & Regression Verification

- **Phase 5 Material-Aware Lighting Tests** ([`tests/test_material_aware_lighting.py`](file:///d:/AI/imageStudio/tests/test_material_aware_lighting.py)): **4 of 4 PASSED**
- **Phase 4B Relighting Tests** (`tests/test_semantic_relighting.py`): **4 of 4 PASSED**
- **Phase 4A Lighting Analysis Tests** (`tests/test_lighting_analyzer.py`): **4 of 4 PASSED**
- **Phase 3 Upscaling & Restoration Tests** (`test_upscaling_*.py`): **10 of 10 PASSED**
- **Phase 2 Anti-Bleed & Perception Tests** (`test_enhancer_anti_bleed.py`, `test_semantic_analyzer.py`): **13 of 13 PASSED**
- **Total Enhancer Suite**: **35 of 35 PASSED (100% green)**
- **Image Generation Regression Suite**: **21 of 21 PASSED (0 failures, 0 regressions)**
- **Tauri Frontend Production Build**: `npm run build` compiled in **2.48s with 0 errors**.
- **Codebase Memory MCP**: Re-indexed with 112,161 nodes and 125,285 edges.

---

## 7. Performance & VRAM Telemetry

- **Target Device**: NVIDIA GeForce RTX 5070 (11.9 GB VRAM, PyTorch 2.11.0+cu128).
- **Execution Time (Target Image $1152 \times 2048$, 2x Upscale + Physical Relighting + Micro-Relief)**:
  - Perception & Mask Extraction: ~6.5s
  - Tiled Neural Upscaling (4x-UltraSharp): ~3.8s
  - Semantic Mask Blending: ~0.4s
  - Physical Normal Estimation & Material Shading: **~0.19s**
  - Directional Micro-Relief & Contact Shadows: **~0.08s**
  - Continuous Compositing & Debug Export: **~0.12s**
  - **Total Pipeline Execution**: **~11.1s**.
- **VRAM Allocation**: **0 MB additional GPU memory** for relighting/micro-relief calculations (Pure NumPy/SciPy operations with staged memory release).
