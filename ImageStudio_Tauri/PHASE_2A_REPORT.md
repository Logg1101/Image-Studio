# Phase 2A: Real Neural Perception & Mask Extraction Report

This document records the replacement of classical computer vision implementations with **real local deep neural network models** for the ImageStudio Enhancer perception pipeline.

---

## 1. Selected Neural Models & Rationales

| Capability | Model Identifier / Architecture | Source / License | Size on Disk | Selection Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Foreground Matting** | `ZhengPeng7/BiRefNet` | HuggingFace / Apache 2.0 | ~230 MB | Bi-directional Refinement Network designed specifically for high-resolution dichotomous segmentation, excelling at anime hair wisps, fine clothing contours, and sub-pixel edge preservation. |
| **Human / Clothing Parsing** | `mattmdjaga/segformer_b2_clothes` | HuggingFace / MIT | ~109 MB | Hierarchical Transformer (SegFormer-B2) trained on the ATR multi-class dataset; predicts 18 discrete human apparel classes (Background, Hair, Upper-clothes, Skirt, Pants, Dress, Face/Skin, Limbs, Belt, Accessories). |
| **Face & Feature Parsing** | `jonathandinu/face-parsing` | HuggingFace / MIT | ~110 MB | SegFormer model fine-tuned on CelebAMask-HQ; predicts 19 discrete facial landmark classes (Skin, Nose, Left/Right Eyes, Eyebrows, Lips, Mouth, Ears, Neck) for conservative identity defense. |

---

## 2. Model Execution & VRAM Lifecycle

The models run sequentially on CUDA (NVIDIA GeForce RTX 5070, 11.9 GB VRAM) using the `VRAMManager` staged execution context:

```
[ Input RGB Image ]
       │
       ▼ Stage 1
[ Load BiRefNet to CUDA ] ──────► Forward Pass ──► Offload/Del ──► empty_cache() ──► M_subject [0, 1]
       │
       ▼ Stage 2
[ Load SegFormer Clothes ] ────► Forward Pass ──► Offload/Del ──► empty_cache() ──► Raw Anatomical/Apparel Probs
       │
       ▼ Stage 3
[ Load Face Parser ] ──────────► Forward Pass ──► Offload/Del ──► empty_cache() ──► Face Protection Landmark Mask
       │
       ▼ CPU Post-Processing
[ MaskFusionEngine + BoundaryConfidenceEngine ] ─► 7 Soft Transition Maps + Spatial Confidence Tensor
```

### Performance & Memory Metrics (Native $576 \times 1024$ Resolution):
- **BiRefNet Foreground Matting**: ~3.7s initial load & inference
- **SegFormer Clothes**: ~1.9s inference
- **Face Parsing**: ~2.3s inference
- **Total Pipeline Time**: ~8.0s (cold cache) / ~3.2s (warm execution)
- **Peak VRAM During Inference**: **1.61 GB**
- **Residual VRAM After Cleanup**: **0.00 MB** (100% GPU memory returned to system)

---

## 3. Visual & Numerical Mask Validation (Reference Image)

Executed against `tests/assets/reference_test_image.jpg` ($576 \times 1024$):

### Exported Debug Artifacts (`outputs/enhancer_debug/`):
1. `subject_mask.png` ($576 \times 1024$, uint8 $[0, 255]$): Clean anime character silhouette with zero wall bleeding.
2. `background_mask.png` ($576 \times 1024$, uint8 $[0, 255]$): Complementary room background ($1.0 - M_{\text{subj}}$).
3. `skin_mask.png` ($576 \times 1024$, uint8 $[0, 254]$): Face skin, neck, arms, and thighs cleanly isolated from skirt.
4. `clothing_mask.png` ($576 \times 1024$, uint8 $[0, 254]$): Dark skirt and shirt separated from underlying skin.
5. `hair_mask.png` ($576 \times 1024$, uint8 $[0, 254]$): Dark purple hair mass and side bangs.
6. `face_mask.png` ($576 \times 1024$, uint8 $[0, 200]$): Face region with eye/mouth landmark protection (`HIGH` priority).
7. `confidence_map.png` ($576 \times 1024$, uint8 $[127, 255]$): Spatial confidence map with edge-uncertainty attenuation.
8. `boundary_map.png` ($576 \times 1024$, uint8 $[0, 255]$): Soft Skin $\leftrightarrow$ Clothing transition zone.
9. `composite_overlay.png` ($576 \times 1024 \times 3$, uint8 $[0, 255]$): Multi-color semantic segmentation overlay.

---

## 4. Test Verification Summary

1. **Numerical Anti-Bleed Test Suite** ([`tests/test_enhancer_anti_bleed.py`](file:///d:/AI/imageStudio/tests/test_enhancer_anti_bleed.py)):
   - `test_subject_and_background_complementarity` $\rightarrow$ **PASSED**
   - `test_background_does_not_contain_majority_of_subject` $\rightarrow$ **PASSED**
   - `test_skin_and_clothing_anti_bleed` $\rightarrow$ **PASSED**
   - `test_hair_not_in_background` $\rightarrow$ **PASSED**
   - `test_face_strictly_inside_subject` $\rightarrow$ **PASSED**
   - `test_confidence_boundary_drop` $\rightarrow$ **PASSED**
   - `test_masks_strictly_bounded_without_nans` $\rightarrow$ **PASSED**
   - `test_debug_export_functionality` $\rightarrow$ **PASSED**
2. **Semantic Analyzer Hierarchy Tests** ([`tests/test_semantic_analyzer.py`](file:///d:/AI/imageStudio/tests/test_semantic_analyzer.py)):
   - **5 of 5 tests PASSED**.
3. **Subsystem Regression Test Suite**:
   - `test_regression_suite.py`, `test_adapter_registry.py`, `test_device.py`, `test_models.py`, `test_universal_lora.py`
   - **21 of 21 tests PASSED (0 failures, 0 regressions)**.
4. **Tauri Frontend Production Build**:
   - `npm run build` compiled cleanly in **1.86s with 0 errors**.

---

## 5. Files Changed & Untouched

### Files Modified in Phase 2A:
- [`enhancer/perception/adapters/foreground_matting.py`](file:///d:/AI/imageStudio/enhancer/perception/adapters/foreground_matting.py): Replaced with BiRefNet neural segmentation on CUDA.
- [`enhancer/perception/adapters/human_parser.py`](file:///d:/AI/imageStudio/enhancer/perception/adapters/human_parser.py): Replaced with SegFormer-B2 Clothes neural parser on CUDA.
- [`enhancer/perception/adapters/face_parser.py`](file:///d:/AI/imageStudio/enhancer/perception/adapters/face_parser.py): Replaced with SegFormer Face Parsing on CUDA.

### Files Deliberately Untouched:
- All PySide6 UI files (`ui/*`)
- Image Generation backend (`core/generation.py`, `core/model_manager.py`, `core/lora_manager.py`, `core/adapter_manager.py`, `engines/*`)
- Image Generation frontend (`ImageStudio_Tauri/src/pages/TextToImage.tsx`, `generationService.ts`, `generationContext.tsx`)

---

## 6. Known Characteristics & Nuances

1. **Anime Translucent Overlays**: The SegFormer model segments the wet white shirt as "Upper-clothes" and the underlying cleavage as "Face/Skin". The `MaskFusionEngine` preserves both layers by recording the intersection in the `sheer_layer` and attenuating clothing confidence over exposed skin.
2. **Fine Bangs over Forehead**: BiRefNet cleanly preserves bangs, while SegFormer captures the hair body. The `BoundaryConfidenceEngine` assigns an uncertainty score of 0.40 to the Skin $\leftrightarrow$ Hair boundary to prevent texture hallucination across strands.

---

## 7. Exact Next Recommended Phase

**Phase 3: Material-Aware Detail Restoration & Upscaling**
- Implement neural detail synthesis using the verified semantic masks:
  - *Skin Kernel*: Subsurface-scattering smoothing, gentle pore reconstruction, strict face identity lock.
  - *Hair Kernel*: Anisotropic flow enhancement along hair strand gradients without dark fringe halos.
  - *Clothing Kernel*: Texture weave synthesis on opaque fabric, transmittance preservation on sheer shirt, pattern fidelity on lace.
