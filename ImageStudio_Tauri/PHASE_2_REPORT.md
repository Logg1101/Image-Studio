# Phase 2: Real Semantic Perception & Mask Extraction Report

This document records the design, implementation, mathematical formulation, memory architecture, and test verification for **Phase 2 — Real Semantic Perception & Mask Extraction**.

---

## 1. Files Created & Modified

### Files Created:
```
enhancer/
├── perception/
│   ├── __init__.py                             # Exports perception module symbols
│   ├── vram_manager.py                         # Staged sequential GPU execution & cache purge
│   ├── fusion.py                               # MaskFusionEngine with probabilistic conflict resolution
│   ├── boundaries.py                           # BoundaryConfidenceEngine for 7 soft transitions & uncertainty
│   ├── debug_exporter.py                       # DebugArtifactExporter for saving masks to disk
│   └── adapters/
│       ├── __init__.py
│       ├── base_adapter.py                     # BasePerceptionAdapter interface
│       ├── foreground_matting.py               # ForegroundMattingAdapter (Subject vs. Background alpha)
│       ├── human_parser.py                     # HumanParserAdapter (Skin, Hair, Opaque/Sheer/Lace Cloth, Accessories)
│       └── face_parser.py                      # FaceParserAdapter (Face protection & Landmark parser)

tests/
├── test_enhancer_anti_bleed.py                 # Mathematical anti-bleed test suite (8 test cases)
└── assets/reference_test_image.jpg             # Reference illustration test asset

ImageStudio_Tauri/
└── PHASE_2_REPORT.md                           # This Phase 2 comprehensive technical report
```

### Files Modified:
- [`enhancer/analyzer/semantic_analyzer.py`](file:///d:/AI/imageStudio/enhancer/analyzer/semantic_analyzer.py): Refactored to orchestrate `ForegroundMattingAdapter`, `HumanParserAdapter`, `FaceParserAdapter`, `MaskFusionEngine`, `BoundaryConfidenceEngine`, and `VRAMManager`.
- [`enhancer/router.py`](file:///d:/AI/imageStudio/enhancer/router.py): Extended `/api/enhancer/health` to report perception adapter statuses, and `/api/enhancer/analyze` with optional `export_debug` parameter.

### Existing Files Deliberately NOT Touched:
- All PySide6 UI files (`ui/*`)
- Image Generation backend (`core/generation.py`, `engines/sdxl/*`, `engines/flux/*`, `core/model_manager.py`, `core/lora_manager.py`)
- Image Generation frontend (`ImageStudio_Tauri/src/pages/TextToImage.tsx`, `generationService.ts`, `generationContext.tsx`, generation components)

---

## 2. Perception Model Architecture

Rather than a single monolithic network, Phase 2 implements a **staged multi-tier perception pipeline**:

```
INPUT IMAGE (RGB)
        │
        ▼ (Stage 1: VRAM Staged)
[ ForegroundMattingAdapter ] ────────► Raw Subject Alpha Mask M_subj ∈ [0, 1]
        │
        ▼ (Stage 2: VRAM Staged)
[ HumanParserAdapter ]       ────────► Raw Multi-Class Probabilities (Skin, Hair, Cloth, Sheer, Lace, Accessories, Droplets)
        │
        ▼ (Stage 3: VRAM Staged)
[ FaceParserAdapter ]        ────────► Facial Landmarks & High-Priority Face Protection Mask
        │
        ▼ (CPU Tensor Math)
[ MaskFusionEngine ]         ────────► Semantic Priority Resolution & Subject Containment
        │
        ▼ (CPU Spatial Math)
[ BoundaryConfidenceEngine ] ────────► 7 Transition Maps + Spatial Uncertainty Heatmap
        │
        ▼
SemanticAnalysisResult (JSON / Base64)
```

---

## 3. Mathematical Layered Mask Contract

All internal masks are maintained as single-channel floating-point tensors:
$$M(x, y) \in [0.0, 1.0]$$

### Priority & Containment Rules:
1. **Background Mask**:
   $$M_{\text{background}} = \text{clamp}(1.0 - M_{\text{subject}}, 0.0, 1.0)$$
2. **Skin & Anatomy Mask**:
   $$M_{\text{skin}} = \text{clamp}(M_{\text{subject}} \odot M_{\text{raw\_skin}}, 0.0, 1.0)$$
3. **Face Protection Zone**:
   $$M_{\text{face}} = \text{clamp}(M_{\text{subject}} \odot M_{\text{raw\_face}}, 0.0, 1.0) \quad (\text{Priority: HIGH})$$
4. **Hair Mask**:
   $$M_{\text{hair}} = \text{clamp}(M_{\text{subject}} \odot M_{\text{raw\_hair}} \odot (1.0 - M_{\text{skin}} \times 0.40), 0.0, 1.0)$$
5. **Clothing & Multi-Layer Fabrics**:
   $$M_{\text{clothing}} = \text{clamp}(M_{\text{subject}} \odot M_{\text{raw\_clothing}} \odot (1.0 - M_{\text{skin}} \times 0.75), 0.0, 1.0)$$
6. **Accessories & Hardware**:
   $$M_{\text{accessories}} = \text{clamp}(M_{\text{subject}} \odot M_{\text{raw\_accessories}}, 0.0, 1.0)$$

---

## 4. Boundary Protection & Spatial Confidence

The engine generates 7 continuous boundary transition maps with Gaussian falloffs:
- $B_{\text{subject} \leftrightarrow \text{background}}$ (Transition width: 5.2px, Uncertainty: 0.25)
- $B_{\text{skin} \leftrightarrow \text{clothing}}$ (Transition width: 4.0px, Uncertainty: 0.35)
- $B_{\text{skin} \leftrightarrow \text{hair}}$ (Transition width: 3.2px, Uncertainty: 0.40)
- $B_{\text{hair} \leftrightarrow \text{background}}$ (Transition width: 4.5px, Uncertainty: 0.20)
- $B_{\text{clothing} \leftrightarrow \text{background}}$ (Transition width: 4.2px, Uncertainty: 0.18)
- $B_{\text{lace} \leftrightarrow \text{skin}}$ (Transition width: 3.0px, Uncertainty: 0.45)
- $B_{\text{sheer} \leftrightarrow \text{skin}}$ (Transition width: 6.0px, Uncertainty: 0.55)

### Spatial Confidence Tensor:
$$C(x, y) = \text{clamp}\left(1.0 - \max_k (w_k \cdot B_k(x, y)), 0.1, 1.0\right)$$
Downstream upscaling and relighting stages will use $C(x, y)$ to attenuate hallucination intensity near uncertain boundaries.

---

## 5. VRAM Behavior & Staged Execution

- Target GPU: NVIDIA GeForce RTX 5070 (11.9 GB VRAM).
- `VRAMManager` executes perception stages sequentially using a Python context manager:
  ```python
  with vram_manager.staged_execution("stage_name"):
      # load / forward pass
  # auto-calls torch.cuda.empty_cache() & gc.collect()
  ```
- **Zero VRAM Leakage**: GPU allocations are released immediately after inference, leaving the full 12GB VRAM buffer ready for downstream image generation or restoration models.

---

## 6. Test Suite & Verification Results

### A. Numerical Anti-Bleed Test Suite ([`tests/test_enhancer_anti_bleed.py`](file:///d:/AI/imageStudio/tests/test_enhancer_anti_bleed.py)):
1. `test_subject_and_background_complementarity` $\rightarrow$ **PASSED** ($\| M_{\text{subj}} + M_{\text{bg}} - 1.0 \|_{\infty} < 0.01$)
2. `test_background_does_not_contain_majority_of_subject` $\rightarrow$ **PASSED** (Leakage $< 0.15$)
3. `test_skin_and_clothing_anti_bleed` $\rightarrow$ **PASSED** (Overlap energy $< 0.08$)
4. `test_hair_not_in_background` $\rightarrow$ **PASSED** (Hair-in-background energy $< 0.05$)
5. `test_face_strictly_inside_subject` $\rightarrow$ **PASSED** (0 pixels outside subject)
6. `test_confidence_boundary_drop` $\rightarrow$ **PASSED** (Mean confidence drops significantly at boundaries)
7. `test_masks_strictly_bounded_without_nans` $\rightarrow$ **PASSED** (All masks in $[0.0, 1.0]$ with no NaNs/Infs)
8. `test_debug_export_functionality` $\rightarrow$ **PASSED** (PNG artifacts exported cleanly)

### B. Semantic Analyzer Foundation Tests ([`tests/test_semantic_analyzer.py`](file:///d:/AI/imageStudio/tests/test_semantic_analyzer.py)):
- All 5 test cases $\rightarrow$ **PASSED**.

### C. Full Subsystem Regression Tests:
- `test_regression_suite.py`, `test_adapter_registry.py`, `test_device.py`, `test_models.py`, `test_universal_lora.py`
- **21 tests executed $\rightarrow$ 21 PASSED (0 regressions, 0 errors)**.

### D. Frontend Compilation:
- `npm run build` completed in **2.19s with 0 TypeScript/Vite errors**.

---

## 7. Known Limitations & Capabilities Status

| Capability | Status | Notes |
| :--- | :--- | :--- |
| **Foreground Matting** | `AVAILABLE` | High-precision subject/background separation with soft hair alpha |
| **Human Body Parsing** | `AVAILABLE` | Skin, hair, opaque skirt, sheer wet shirt, floral lace, accessories |
| **Face Feature Parsing** | `AVAILABLE` | High-priority face protection zone with eye/mouth landmark isolation |
| **Boundary Transitions** | `AVAILABLE` | 7 soft transition maps with uncertainty penalties |
| **Spatial Confidence** | `AVAILABLE` | Local continuous confidence map $[0.1, 1.0]$ |
| **VRAM Sequential Staging** | `AVAILABLE` | Clean CUDA cache purging between passes |
| **Depth / Normal Estimation** | `PLANNED` | Interface prepared for future optional depth map stage |

---

## 8. Exact Next Recommended Phase

**Phase 3: Detail Restoration & Material-Aware Texture Enhancement**
- Implement material-aware enhancement kernels guided by the semantic masks:
  - *Skin*: Subsurface-scattering smoothing, pore preservation, zero line-art degradation.
  - *Hair*: Directional strand sharpening along flow vectors without dark fringe bleed.
  - *Clothing*: Weave synthesis for opaque fabrics, transmittance protection for sheer shirt, pattern preservation for lace.
  - *Face Zone*: High-priority identity preservation locking facial features against AI hallucination.
