# Semantic Image Analysis Foundation (Phase 1)

This document describes the architecture, hierarchical data representation, boundary defense mechanics, and integration interface for the **Enhancer & Re-Light Semantic Analysis Subsystem**.

---

## 1. Overview & Semantic Hierarchy

The semantic analysis subsystem decomposes a 2D source image into a multi-tier physical and anatomical hierarchy rather than treating it as a flat RGB raster.

```
IMAGE ROOT
├── BACKGROUND (Material: Architectural Diffuse / Planar)
│   ├── Ceiling & Lamps (Light emitter)
│   ├── Planar Walls
│   └── Door & Architrave Moldings
│
└── SUBJECT (Material: Organic Skin / Textiles / Anisotropic Hair)
    ├── SKIN & ANATOMY (Subsurface scattering, moisture sheen)
    │   ├── Face (Protected Priority: HIGH)
    │   ├── Neck & Collarbone
    │   ├── Arms & Hands
    │   └── Thighs & Legs
    │
    ├── HAIR (Anisotropic strand continuity)
    │   ├── Hair Mass Body
    │   └── Fine Bangs & Stray Wisps
    │
    ├── CLOTHING & TEXTILES (Material-specific physics)
    │   ├── Opaque Fabric (Skirt, collar, cuffs)
    │   ├── Translucent / Sheer Cling (Wet white shirt over undergarment)
    │   ├── Open-Weave Lace (Stocking tops floral patterns)
    │   └── Seams & Stitches (Placket, buttons, hem)
    │
    └── ACCESSORIES & HARDWARE (High specular / crisp geometry)
        ├── Ribbons & Bows
        └── Garter Straps & Metallic Chrome Buckles
```

---

## 2. Mathematical Representation

### A. Soft & Probabilistic Masks
Every semantic layer is represented as a single-channel spatial floating-point probability map:
$$M_{\text{region}}(x, y) \in [0.0, 1.0]$$
- Values of `1.0` represent complete membership certainty.
- Values in `(0.0, 1.0)` represent sub-pixel boundaries, fine hair wisps, open lace holes, or translucent fabric layers.
- Overlapping layers are permitted (e.g. $M_{\text{skin}} \cap M_{\text{sheer\_shirt}} > 0$).

### B. Boundary & Transition Zones
Transition zones between competing materials are computed via morphological dilation intersection and Gaussian falloff:
$$B_{A \leftrightarrow B} = \mathcal{G}_{\sigma}\left( \text{Dilate}(M_A > 0.3, r) \cap \text{Dilate}(M_B > 0.3, r) \right)$$
- `transition_width_px`: The physical pixel thickness of the boundary.
- `uncertainty_score`: Ambiguity metric ($0.0 \rightarrow 1.0$).

### C. Spatial Confidence Map
The confidence tensor penalizes edge ambiguities and high-entropy transitions:
$$C(x, y) = 1.0 - \max_k \left( w_k \cdot B_k(x, y) \right)$$
Downstream restoration stages attenuate their hallucination strength where $C(x, y)$ drops.

---

## 3. Analyzer Interface Contract

All analyzers implement the abstract base contract:

```python
from abc import ABC, abstractmethod
from PIL import Image
from enhancer.types import SemanticAnalysisResult

class BaseSemanticAnalyzer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @abstractmethod
    def analyze(self, image: Image.Image) -> SemanticAnalysisResult:
        """Parses image and returns structured regions, boundaries, and confidence."""
        pass
```

### Result Schema (`SemanticAnalysisResult`)
- `image`: `{ width, height, aspect_ratio, channels, color_space }`
- `regions`: Dictionary of `SemanticRegion` objects (mask data URL, coverage %, confidence, material type, protection status, sub-regions).
- `boundaries`: Dictionary of `BoundaryTransition` objects (mask data URL, transition width, uncertainty score).
- `surface_modifiers`: Dictionary of `SurfaceModifier` objects (water droplets / specular highlights).
- `global_confidence`: Overall reliability float ($0.0 - 1.0$).
- `confidence_map`: PNG Base64 spatial confidence heatmap.
- `composite_preview`: Color-coded semantic segmentation visualizer.

---

## 4. Subsystem Isolation & Tauri Communication

```
[ Tauri Desktop UI (React / TypeScript) ]
  ├── Page: EnhanceStudio (src/pages/EnhanceStudio.tsx)
  ├── Context: EnhancerProvider (src/state/enhancerContext.tsx)
  ├── Diagnostics: SemanticDiagnosticsPanel (src/components/enhancer/SemanticDiagnosticsPanel.tsx)
  └── Canvas: ComparisonViewport (src/components/enhancer/ComparisonViewport.tsx)
              │
              ▼ HTTP REST (Port 8188)
[ Isolated Router: enhancer/router.py ]
  ├── POST /api/enhancer/analyze -> Analyzes image via SemanticAnalyzer
  └── GET  /api/enhancer/health  -> Returns analyzer ready status
              │
              ▼
[ Core Module: enhancer/ ]
  ├── enhancer/analyzer/semantic_analyzer.py
  ├── enhancer/analyzer/base.py
  └── enhancer/types.py
```

- **Zero Coupling**: The Image Generation backend (`core/generation.py`, `SDXLEngine`, `FluxEngine`, `/api/generate`) is completely untouched and unaffected.
- **Standby Fallback**: If the Python bridge is restarting or offline, the Tauri frontend utilizes a client fallback parser to maintain 60 FPS responsiveness.
