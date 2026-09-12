# 🎨 ImageStudio AI Workstation

<div align="center">

![ImageStudio Banner](https://raw.githubusercontent.com/placeholder/imagestudio/main/docs/banner.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6.svg)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6.1-646CFF.svg)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC.svg)](https://tailwindcss.com/)

**A high-performance, modular, and VRAM-optimized generative AI creation suite designed for NVIDIA RTX GPUs.**  
Featuring dual **SDXL** & **Flux.1 (GGUF)** diffusion backends, precision **Inpainting & Anatomical Retouching**, **Multi-Character Regional Prompting**, **ControlNet Guidance**, and **StoryStudio narrative synthesis**.

[Features](#-key-features) • [Hardware Requirements](#-hardware--system-requirements) • [Quickstart](#-quickstart--installation) • [Model Directory Guide](#-model--weight-placement-guide) • [Architecture](#-architecture) • [Troubleshooting](#-troubleshooting--faq)

</div>

---

## 🌟 Key Features

### 1. 🎨 Dual-Engine Text-to-Image Diffusion
- **SDXL & Flux.1 First-Class Citizens**: Native support for SDXL `.safetensors` checkpoints and Flux.1 quantized `.gguf` (Q4/Q8/FP8) checkpoints.
- **Dynamic Checkpoint Scanner**: Auto-detects and categorizes all models in your `models/checkpoints/` directory on startup with zero phantom/hallucinated fallbacks.
- **Granular Controls**: Aspect ratio presets (`1:1`, `9:16`, `16:9`, `4:5`, `21:9`), custom dimensions, CFG scale, scheduler selection (Euler a, DPM++ 2M Karras, DDIM, UniPC), step count, and seed locking.
- **VRAM-Aware Pipeline**: Intelligent offloading, dynamic CUDA cache freeing, and GPU memory monitoring tailored for 8GB, 12GB, and 16GB+ VRAM hardware.

### 2. 🧩 Categorized Multi-LoRA Stacking Rack
- Organize and stack multiple LoRAs simultaneously with live weight sliders (`-2.0` to `+2.0`).
- LoRAs are automatically grouped by folder hierarchy:
  - `characters/` • `styles/` • `lighting/` • `poses/` • `textures/` • `concepts/`
- Independent enable/disable toggles, drag-and-drop layer reordering, and search filtering.

### 3. 👥 Multi-Character Regional Composition
- Compose multi-character scenes with **zero concept bleed**.
- Assign dedicated positive prompts, character-specific LoRAs, and spatial bounding boxes to individual characters while preserving a cohesive overall background.

### 4. 🎯 Directed Composition Studio (ControlNet & Guidance)
- **Multi-ControlNet Support**: Guide your layout and anatomy using OpenPose, Depth, Canny edge detection, and Lineart.
- **IP-Adapter & Reference Conditioning**: Transfer aesthetic style, character likeness, and composition directly from reference images.
- **PuLID Identity Retention**: Advanced facial identity preservation without overfitting.

### 5. 🖌️ Inpainting Lab
- **Interactive Mask Canvas**: Smooth HTML5 Canvas brush with dynamic brush resizing, edge feathering, and mask inversion (`inpaint painted` vs `inpaint background`).
- **Generative SDXL Inpainting**: High-fidelity prompt-guided redraw using Gaussian-feathered alpha blending and single-step Euler a scheduling to eliminate seams and blur.
- **Face-Lock**: Integrated face detection and IP-Adapter conditioning to preserve facial identity across edits.

### 6. 🩹 Precision Retouch Studio
- **Fix Artifacts (Generative Redraw)**: Target and regenerate mangled hands, fingers, limbs, eyes, or clothing using SDXL inpainting with custom prompt guidance.
- **Remove Watermark / Objects**: Cleanly erase unwanted elements, signatures, and objects using LaMa (Large Mask Inpainting).
- **Remove Background**: Instant high-precision alpha cutouts powered by Rembg (U^2-Net / BirefNet) with PNG transparency export.

### 7. 🤖 Built-In AI Prompt Generator
- Transform plain English descriptions into rich, high-detail diffusion prompts.
- Built-in prompt profiles: **JoyCaption (Descriptive Prose)**, **SDXL Base**, **Illustrious XL (Danbooru)**, **Pony Diffusion (score_9 tags)**, and **Animagine XL**.
- Connects to local LLM providers (**Ollama**, **LM Studio**, **LocalAI**) or any OpenAI-compatible endpoint.

### 8. 📖 StoryStudio & Degenerate Mode
- Multi-frame narrative and comic generation studio.
- Deconstruct stories into cinematic scene sequences, manage character catalogs, and maintain visual continuity across panels.

### 9. 🏛️ The Vault (Local Generation Ledger)
- SQLite-backed generation history with instant search, filtering, and high-resolution lightbox previews.
- **One-Click Parameter Recall**: Click "Send to T2I" to restore prompt, seed, sampler, steps, LoRAs, and resolution directly into your workspace.

---

## 💻 Hardware & System Requirements

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **OS** | Windows 10 / 11 64-bit | Windows 11 64-bit |
| **GPU** | NVIDIA RTX GPU with 8 GB VRAM | NVIDIA RTX 3060 / 4070 / 4080 / 5070 with 12 GB–16 GB+ VRAM |
| **CUDA** | CUDA 12.1+ | CUDA 12.4+ |
| **Python** | Python 3.10 or 3.11 | Python 3.10.x / 3.11.x (64-bit) |
| **Node.js** | Node.js 18 LTS | Node.js 20 LTS (with npm) |
| **Storage** | 30 GB free space (SSD strongly recommended) | 100 GB+ NVMe SSD (for model checkpoints & LoRAs) |

> [!NOTE]
> AMD ROCm users: A dedicated `requirements_ROCm.txt` is provided for Linux ROCm environments. Windows users with NVIDIA GPUs should follow the default setup instructions below.

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Logg1101/Image-Studio.git
cd Image-Studio
```

### 2. Set Up Python Virtual Environment
Open PowerShell or Command Prompt in the repository root:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# Install PyTorch with CUDA 12.4 support and all core dependencies
pip install -r requirements.txt
```

### 3. Set Up Frontend Dependencies
```bash
cd ImageStudio_Tauri
npm install
cd ..
```

### 4. Download Flux Configuration (Required for Flux GGUF)
To enable Flux.1 GGUF generation, download the lightweight configuration files (takes ~30 seconds, does not download heavy weights):
```bash
python scripts/download_configs.py
```

### 5. Launch ImageStudio
Simply run the launcher script from the root directory:
```bash
# Double-click or run from terminal:
.\run.bat
```

`run.bat` automatically:
1. Verifies the virtual environment.
2. Cleans up any stale GPU processes and frees ports `8188` and `1420`.
3. Starts the FastAPI GPU AI Engine on `http://127.0.0.1:8188`.
4. Launches the Vite development workstation at `http://localhost:1420` in your default browser.

---

## 📁 Model & Weight Placement Guide

ImageStudio uses an intuitive folder structure under `models/`. Simply download your favorite `.safetensors` or `.gguf` weights from Civitai, Hugging Face, or model repositories and place them into the designated folders:

```
imageStudio/
└── models/
    ├── checkpoints/
    │   ├── SDXL/                 <-- SDXL base & custom fine-tuned checkpoints (.safetensors)
    │   └── Flux/                 <-- Flux.1 GGUF models (.gguf) & flux-schnell-config/
    ├── loras/
    │   ├── characters/           <-- Character-specific LoRAs (.safetensors)
    │   ├── styles/               <-- Art styles, aesthetics, watercolor, cyberpunk (.safetensors)
    │   ├── lighting/             <-- Cinematic lighting, studio, volumetric (.safetensors)
    │   ├── poses/                <-- Action, dynamic pose LoRAs (.safetensors)
    │   ├── textures/             <-- Fabric, metallic, skin textures (.safetensors)
    │   └── concepts/             <-- Objects, clothing, special effects (.safetensors)
    ├── controlnet/
    │   ├── openpose/             <-- SDXL OpenPose ControlNet models (.safetensors)
    │   ├── depth/                <-- SDXL Depth ControlNet models (.safetensors)
    │   ├── canny/                <-- SDXL Canny edge ControlNet models (.safetensors)
    │   └── lineart/              <-- SDXL Lineart ControlNet models (.safetensors)
    ├── ipadapter/                <-- IP-Adapter SDXL weights (.safetensors / .bin)
    ├── clip_vision/              <-- CLIP Vision / ViT encoders for image conditioning
    ├── pulid/                    <-- PuLID identity models & EVA CLIP weights
    ├── vae/                      <-- Optional custom SDXL VAEs (e.g. sdxl_vae.safetensors)
    └── upscalers/                <-- Super-resolution ESRGAN models (.pth / .onnx)
```

### Model Category Details

| Model Type | Destination Folder | Accepted Formats | Recommended Models & Examples |
| :--- | :--- | :--- | :--- |
| **SDXL Checkpoints** | `models/checkpoints/SDXL/` (or `models/checkpoints/`) | `.safetensors` | • [Juggernaut XL](https://civitai.com/)<br>• [Animagine XL 3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1)<br>• [Illustrious XL](https://civitai.com/)<br>• [Pony Diffusion V6 XL](https://civitai.com/) |
| **Flux.1 Checkpoints** | `models/checkpoints/Flux/` | `.gguf`, `.safetensors` | • `flux1-schnell-q4_k_m.gguf`<br>• `flux1-dev-q4_k_m.gguf`<br>• `flux1-schnell-q8_0.gguf` |
| **LoRA Adapters** | `models/loras/<category>/` | `.safetensors` | • Subfolders (`characters`, `styles`, `lighting`, `poses`, `textures`, `concepts`) automatically create UI categories! |
| **ControlNet** | `models/controlnet/<type>/` | `.safetensors` | • [diffusers/controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)<br>• OpenPose XL, Depth XL |
| **IP-Adapter** | `models/ipadapter/` | `.safetensors`, `.bin` | • `ip-adapter_sdxl_vit-h.safetensors`<br>• `ip-adapter-plus_sdxl_vit-h.safetensors` |
| **CLIP Vision** | `models/clip_vision/` | `.safetensors`, `.bin` | • `CLIP-ViT-H-14-laion2B-s32B-b79K` |
| **Upscalers** | `models/upscalers/` | `.pth`, `.onnx` | • `4x-UltraSharp.pth`<br>• `4x_NMKD-Superscale-SP_178000_G.pth`<br>• RealCUGAN / ESRGAN |
| **Custom VAE** | `models/vae/` | `.safetensors`, `.pt` | • `sdxl_vae.safetensors` (Diffusers auto-loads base VAE if omitted) |

> [!TIP]
> You can create any subfolder inside `models/loras/` (e.g. `models/loras/outfits/`), and ImageStudio will automatically create that category in the LoRA selector drawer!

---

## ⚙️ Configuration & Environment

### Environment Variables (`.env`)
Create a `.env` file in the root directory if you want to configure offline operation:
```bash
# Force Hugging Face offline mode once models and tokenizers are cached
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

### AI Prompt Generator Settings
In the UI, expand the **AI Prompt Generator** card:
- **Local Ollama**: Set Provider to `OpenAI Compatible`, Base URL to `http://127.0.0.1:11434/v1`, Model to `llama3` or `joycaption`.
- **LM Studio**: Set Base URL to `http://127.0.0.1:1234/v1`.
- **Cloud Providers**: Set Provider to `OpenAI Compatible`, enter your API Key and model name.
- Click **Test Connection** to automatically fetch available models.

---

## 🏗️ Architecture

```
imageStudio/
├── run.bat                         # One-click Windows launcher & VRAM cleaner
├── requirements.txt                # Python dependencies (PyTorch CUDA 12.4)
├── main.py                         # Desktop application entrypoint
├── config/                         # Paths, resolutions, hardware defaults
├── core/                           # Device manager, memory offload, model scanning
├── engines/                        # AI generation backends
│   ├── sdxl/                       # SDXL pipeline, samplers, inpaint engine
│   ├── flux/                       # Flux.1 GGUF loader & inference engine
│   └── supir_enhancer.py           # Super-resolution & enhancement
├── adapters/                       # Universal LoRA, ControlNet & IP-Adapter handlers
├── enhancer/                       # Perception, segmentation, relighting & upscaling
├── prompt_generator/               # Heuristic & LLM prompt synthesis engine
├── StoryStudio/                    # Cinematic narrative generation engine
├── history/                        # SQLite generation metadata tracker
├── outputs/                        # Default output directories for images & metadata
└── ImageStudio_Tauri/              # Modern React + Vite + Tailwind UI
    ├── backend_bridge.py           # FastAPI server bridge on port 8188
    ├── vite.config.ts              # Vite configuration
    └── src/                        # React frontend components & pages
        ├── pages/                  # TextToImage, InpaintStudio, RetouchStudio, Vault...
        ├── components/             # LoRA rack, Canvas, Navigation, System stats
        └── services/               # API clients & backend communication
```

---

## ❓ Troubleshooting & FAQ

### 1. Port 8188 or 1420 is already in use
Running `.\run.bat` automatically checks for and terminates orphaned backend processes on ports `8188`, `1420`, and `8400`. If running manually, execute:
```powershell
for /f "tokens=5" %a in ('netstat -aon ^| findstr :8188') do taskkill /f /pid %a
for /f "tokens=5" %a in ('netstat -aon ^| findstr :1420') do taskkill /f /pid %a
```

### 2. CUDA Out of Memory (OOM)
- If generating at high resolutions (`>1024x1024`) on an 8GB VRAM card, select lower step counts (`25-30`) or use standard `1024x1024` and upscale using the **Retouch / Upscale** tab.
- For Flux models on 8GB–12GB cards, always use quantized GGUF weights (e.g. `q4_k_m`).

### 3. Models or LoRAs are not appearing in the dropdown
- Ensure your model files have valid extensions: `.safetensors` or `.gguf`.
- Check that checkpoints are inside `models/checkpoints/` (or subfolders like `models/checkpoints/SDXL/`).
- Check that LoRAs are inside `models/loras/` or category subfolders.
- Click the refresh button 🔄 next to the model selector in the UI.

### 4. Inpainting produces a blur instead of redrawing
- Make sure **Denoising Strength** in the Inpaint Studio is set between `0.75` and `0.95`.
- Ensure an informative prompt is provided (e.g. "detailed face, sharp eyes, cinematic lighting" or "slender hand with five fingers").
- Low denoise strengths (`< 0.40`) preserve existing pixels; high denoise strengths (`> 0.70`) allow generative reconstruction.

---

## 🤝 Authorship & AI Disclosure

This project was conceived, architected, and directed by **Ankit Kumar Sinha** ([@Logg1101](https://github.com/Logg1101)), who designed the product vision, creative studio workflows, pipeline orchestration, UI experience, and feature specifications.

In the spirit of open transparency: the underlying code implementation, frontend components, and backend service integrations were developed in collaborative partnership with AI coding assistants. Human vision, workflow design, and iterative testing directed the construction of the entire suite.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Black Forest Labs (Flux.1)](https://blackforestlabs.ai/)
- [Stability AI (SDXL)](https://stability.ai/)
- [Tauri](https://tauri.app/) & [Vite](https://vitejs.dev/)
- [Civitai Community](https://civitai.com/)