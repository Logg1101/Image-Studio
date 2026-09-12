# ImageStudio — Existing Architecture & Tauri Frontend Integration Notes

## 1. Overview
This document records the architectural inspection of the existing ImageStudio Python AI engine, establishing a clean interface contract for the new Tauri + React + TypeScript desktop frontend.

---

## 2. Python Generation Backend Entry Points
* **Generation Coordinator**: `core.generation.GenerationCoordinator` (`core/generation.py`)
  * Dynamically accepts `GenerationRequest` objects.
  * Dispatches to `SDXLEngine` (`engines/sdxl/engine.py`) or `FluxEngine` (`engines/flux/engine.py`).
  * Handles multi-LoRA synchronization (`adapters.registry.adapter_registry.sync_loras()`).
  * Persists generation history and metadata via `history.manager.HistoryManager`.
  * Frees CUDA memory upon completion via `core.memory.clear_vram()`.

---

## 3. Model & Checkpoint Discovery
* **Discovery Modules**: `core.model_manager.ModelManager` (`core/model_manager.py`) & `core.models.detector.ModelDetector` (`core/models/detector.py`).
* **Path**: `config.paths.MODELS_DIR` (`D:/AI/ImageStudio/models/checkpoints`).
* **Structure**: `ModelInfo` dataclass (`core/types.py`):
  * `id`: Unique identifier/display name (e.g. `animpro-Q5_K_M`, `sd_xl_base_1.0`).
  * `architecture`: Architecture tag (`sdxl`, `flux`, `sd15`).
  * `variant`: Precision/quantization (`fp16`, `bf16`, `q5_k_m`).
  * `transformer_path` / `text_encoder_paths` / `vae_path` / `config_path`.

---

## 4. LoRA Discovery & Parameter Representation
* **Discovery Modules**: `core.lora_manager.LoraManager` (`core/lora_manager.py`) & `adapters.registry.AdapterRegistry` (`adapters/registry.py`).
* **Path**: `config.paths.LORAS_DIR` (`D:/AI/ImageStudio/models/loras`).
* **File Extensions**: `.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`.
* **LoRA Payload**: `Dict[str, float]` mapping absolute file path string to float weight `[-2.0, 2.0]`.

---

## 5. Generation Request & Parameter Representation
* **Structure**: `GenerationRequest` dataclass (`core/types.py`):
  * `model: ModelInfo`
  * `prompt: str` (multiline positive conditioning)
  * `negative_prompt: Optional[str]`
  * `width: int` (default `1024`, multiple of 8)
  * `height: int` (default `1024`, multiple of 8)
  * `steps: int` (default `28` - `30`)
  * `guidance_scale: float` (CFG, default `7.0`)
  * `sampler: str` (e.g. `Euler a`, `DPM++ 2M Karras`, `Euler`)
  * `scheduler: str` (e.g. `Normal`, `Karras`, `Exponential`, `SGM Uniform`)
  * `seed: int` (32-bit signed integer `[0, 2147483647]`, sanitized by `sanitize_seed`)
  * `loras: Dict[str, float]` (active LoRA map)
  * `upscale_method: str` (`None`, `Lanczos`, `Bicubic`)
  * `upscale_factor: float` (`1.5`, `2.0`, `3.0`, `4.0`)
  * `step_callback: Optional[Callable[[int, int], None]]`

---

## 6. Output Generation & Persistence
* **Structure**: `GenerationResult` dataclass (`core/types.py`):
  * `image_path: str` (saved in `outputs/` or `outputs/Projects/<name>/`)
  * `metadata_path: str` (JSON metadata with embedded seed, prompt, and adapter loadouts)
  * `generation_time_ms: float`
  * `seed: int`
* **Metadata Embedding**: Stored in PNG text chunks (`parameters` JSON payload) for lossless inspection.

---

## 7. Tauri Frontend Interface Abstraction (`GenerationService`)
The React / TypeScript frontend defines an abstract service layer `IGenerationService` with concrete mock & IPC implementations:
* `getModels(): Promise<ModelInfo[]>`
* `getLoras(): Promise<LoraInfo[]>`
* `generate(request: GenerationRequest): Promise<GenerationResult>`
* `cancel(): Promise<void>`
* `getSystemStatus(): Promise<SystemStatus>`
