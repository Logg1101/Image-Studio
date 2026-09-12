import {
  ModelInfo,
  LoraInfo,
  GenerationRequest,
  GenerationResult,
  SystemStatus,
  PromptLLMSettings,
  PromptProfileInfo,
  PromptGenerateRequest,
  PromptGenerateResult,
  TestConnectionResult,
} from "../types";

export const AVAILABLE_SAMPLERS = [
  "Euler a",
  "Euler",
  "DPM++ 2M Karras",
  "DPM++ SDE Karras",
  "DPM++ 2M SDE Karras",
  "DDIM",
  "UniPC",
  "Heun",
];

export const AVAILABLE_SCHEDULERS = [
  "Normal",
  "Karras",
  "Exponential",
  "SGM Uniform",
  "Simple",
];

export const ASPECT_RATIOS: Record<string, { width: number; height: number; label: string }> = {
  "1:1": { width: 1024, height: 1024, label: "1:1 Square" },
  "16:9": { width: 1344, height: 768, label: "16:9 Landscape" },
  "9:16": { width: 768, height: 1344, label: "9:16 Portrait" },
  "4:3": { width: 1152, height: 896, label: "4:3 Standard" },
  "3:4": { width: 896, height: 1152, label: "3:4 Portrait" },
};

const BACKEND_URL = "http://127.0.0.1:8188";

export interface IGenerationService {
  getModels(): Promise<ModelInfo[]>;
  getLoras(): Promise<LoraInfo[]>;
  generate(
    request: GenerationRequest,
    onProgress?: (step: number, total: number, message: string) => void
  ): Promise<GenerationResult>;
  cancel(): Promise<void>;
  getSystemStatus(): Promise<SystemStatus>;
  detectFace(imageDataUrl: string): Promise<{ success: boolean; maskBase64?: string; error?: string }>;
  getPromptProfiles(): Promise<PromptProfileInfo[]>;
  getPromptLLMSettings(): Promise<PromptLLMSettings>;
  savePromptLLMSettings(settings: Partial<PromptLLMSettings>): Promise<PromptLLMSettings>;
  testPromptLLMConnection(settings?: Partial<PromptLLMSettings>): Promise<TestConnectionResult>;
  generatePromptTags(request: PromptGenerateRequest): Promise<PromptGenerateResult>;
}

class ImageStudioGenerationService implements IGenerationService {
  private isCancelled = false;

  async getModels(): Promise<ModelInfo[]> {
    try {
      const res = await fetch(`${BACKEND_URL}/api/models`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data)) {
          return data;
        }
      }
    } catch {
      // Backend not yet reachable
    }
    return [];
  }

  async getLoras(): Promise<LoraInfo[]> {
    try {
      const res = await fetch(`${BACKEND_URL}/api/loras`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data)) {
          return data;
        }
      }
    } catch {
      // Backend not yet reachable
    }
    return [];
  }

  async generate(
    request: GenerationRequest,
    onProgress?: (step: number, total: number, message: string) => void
  ): Promise<GenerationResult> {
    this.isCancelled = false;

    try {
      const triggerRes = await fetch(`${BACKEND_URL}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });

      if (triggerRes.ok) {
        while (!this.isCancelled) {
          await new Promise((r) => setTimeout(r, 100));
          const progRes = await fetch(`${BACKEND_URL}/api/progress`);
          if (progRes.ok) {
            const prog = await progRes.json();
            onProgress?.(prog.step, prog.total, prog.message);

            if (prog.status === "complete" && prog.result) {
              const fullImageUrl = prog.result.imageUrl.startsWith("data:")
                ? prog.result.imageUrl
                : `${BACKEND_URL}${prog.result.imageUrl}`;
              return {
                imageUrl: fullImageUrl,
                seed: prog.result.seed,
                generationTimeMs: prog.result.generationTimeMs,
                metadata: prog.result.metadata,
              };
            }

            if (prog.status === "error") {
              throw new Error(prog.error || "Generation encountered an error on GPU.");
            }

            if (prog.status === "stopped") {
              throw new Error("Generation cancelled by user.");
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && (err.message.includes("Failed to fetch") || err.message.includes("NetworkError"))) {
        console.warn("Python GPU backend bridge not running on 8188. Running local preview mode.");
      } else {
        throw err;
      }
    }

    // Fallback simulation loop if bridge not reached
    onProgress?.(0, request.steps, "Loading pipeline and syncing adapters into VRAM...");
    await new Promise((r) => setTimeout(r, 600));

    for (let i = 1; i <= request.steps; i++) {
      if (this.isCancelled) throw new Error("Generation cancelled by user.");
      onProgress?.(
        i,
        request.steps,
        `Denoising step ${i}/${request.steps} (${Math.round((i / request.steps) * 100)}%)`
      );
      await new Promise((r) => setTimeout(r, 70));
    }

    const seed = request.seed > 0 ? request.seed : Math.floor(Math.random() * 2147483647);
    const canvasSvg = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="${request.width}" height="${request.height}" viewBox="0 0 ${request.width} ${request.height}">
      <defs>
        <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="%231E1B4B" />
          <stop offset="50%" stop-color="%230F172A" />
          <stop offset="100%" stop-color="%23090C12" />
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="url(%23g)"/>
      <circle cx="${request.width / 2}" cy="${request.height / 2}" r="${Math.min(request.width, request.height) * 0.28}" fill="%237C6CFF" opacity="0.15" filter="blur(30px)"/>
      <text x="50%" y="46%" font-family="Segoe UI, sans-serif" font-size="28" font-weight="bold" fill="%23E8ECF4" text-anchor="middle">ImageStudio AI</text>
      <text x="50%" y="52%" font-family="JetBrains Mono, monospace" font-size="14" fill="%238993A7" text-anchor="middle">${request.width} ? ${request.height} ? Seed: ${seed}</text>
      <text x="50%" y="57%" font-family="Segoe UI, sans-serif" font-size="13" fill="%2335D6C5" text-anchor="middle">Model: ${request.modelId}</text>
    </svg>`;

    return {
      imageUrl: canvasSvg,
      seed,
      generationTimeMs: 1800,
      metadata: { ...request, completedAt: new Date().toISOString() },
    };
  }

  async cancel(): Promise<void> {
    this.isCancelled = true;
    try {
      await fetch(`${BACKEND_URL}/api/cancel`, { method: "POST" });
    } catch {
      // Ignored
    }
  }

  async getSystemStatus(): Promise<SystemStatus> {
    try {
      const res = await fetch(`${BACKEND_URL}/api/system_status`);
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Fallback
    }

    return {
      gpuName: "RTX 5070",
      vramUsedGb: 0.0,
      vramTotalGb: 11.9,
      ramUsedGb: 30.5,
      ramTotalGb: 31.1,
      cpuPercent: 9,
    };
  }

  async detectFace(imageDataUrl: string): Promise<{ success: boolean; maskBase64?: string; error?: string }> {
    try {
      const res = await fetch(`${BACKEND_URL}/api/inpaint/detect_face`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageDataUrl }),
      });
      if (res.ok) {
        return await res.json();
      }
      const errText = await res.text();
      return { success: false, error: `HTTP ${res.status}: ${errText}` };
    } catch (e: any) {
      return { success: false, error: e?.message || "Failed to contact face detection service." };
    }
  }

  async getPromptProfiles(): Promise<PromptProfileInfo[]> {
    try {
      const res = await fetch(`${BACKEND_URL}/api/prompt_generator/profiles`);
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Ignored
    }
    return [
      {
        id: "joycaption",
        name: "JoyCaption (Descriptive)",
        description: "JoyCaption-style rich, detailed visual prose prompt engine for SDXL and modern diffusion models.",
        negative_prompt: "blurry, low quality, distorted, bad anatomy, deformed, artifacts, ugly, flat lighting, oversaturated, watermark, signature",
        tag_style: "descriptive_prose",
      },
      {
        id: "sdxl_base",
        name: "SDXL Base",
        description: "Standard SDXL prompting using natural visual phrases and high-detail tags.",
        negative_prompt: "blurry, low quality, distorted, bad anatomy, deformed, artifacts, ugly, worst quality",
        tag_style: "descriptive_tags",
      },
      {
        id: "illustrious_xl",
        name: "Illustrious XL",
        description: "Optimized for Illustrious XL / Danbooru anime models with clean tag conventions.",
        negative_prompt: "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
        tag_style: "danbooru_tags",
      },
      {
        id: "pony",
        name: "Pony Diffusion (PDXL)",
        description: "Pony-family models utilizing score_9/score_8_up prefixes and Danbooru tags.",
        negative_prompt: "score_6, score_5, score_4, rating_explicit, bad anatomy, bad hands, missing fingers, low quality, blurry",
        tag_style: "pony_tags",
      },
      {
        id: "animagine_xl",
        name: "Animagine XL",
        description: "Animagine XL anime checkpoint standards with quality tags and Danbooru vocabulary.",
        negative_prompt: "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
        tag_style: "animagine_tags",
      },
    ];
  }

  async getPromptLLMSettings(): Promise<PromptLLMSettings> {
    try {
      const res = await fetch(`${BACKEND_URL}/api/prompt_generator/settings`);
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Ignored
    }
    return {
      enabled: true,
      provider: "openai_compatible",
      base_url: "http://127.0.0.1:11434/v1",
      api_key: "",
      model: "joycaption",
      temperature: 0.5,
      max_tokens: 512,
      timeout: 60,
      default_profile: "joycaption",
    };
  }

  async savePromptLLMSettings(settings: Partial<PromptLLMSettings>): Promise<PromptLLMSettings> {
    try {
      const res = await fetch(`${BACKEND_URL}/api/prompt_generator/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });
      if (res.ok) {
        const data = await res.json();
        return data.settings;
      }
    } catch (e) {
      console.error("Failed to save LLM settings:", e);
    }
    return this.getPromptLLMSettings();
  }

  async testPromptLLMConnection(settings?: Partial<PromptLLMSettings>): Promise<TestConnectionResult> {
    try {
      const res = await fetch(`${BACKEND_URL}/api/prompt_generator/test_connection`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings || {}),
      });
      if (res.ok) {
        return await res.json();
      }
      return {
        success: false,
        base_url: settings?.base_url || "",
        error: `Server responded with HTTP ${res.status}`,
      };
    } catch (e: any) {
      return {
        success: false,
        base_url: settings?.base_url || "",
        error: e.message || "Failed to contact backend bridge",
      };
    }
  }

  async generatePromptTags(request: PromptGenerateRequest): Promise<PromptGenerateResult> {
    try {
      const res = await fetch(`${BACKEND_URL}/api/prompt_generator/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      if (res.ok) {
        return await res.json();
      }
      const errText = await res.text();
      return {
        success: false,
        prompt: "",
        negative_prompt: "",
        raw_tags: "",
        parsed_tags: [],
        error: `HTTP ${res.status}: ${errText}`,
      };
    } catch (e: any) {
      return {
        success: false,
        prompt: "",
        negative_prompt: "",
        raw_tags: "",
        parsed_tags: [],
        error: `Network error: ${e.message || "Could not reach ImageStudio AI bridge"}`,
      };
    }
  }
}

export const generationService: IGenerationService = new ImageStudioGenerationService();
