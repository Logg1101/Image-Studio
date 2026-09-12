export interface ModelInfo {
  id: string;
  name: string;
  architecture: "sdxl" | "flux" | "sd15" | "sd3";
  variant?: string;
  format?: string;
  description?: string;
}

export interface LoraInfo {
  id: string;
  filename: string;
  displayName: string;
  category?: string;
  path: string;
  rank?: number;
  alpha?: number;
  architecture?: string;
  sizeBytes?: number;
}

export interface ActiveLora {
  id: string;
  filename: string;
  displayName: string;
  category?: string;
  path: string;
  weight: number;
  enabled: boolean;
}

export type AspectRatioPreset = "1:1" | "16:9" | "9:16" | "4:3" | "3:4";

export interface CharacterRegionPayload {
  prompt: string;
  lora_name?: string;
  lora_strength?: number;
  box: [number, number, number, number];
  feather?: number;
}

export interface GenerationRequest {
  modelId: string;
  prompt: string;
  negativePrompt?: string;
  characters?: CharacterRegionPayload[];
  width: number;
  height: number;
  steps: number;
  cfgScale: number;
  sampler: string;
  scheduler: string;
  seed: number;
  loras: Record<string, number>;
  initImage?: string;
  maskImage?: string;
  maskBlur?: number;
  denoisingStrength?: number;
  upscaleMethod?: string;
  upscaleFactor?: number;
}

export interface GenerationResult {
  imageUrl: string;
  seed: number;
  generationTimeMs: number;
  metadata?: Record<string, unknown>;
}

export type GenerationStatus =
  | "idle"
  | "loading_model"
  | "generating"
  | "complete"
  | "error"
  | "stopped";

export interface SystemStatus {
  gpuName: string;
  vramUsedGb: number;
  vramTotalGb: number;
  gpuUtilPercent?: number;
  ramUsedGb: number;
  ramTotalGb: number;
  cpuPercent: number;
}

export type NavModule = "text2img" | "multi_character" | "directed_t2i" | "degenerate" | "inpaint" | "retouch" | "vault";

export interface PromptLLMSettings {
  enabled: boolean;
  provider: string;
  base_url: string;
  api_key: string;
  model: string;
  temperature: number;
  max_tokens: number;
  timeout: number;
  default_profile: string;
}

export interface PromptProfileInfo {
  id: string;
  name: string;
  description: string;
  negative_prompt: string;
  tag_style: string;
}

export interface PromptGenerateRequest {
  text: string;
  profile?: string;
  style?: string;
  existing_prompt?: string;
  mode?: "replace" | "append";
  settings?: Partial<PromptLLMSettings>;
}

export interface PromptGenerateResult {
  success: boolean;
  prompt: string;
  negative_prompt: string;
  raw_tags: string;
  parsed_tags: string[];
  error?: string;
}

export interface TestConnectionResult {
  success: boolean;
  base_url: string;
  models?: string[];
  message?: string;
  error?: string;
}

export * from "./storyStudio";


