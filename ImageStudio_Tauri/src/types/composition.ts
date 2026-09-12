export type ControlNetType = "openpose" | "depth" | "canny" | "lineart";
export type IdentityEngine = "pulid" | "ip_adapter_faceid";
export type StyleTransferMode = "style_only" | "composition_and_clothes";
export type GenerationArchitecture = "sdxl" | "flux";

export interface IdentitySettings {
  enabled: boolean;
  engine: IdentityEngine;
  strength: number; // 0.0 - 1.5 (default 0.85)
  referenceImageUrl: string | null;
  referenceImageName: string | null;
}

export interface ControlNetSettings {
  enabled: boolean;
  type: ControlNetType;
  strength: number; // 0.0 - 2.0 (default 0.75)
  referenceImageUrl: string | null;
  referenceImageName: string | null;
  preprocessedUrl: string | null;
  isPreprocessing: boolean;
  lowThreshold?: number;
  highThreshold?: number;
}

export interface StyleSettings {
  enabled: boolean;
  mode: StyleTransferMode;
  strength: number; // 0.0 - 1.5 (default 0.60)
  referenceImageUrl: string | null;
  referenceImageName: string | null;
}

export interface CompositionParams {
  prompt: string;
  negativePrompt: string;
  architecture: GenerationArchitecture;
  modelId: string;
  width: number;
  height: number;
  steps: number;
  cfgScale: number;
  seed: number;
}

export interface CompositionResult {
  imageUrl: string;
  filePath: string;
  preprocessedControlUrl?: string | null;
  seed: number;
  width: number;
  height: number;
  architecture: string;
  generationTimeMs: number;
  vramPeakMb: number;
}

export interface AvailableAdaptersResponse {
  controlnet: {
    id: string;
    name: string;
    type: string;
    is_available: boolean;
    path: string | null;
    size_mb: number;
  }[];
  ip_adapter: {
    id: string;
    name: string;
    filename: string;
    is_available: boolean;
    size_mb: number;
  }[];
  pulid: {
    id: string;
    name: string;
    architecture: string;
    is_available: boolean;
    size_mb: number;
  }[];
  supported_architectures: string[];
}
