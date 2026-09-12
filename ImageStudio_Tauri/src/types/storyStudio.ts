export interface StoryCatalogueItem {
  id: string;
  name: string;
  prompt?: string;
  prompt_context?: string;
  description?: string;
  width?: number;
  height?: number;
  aspect_ratio?: string;
  is_default?: boolean;
}

export interface CharacterItem {
  id: string;
  name: string;
  lora: string;
  lora_relative_path: string;
  lora_absolute_path: string;
  trigger: string;
  default_weight: number;
  description?: string;
  personality?: string;
  default_clothing?: string;
}

export interface StoryCataloguesResponse {
  clothes_types: StoryCatalogueItem[];
  clothes_colors: StoryCatalogueItem[];
  clothes_details: StoryCatalogueItem[];
  lighting: StoryCatalogueItem[];
  expressions: StoryCatalogueItem[];
  poses: StoryCatalogueItem[];
  povs: StoryCatalogueItem[];
  bondage_types: StoryCatalogueItem[];
  bondage_styles: StoryCatalogueItem[];
  bondage_hands: StoryCatalogueItem[];
  bondage_legs: StoryCatalogueItem[];
  bondage_accessories: StoryCatalogueItem[];
  gag_types: StoryCatalogueItem[];
  scenes: StoryCatalogueItem[];
  styles: StoryCatalogueItem[];
  resolutions: StoryCatalogueItem[];
  upscalers: string[];
  samplers: string[];
  schedulers: string[];
  default_sampler: string;
  default_scheduler: string;
  default_steps: number;
  default_cfg: number;
}

export interface SingleGenParams {
  character_id: string;
  character_weight?: number;
  expression?: string;
  clothes_type: string;
  clothes_color: string;
  clothes_details: string[];
  lighting?: string;
  pose: string;
  pov: string;
  bondage_type?: string;
  bondage_style?: string;
  bondage_hands?: string;
  bondage_legs?: string;
  bondage_accessories?: string;
  gag_type?: string;
  scene: string;
  style?: string;
  positive_prompt?: string;
  negative_prompt?: string;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
  seed: number;
  batch_count?: number;
  upscale_method?: string;
  upscale_factor?: number;
}

export interface StoryLineItem {
  index: number;
  text: string;
  positive_prompt?: string;
  negative_prompt?: string;
  directive?: Record<string, unknown>;
  status: "Pending" | "Generating" | "Completed" | "Failed" | "Stopped";
  image_filename?: string;
  image_url?: string;
  prompt_data?: {
    positive_prompt?: string;
    negative_prompt?: string;
    seed?: number;
    directive?: Record<string, unknown>;
  };
  seed?: number;
  error?: string;
}

export interface StoryProject {
  name: string;
  character: string;
  default_settings: {
    clothes_type?: string;
    clothes_color?: string;
    style?: string;
    width?: number;
    height?: number;
    steps?: number;
    cfg?: number;
    sampler?: string;
    scheduler?: string;
    seed?: number;
  };
  lines: StoryLineItem[];
  continuity?: Record<string, unknown>;
  updated_at?: string;
}

export interface GenerationProgress {
  current: number;
  total: number;
  message: string;
  isGenerating: boolean;
}
