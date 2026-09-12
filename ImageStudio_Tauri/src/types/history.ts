export interface HistoryRecord {
  id: number;
  model_id: string;
  architecture: string;
  variant?: string;
  prompt: string;
  negative_prompt?: string;
  seed: number;
  steps: number;
  guidance_scale: number;
  width: number;
  height: number;
  image_path: string;
  image_url: string;
  is_available: boolean;
  generation_time_ms?: number;
  vram_peak_gb?: number;
  timestamp?: string;
}

export interface HistoryResponse {
  records: HistoryRecord[];
  total: number;
  limit: number;
  offset: number;
}
