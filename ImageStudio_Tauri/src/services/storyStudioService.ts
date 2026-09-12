import {
  StoryCataloguesResponse,
  CharacterItem,
  SingleGenParams,
  StoryProject
} from "../types/storyStudio";

const API_BASE = "http://127.0.0.1:8188/api/story";

export class StoryStudioService {
  async fetchCatalogues(): Promise<StoryCataloguesResponse> {
    const res = await fetch(`${API_BASE}/catalogues`);
    if (!res.ok) throw new Error("Failed to fetch catalogues");
    return res.json();
  }

  async fetchCharacters(): Promise<CharacterItem[]> {
    const res = await fetch(`${API_BASE}/characters`);
    if (!res.ok) throw new Error("Failed to fetch characters");
    return res.json();
  }

  async previewPromptAgent(params: Partial<SingleGenParams>): Promise<{ positive_prompt: string; negative_prompt: string }> {
    const res = await fetch(`${API_BASE}/prompt_agent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error("Failed to preview prompt agent");
    return res.json();
  }

  async previewStoryAgent(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
    const res = await fetch(`${API_BASE}/story_agent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to preview story agent");
    return res.json();
  }

  async generateSingle(params: SingleGenParams): Promise<{
    image_url: string;
    image_path: string;
    seed: number;
    positive_prompt: string;
    negative_prompt: string;
    generation_time_ms: number;
  }> {
    const res = await fetch(`${API_BASE}/generate_single`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Generation failed" }));
      throw new Error(err.detail || "Single generation failed");
    }
    return res.json();
  }

  async generateBatch(params: SingleGenParams & { batch_count: number }): Promise<{
    count: number;
    results: Array<{
      image_url: string;
      image_path: string;
      seed: number;
      positive_prompt: string;
      negative_prompt: string;
    }>;
  }> {
    const res = await fetch(`${API_BASE}/generate_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Batch generation failed" }));
      throw new Error(err.detail || "Batch generation failed");
    }
    return res.json();
  }

  async fetchProjects(): Promise<Array<{ name: string; character: string; line_count: number; completed_count: number; updated_at: string }>> {
    const res = await fetch(`${API_BASE}/projects`);
    if (!res.ok) throw new Error("Failed to fetch projects");
    return res.json();
  }

  async decomposeStory(payload: {
    story_text: string;
    character_id: string;
    default_settings?: Record<string, unknown>;
  }): Promise<{ scenes: Array<Record<string, unknown>>; count: number }> {
    const res = await fetch(`${API_BASE}/decompose_story`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Story breakdown failed" }));
      throw new Error(err.detail || "Story breakdown failed");
    }
    return res.json();
  }

  async createProject(data: {
    name: string;
    lines?: string[];
    scenes?: Array<Record<string, unknown>>;
    story_text?: string;
    character: string;
    default_settings?: Record<string, unknown>;
  }): Promise<StoryProject> {
    const res = await fetch(`${API_BASE}/projects/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to create project");
    return res.json();
  }

  async updateProjectScenes(
    projectName: string,
    scenes: Array<Record<string, unknown>>
  ): Promise<StoryProject> {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectName)}/update_scenes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenes }),
    });
    if (!res.ok) throw new Error("Failed to update project scenes");
    return res.json();
  }

  async getProject(name: string): Promise<StoryProject> {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error("Failed to get project");
    return res.json();
  }

  async generateStoryLine(projectName: string, lineIndex: number): Promise<{
    status: string;
    line_index: number;
    image_url: string;
    image_path: string;
    seed: number;
    directive: Record<string, unknown>;
    positive_prompt: string;
    negative_prompt: string;
  }> {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectName)}/generate_line`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ line_index: lineIndex }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Story line generation failed" }));
      throw new Error(err.detail || "Story line generation failed");
    }
    return res.json();
  }

  async upscaleImage(payload: {
    image_url: string;
    model_id?: string;
    scale?: number;
  }): Promise<{
    image_url: string;
    width: number;
    height: number;
    scale: number;
    model_id: string;
  }> {
    const res = await fetch(`${API_BASE}/upscale_image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upscale failed" }));
      throw new Error(err.detail || "Upscale failed");
    }
    return res.json();
  }

  async cancel(): Promise<void> {
    await fetch(`${API_BASE}/cancel`, { method: "POST" });
  }
}

export const storyStudioService = new StoryStudioService();
