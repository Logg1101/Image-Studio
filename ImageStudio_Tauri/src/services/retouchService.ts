/**
 * Retouch Service
 * Bridges the Retouch & Clean UI with backend inpainting, restoration, and background removal models.
 */

const BACKEND_URL = "http://127.0.0.1:8188";

export interface RetouchResponse {
  success: boolean;
  image: string; // Base64 data URL
  detail?: string;
}

export interface IRetouchService {
  removeBackground(imageSrc: string): Promise<RetouchResponse>;
  eraseObject(imageSrc: string, maskSrc: string): Promise<RetouchResponse>;
  fixArtifacts(imageSrc: string, maskSrc: string, prompt?: string): Promise<RetouchResponse>;
}

class RetouchService implements IRetouchService {
  /**
   * Delete Background (Cutout):
   * Strips the background using U-2-Net and returns an RGBA image with alpha transparency.
   */
  async removeBackground(imageSrc: string): Promise<RetouchResponse> {
    const res = await fetch(`${BACKEND_URL}/api/retouch/remove-bg`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageSrc }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.detail || "Background removal failed.");
    }
    return data;
  }

  /**
   * Erase Things / Objects:
   * Reconstructs the background under the drawn mask using LaMa structural inpainting.
   */
  async eraseObject(imageSrc: string, maskSrc: string): Promise<RetouchResponse> {
    const res = await fetch(`${BACKEND_URL}/api/retouch/erase`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageSrc, mask: maskSrc }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.detail || "Object erase operation failed.");
    }
    return data;
  }

  /**
   * Fix Artifacts & Glitches:
   * Generates anatomically correct details (hands, legs, features) via prompt-guided AI inpainting.
   */
  async fixArtifacts(imageSrc: string, maskSrc: string, prompt?: string, modelId?: string): Promise<RetouchResponse> {
    const res = await fetch(`${BACKEND_URL}/api/retouch/fix`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageSrc, mask: maskSrc, prompt, modelId }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.detail || "Fix artifacts operation failed.");
    }
    return data;
  }
}

export const retouchService = new RetouchService();
