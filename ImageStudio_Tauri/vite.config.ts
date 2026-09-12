import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import fs from "fs";

function scanDir(dir: string): Array<{ name: string; path: string; size: number; architecture: "sdxl" | "flux" | "sd15" }> {
  let results: Array<{ name: string; path: string; size: number; architecture: "sdxl" | "flux" | "sd15" }> = [];
  if (!fs.existsSync(dir)) return results;
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      results = results.concat(scanDir(full));
    } else if (/\.(safetensors|ckpt|bin|pt|pth|gguf)$/i.test(e.name)) {
      // Exclude sub-parts of split models
      if (e.name.startsWith("model-") || e.name === "model.safetensors" || e.name === "diffusion_pytorch_model.safetensors") {
        continue;
      }
      const arch: "sdxl" | "flux" | "sd15" = full.toLowerCase().includes("flux")
        ? "flux"
        : full.toLowerCase().includes("sd15")
        ? "sd15"
        : "sdxl";
      results.push({ name: e.name, path: full.replace(/\\/g, "/"), size: fs.statSync(full).size, architecture: arch });
    }
  }
  return results;
}

// Custom Vite plugin to serve live filesystem models and loras to the frontend
function modelDiscoveryPlugin() {
  return {
    name: "imagestudio-model-discovery",
    configureServer(server: any) {
      server.middlewares.use((req: any, res: any, next: any) => {
        if (req.url === "/api/models") {
          const modelsDir = path.resolve(__dirname, "../models/checkpoints");
          const scanned = scanDir(modelsDir);
          const formatted = scanned.map((m) => {
            const stem = path.parse(m.name).name;
            const sizeGb = (m.size / (1024 * 1024 * 1024)).toFixed(2);
            return {
              id: stem,
              name: stem,
              filename: m.name,
              path: m.path,
              architecture: m.architecture,
              description: `${m.architecture.toUpperCase()} Checkpoint • ${sizeGb} GB`,
            };
          });
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(formatted));
          return;
        }

        if (req.url === "/api/loras") {
          const lorasDir = path.resolve(__dirname, "../models/loras");
          const scanned = scanDir(lorasDir);
          const formatted = scanned.map((l) => {
            const stem = path.parse(l.name).name;
            return {
              id: l.name,
              filename: l.name,
              displayName: stem,
              path: l.path,
              architecture: l.architecture,
              sizeBytes: l.size,
            };
          });
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(formatted));
          return;
        }

        next();
      });
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), modelDiscoveryPlugin()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 1420,
    strictPort: true,
  },
  clearScreen: false,
});
