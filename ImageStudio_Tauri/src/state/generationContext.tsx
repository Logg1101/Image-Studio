import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from "react";
import {
  ModelInfo,
  LoraInfo,
  ActiveLora,
  AspectRatioPreset,
  GenerationResult,
  GenerationStatus,
  SystemStatus,
} from "../types";
import {
  generationService,
  ASPECT_RATIOS,
  AVAILABLE_SAMPLERS,
  AVAILABLE_SCHEDULERS,
} from "../services/generationService";

interface GenerationContextType {
  // Model & Prompt
  models: ModelInfo[];
  availableLoras: LoraInfo[];
  selectedModelId: string;
  setSelectedModelId: (id: string) => void;
  prompt: string;
  setPrompt: (p: string) => void;
  negativePrompt: string;
  setNegativePrompt: (p: string) => void;

  // LoRA State
  activeLoras: ActiveLora[];
  addLora: (lora: LoraInfo) => void;
  removeLora: (id: string) => void;
  updateLoraWeight: (id: string, weight: number) => void;
  toggleLora: (id: string) => void;
  reorderLoras: (fromIndex: number, toIndex: number) => void;

  // Sampling & Dimensions
  width: number;
  height: number;
  aspectRatio: AspectRatioPreset;
  setAspectRatio: (ratio: AspectRatioPreset) => void;
  setDimensions: (width: number, height: number) => void;
  steps: number;
  setSteps: (s: number) => void;
  cfgScale: number;
  setCfgScale: (c: number) => void;
  sampler: string;
  setSampler: (s: string) => void;
  scheduler: string;
  setScheduler: (s: string) => void;
  seed: number;
  setSeed: (s: number) => void;
  isRandomSeed: boolean;
  setIsRandomSeed: (r: boolean) => void;

  // High-Res Upscaler
  upscaleMethod: string;
  setUpscaleMethod: (m: string) => void;
  upscaleFactor: number;
  setUpscaleFactor: (f: number) => void;

  // Execution & Output State
  status: GenerationStatus;
  progressStep: number;
  progressTotal: number;
  progressMessage: string;
  result: GenerationResult | null;
  error: string | null;
  systemStatus: SystemStatus;

  // Actions
  generate: () => Promise<void>;
  cancel: () => Promise<void>;
  enhancePrompt: () => void;
  randomizePrompt: () => void;
  clearPrompt: () => void;
  refreshModelsAndLoras: () => Promise<void>;
}

const GenerationContext = createContext<GenerationContextType | undefined>(undefined);

export const GenerationProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [availableLoras, setAvailableLoras] = useState<LoraInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("");

  const [prompt, setPrompt] = useState<string>(
    "1girl, masterpiece, highly detailed, Belfast, Azur Lane, white long hair, blue eyes, maid headdress, chain choker, maid uniform, elegant, cinematic lighting"
  );
  const [negativePrompt, setNegativePrompt] = useState<string>(
    "blurry, low quality, distorted, bad anatomy, deformed, artifacts, ugly"
  );

  const [activeLoras, setActiveLoras] = useState<ActiveLora[]>([]);
  const [aspectRatio, setAspectRatioState] = useState<AspectRatioPreset>("9:16");
  const [width, setWidth] = useState<number>(768);
  const [height, setHeight] = useState<number>(1344);

  const [steps, setSteps] = useState<number>(30);
  const [cfgScale, setCfgScale] = useState<number>(7.0);
  const [sampler, setSampler] = useState<string>(AVAILABLE_SAMPLERS[0]);
  const [scheduler, setScheduler] = useState<string>(AVAILABLE_SCHEDULERS[0]);
  const [seed, setSeed] = useState<number>(12345);
  const [isRandomSeed, setIsRandomSeed] = useState<boolean>(true);

  // High-Res Upscaler State
  const [upscaleMethod, setUpscaleMethod] = useState<string>("None");
  const [upscaleFactor, setUpscaleFactor] = useState<number>(2.0);

  const [status, setStatus] = useState<GenerationStatus>("idle");
  const [progressStep, setProgressStep] = useState<number>(0);
  const [progressTotal, setProgressTotal] = useState<number>(30);
  const [progressMessage, setProgressMessage] = useState<string>("Ready");
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    gpuName: "RTX 5070",
    vramUsedGb: 0.0,
    vramTotalGb: 11.9,
    ramUsedGb: 30.5,
    ramTotalGb: 31.1,
    cpuPercent: 9,
  });

  const refreshModelsAndLoras = useCallback(async () => {
    try {
      const [mList, lList, sStatus] = await Promise.all([
        generationService.getModels(),
        generationService.getLoras(),
        generationService.getSystemStatus(),
      ]);
      setModels(mList);
      setAvailableLoras(lList);
      setSystemStatus(sStatus);
      setSelectedModelId((prev) => {
        if (prev && mList.some((m) => m.id === prev)) return prev;
        try {
          const saved = localStorage.getItem("imagestudio_selected_checkpoint");
          if (saved && mList.some((m) => m.id === saved)) return saved;
        } catch {
          // Ignored
        }
        // Always prioritize primary SDXL model (hassakuXLIllustrious_v22) or first real model
        const preferred = mList.find((m) => m.id.toLowerCase().includes("hassaku")) ||
                          mList.find((m) => m.architecture === "sdxl") ||
                          mList[0];
        const finalId = preferred ? preferred.id : "";
        try {
          if (finalId) localStorage.setItem("imagestudio_selected_checkpoint", finalId);
        } catch {}
        return finalId;
      });
    } catch (err) {
      console.error("Failed to load models & loras:", err);
    }
  }, []);

  useEffect(() => {
    refreshModelsAndLoras();
    const interval = setInterval(async () => {
      try {
        const sStatus = await generationService.getSystemStatus();
        setSystemStatus(sStatus);
      } catch {
        // Ignored
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [refreshModelsAndLoras]);

  const setAspectRatio = (ratio: AspectRatioPreset) => {
    setAspectRatioState(ratio);
    if (ASPECT_RATIOS[ratio]) {
      setWidth(ASPECT_RATIOS[ratio].width);
      setHeight(ASPECT_RATIOS[ratio].height);
    }
  };

  const setDimensions = (w: number, h: number) => {
    setWidth(w);
    setHeight(h);
  };

  const addLora = (lora: LoraInfo) => {
    setActiveLoras((prev) => {
      if (prev.some((l) => l.path === lora.path)) return prev;
      return [
        ...prev,
        {
          id: lora.id,
          filename: lora.filename,
          displayName: lora.displayName,
          path: lora.path,
          weight: 0.8,
          enabled: true,
        },
      ];
    });
  };

  const removeLora = (id: string) => {
    setActiveLoras((prev) => prev.filter((l) => l.id !== id));
  };

  const updateLoraWeight = (id: string, weight: number) => {
    setActiveLoras((prev) =>
      prev.map((l) => (l.id === id ? { ...l, weight } : l))
    );
  };

  const toggleLora = (id: string) => {
    setActiveLoras((prev) =>
      prev.map((l) => (l.id === id ? { ...l, enabled: !l.enabled } : l))
    );
  };

  const reorderLoras = (fromIndex: number, toIndex: number) => {
    setActiveLoras((prev) => {
      const next = [...prev];
      const [moved] = next.splice(fromIndex, 1);
      next.splice(toIndex, 0, moved);
      return next;
    });
  };

  const enhancePrompt = () => {
    setPrompt((prev) => {
      const p = prev.trim();
      return p
        ? `${p}, highly detailed, cinematic lighting, 8k resolution, masterwork, sharp focus`
        : "1girl, masterpiece, ultra-detailed portrait, dramatic lighting, 8k, flawless";
    });
  };

  const randomizePrompt = () => {
    const ideas = [
      "1girl, cyberpunk samurai standing in neon rain, glowing katana, intricate cybernetic armor, volumetric fog, reflection",
      "majestic ethereal dragon soaring through aurora borealis clouds, glowing crystal scales, high fantasy digital painting",
      "cosy anime cafe on a rainy autumn evening, warm glowing lanterns, coffee mug steam, gentle depth of field",
    ];
    setPrompt(ideas[Math.floor(Math.random() * ideas.length)]);
  };

  const clearPrompt = () => {
    setPrompt("");
  };

  const generate = async () => {
    if (status === "generating") return;

    setStatus("loading_model");
    setError(null);
    setProgressStep(0);
    setProgressTotal(steps);
    setProgressMessage(`Loading model checkpoint into GPU VRAM (0/${steps})...`);

    const loraMap: Record<string, number> = {};
    activeLoras.forEach((l) => {
      if (l.enabled) {
        loraMap[l.path] = l.weight;
      }
    });

    const activeSeed = isRandomSeed
      ? Math.floor(Math.random() * 2147483647)
      : seed;

    try {
      setStatus("generating");
      const res = await generationService.generate(
        {
          modelId: selectedModelId,
          prompt,
          negativePrompt: negativePrompt.trim() ? negativePrompt : undefined,
          width,
          height,
          steps,
          cfgScale,
          sampler,
          scheduler,
          seed: activeSeed,
          loras: loraMap,
          upscaleMethod,
          upscaleFactor,
        },
        (step, total, msg) => {
          setProgressStep(step);
          setProgressTotal(total || steps);
          setProgressMessage(msg);
        }
      );

      setResult(res);
      setStatus("complete");
      setProgressMessage("? Render Complete");
    } catch (err: unknown) {
      if (err instanceof Error && err.message.includes("cancelled")) {
        setStatus("stopped");
        setProgressMessage("Generation cancelled.");
      } else {
        setStatus("error");
        setError(err instanceof Error ? err.message : "Unknown generation error");
        setProgressMessage("Generation failed.");
      }
    }
  };

  const cancel = async () => {
    await generationService.cancel();
    setStatus("stopped");
    setProgressMessage("Generation cancelled.");
  };

  const handleSelectModel = (id: string) => {
    setSelectedModelId(id);
    try {
      localStorage.setItem("imagestudio_selected_checkpoint", id);
    } catch {
      // Ignored
    }
  };

  return (
    <GenerationContext.Provider
      value={{
        models,
        availableLoras,
        selectedModelId,
        setSelectedModelId: handleSelectModel,
        prompt,
        setPrompt,
        negativePrompt,
        setNegativePrompt,
        activeLoras,
        addLora,
        removeLora,
        updateLoraWeight,
        toggleLora,
        reorderLoras,
        width,
        height,
        aspectRatio,
        setAspectRatio,
        setDimensions,
        steps,
        setSteps,
        cfgScale,
        setCfgScale,
        sampler,
        setSampler,
        scheduler,
        setScheduler,
        seed,
        setSeed,
        isRandomSeed,
        setIsRandomSeed,
        upscaleMethod,
        setUpscaleMethod,
        upscaleFactor,
        setUpscaleFactor,
        status,
        progressStep,
        progressTotal,
        progressMessage,
        result,
        error,
        systemStatus,
        generate,
        cancel,
        enhancePrompt,
        randomizePrompt,
        clearPrompt,
        refreshModelsAndLoras,
      }}
    >
      {children}
    </GenerationContext.Provider>
  );
};

export const useGeneration = (): GenerationContextType => {
  const context = useContext(GenerationContext);
  if (!context) {
    throw new Error("useGeneration must be used within a GenerationProvider");
  }
  return context;
};
