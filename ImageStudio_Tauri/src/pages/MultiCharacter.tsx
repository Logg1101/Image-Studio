import React, { useState } from "react";
import {
  Users,
  Play,
  Square,
  RefreshCw,
  AlertCircle,
  Download,
  Sliders,
  Brain,
  ChevronDown,
  ChevronUp,
  Image as ImageIcon,
  Sparkles,
} from "lucide-react";
import { useGeneration } from "../state/generationContext";
import {
  AVAILABLE_SAMPLERS,
  AVAILABLE_SCHEDULERS,
  ASPECT_RATIOS,
  generationService,
} from "../services/generationService";
import { AspectRatioPreset, CharacterRegionPayload, GenerationResult } from "../types";

const CHAR_THEMES = [
  { name: "Character 1", border: "border-blue-500", text: "text-blue-400", bg: "bg-blue-500/10", fill: "rgba(59, 130, 246, 0.25)", solid: "#3B82F6" },
  { name: "Character 2", border: "border-emerald-500", text: "text-emerald-400", bg: "bg-emerald-500/10", fill: "rgba(16, 185, 129, 0.25)", solid: "#10B981" },
  { name: "Character 3", border: "border-amber-500", text: "text-amber-400", bg: "bg-amber-500/10", fill: "rgba(245, 158, 11, 0.25)", solid: "#F59E0B" },
  { name: "Character 4", border: "border-purple-500", text: "text-purple-400", bg: "bg-purple-500/10", fill: "rgba(168, 85, 247, 0.25)", solid: "#A855F7" },
];

interface CharacterState {
  prompt: string;
  loraName: string;
  loraStrength: number;
  regionPreset: "auto" | "left" | "right" | "left13" | "center13" | "right13" | "custom";
  customBox: [number, number, number, number];
}

const DEFAULT_PROMPTS = [
  "1girl, silver hair, ponytail, emerald eyes, white royal sorceress robe, gold ornaments, holding glowing staff, standing elegantly",
  "1girl, long dark crimson hair, amber eyes, black gothic knight armor, ornate greatsword on back, confident expression",
  "1boy, messy brown hair, blue eyes, adventurer tunic, leather armor, smiling friendly",
  "1girl, blonde twin braids, violet eyes, celestial priestess gown, floating ribbon, gentle smile",
];

export const MultiCharacter: React.FC = () => {
  const {
    models,
    availableLoras,
    selectedModelId,
    setSelectedModelId,
    width,
    height,
    aspectRatio,
    setAspectRatio,
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
    upscaleMethod,
    upscaleFactor,
    refreshModelsAndLoras,
  } = useGeneration();

  // Multi-Character Local State
  const [globalPrompt, setGlobalPrompt] = useState<string>(
    "masterpiece, highly detailed, fantasy garden with ancient stone ruins, blooming wisteria, golden hour sunlight, volumetric god rays"
  );
  const [negativePrompt, setNegativePrompt] = useState<string>(
    "blurry, low quality, distorted anatomy, duplicate characters, fused bodies, extra limbs, bad hands, mutated fingers"
  );
  const [showNegPrompt, setShowNegPrompt] = useState<boolean>(false);
  const [characterCount, setCharacterCount] = useState<number>(2);

  const [characters, setCharacters] = useState<CharacterState[]>([
    {
      prompt: DEFAULT_PROMPTS[0],
      loraName: "",
      loraStrength: 0.8,
      regionPreset: "auto",
      customBox: [0.0, 0.0, 0.5, 1.0],
    },
    {
      prompt: DEFAULT_PROMPTS[1],
      loraName: "",
      loraStrength: 0.8,
      regionPreset: "auto",
      customBox: [0.5, 0.0, 1.0, 1.0],
    },
    {
      prompt: DEFAULT_PROMPTS[2],
      loraName: "",
      loraStrength: 0.8,
      regionPreset: "auto",
      customBox: [0.66, 0.0, 1.0, 1.0],
    },
    {
      prompt: DEFAULT_PROMPTS[3],
      loraName: "",
      loraStrength: 0.8,
      regionPreset: "auto",
      customBox: [0.75, 0.0, 1.0, 1.0],
    },
  ]);

  // Generation status
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [progressStep, setProgressStep] = useState<number>(0);
  const [progressTotal, setProgressTotal] = useState<number>(30);
  const [progressMessage, setProgressMessage] = useState<string>("");
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [genError, setGenError] = useState<string | null>(null);

  // Helper for computing active region box per character
  const getComputedBox = (char: CharacterState, index: number, total: number): [number, number, number, number] => {
    if (char.regionPreset === "custom") {
      return char.customBox;
    }
    if (char.regionPreset === "left") return [0.0, 0.0, 0.5, 1.0];
    if (char.regionPreset === "right") return [0.5, 0.0, 1.0, 1.0];
    if (char.regionPreset === "left13") return [0.0, 0.0, 0.3333, 1.0];
    if (char.regionPreset === "center13") return [0.3333, 0.0, 0.6667, 1.0];
    if (char.regionPreset === "right13") return [0.6667, 0.0, 1.0, 1.0];

    // Default auto column partition
    if (total === 2) {
      return index === 0 ? [0.0, 0.0, 0.5, 1.0] : [0.5, 0.0, 1.0, 1.0];
    } else if (total === 3) {
      const step = 1.0 / 3.0;
      return [index * step, 0.0, (index + 1) * step, 1.0];
    } else {
      const step = 0.25;
      return [index * step, 0.0, (index + 1) * step, 1.0];
    }
  };

  const handleSetCount = (newCount: number) => {
    setCharacterCount(newCount);
  };

  const handleUpdateChar = (idx: number, patch: Partial<CharacterState>) => {
    setCharacters((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], ...patch };
      return next;
    });
  };

  const handleGenerate = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    setGenError(null);
    setProgressStep(0);
    setProgressTotal(steps);
    setProgressMessage("Initializing multi-character regional pipeline on GPU...");

    try {
      const charPayloads: CharacterRegionPayload[] = characters.slice(0, characterCount).map((c, i) => {
        const box = getComputedBox(c, i, characterCount);
        return {
          prompt: c.prompt.trim(),
          lora_name: c.loraName || undefined,
          lora_strength: c.loraStrength,
          box,
          feather: 0.04,
        };
      });

      const actualSeed = isRandomSeed ? Math.floor(Math.random() * 2147483647) : seed;

      const res = await generationService.generate(
        {
          modelId: selectedModelId,
          prompt: globalPrompt.trim() || "masterpiece, high quality",
          negativePrompt: negativePrompt.trim() || undefined,
          characters: charPayloads,
          width,
          height,
          steps,
          cfgScale,
          sampler,
          scheduler,
          seed: actualSeed,
          loras: {},
          upscaleMethod: upscaleMethod === "None" ? undefined : upscaleMethod,
          upscaleFactor: upscaleFactor,
        },
        (step, total, msg) => {
          setProgressStep(step);
          if (total > 0) setProgressTotal(total);
          if (msg) setProgressMessage(msg);
        }
      );

      setResult(res);
      if (isRandomSeed) {
        setSeed(res.seed);
      }
    } catch (err: any) {
      setGenError(err?.message || "Failed to generate multi-character image.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCancel = async () => {
    try {
      await generationService.cancel();
    } catch {
      // Ignored
    }
  };

  const handleDownload = () => {
    if (!result?.imageUrl) return;
    const a = document.createElement("a");
    a.href = result.imageUrl;
    a.download = `multi_character_${result.seed || Date.now()}.png`;
    a.click();
  };

  const activeModel = models.find((m) => m.id === selectedModelId);
  const percent =
    progressStep > 0 && progressTotal > 0
      ? Math.min(100, Math.round((progressStep / progressTotal) * 100))
      : 0;

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0B0E14] text-[#E2E8F0] overflow-hidden">
      {/* Top Header Bar */}
      <div className="h-11 px-5 bg-[#0D1117] border-b border-[#21262D] flex items-center justify-between shrink-0 select-none">
        <div className="flex items-center space-x-3">
          <Users className="w-4 h-4 text-purple-400" />
          <span className="font-bold text-xs tracking-wide bg-gradient-to-r from-purple-400 to-indigo-300 bg-clip-text text-transparent">
            MULTI-CHARACTER STUDIO
          </span>
          <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-purple-950/60 text-purple-300 border border-purple-800/50">
            Regional Anti-Bleed
          </span>
        </div>

        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="text-[#8B949E] text-[11px]">Checkpoint:</span>
          <span className="text-white font-semibold px-2 py-0.5 rounded bg-[#161B22] border border-[#30363D]">
            {activeModel?.name || selectedModelId || "SDXL"}
          </span>
          {activeModel?.architecture && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-950/60 border border-purple-800/40 text-purple-300 font-bold uppercase">
              {activeModel.architecture}
            </span>
          )}
        </div>
      </div>

      {/* Error notification banner */}
      {genError && (
        <div className="px-6 py-2 bg-red-950/80 border-b border-red-800/80 text-red-200 text-xs flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
            <span>{genError}</span>
          </div>
        </div>
      )}

      {/* Main Workspace Body */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Left Control Column */}
        <div className="w-[540px] border-r border-[#21262D] bg-[#0D1117] flex flex-col h-full overflow-y-auto custom-scrollbar p-4 space-y-3 shrink-0">
          {/* 1. Checkpoint Selector */}
          <div className="space-y-1.5 p-3 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-bold text-purple-300 uppercase tracking-wider">
                <Brain className="w-4 h-4 text-purple-400" />
                <span>Model Checkpoint</span>
              </div>
              <button
                type="button"
                onClick={() => refreshModelsAndLoras()}
                className="p-1 rounded text-purple-400 hover:text-purple-200 hover:bg-purple-900/40 transition flex items-center space-x-1 text-[10px]"
              >
                <RefreshCw className="w-3 h-3" />
                <span>Rescan</span>
              </button>
            </div>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500 font-medium cursor-pointer"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} [{m.architecture?.toUpperCase() || "MODEL"}]
                </option>
              ))}
            </select>
          </div>

          {/* 2. Global Scene Prompt */}
          <div className="space-y-1.5 p-3 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-[#E2E8F0] tracking-wide flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                <span>GLOBAL SCENE PROMPT (Shared Environment)</span>
              </span>
              <span className="text-[10px] font-mono text-[#8B949E]">
                {globalPrompt.length} chars
              </span>
            </div>
            <textarea
              rows={2}
              value={globalPrompt}
              onChange={(e) => setGlobalPrompt(e.target.value)}
              placeholder="Describe background scene, environment, lighting, weather, and composition..."
              className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg p-2.5 text-xs text-white placeholder-[#484F58] focus:outline-none focus:border-purple-500 resize-none font-sans"
            />

            {/* Negative Prompt Toggle */}
            <div className="pt-1">
              <button
                type="button"
                onClick={() => setShowNegPrompt(!showNegPrompt)}
                className="text-[11px] text-[#8B949E] hover:text-[#E2E8F0] flex items-center space-x-1 font-medium"
              >
                {showNegPrompt ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                <span>Negative Prompt</span>
              </button>
              {showNegPrompt && (
                <textarea
                  rows={2}
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  placeholder="Elements to suppress..."
                  className="w-full mt-1.5 bg-[#0D1117] border border-[#30363D] rounded-lg p-2 text-xs text-white placeholder-[#484F58] focus:outline-none focus:border-purple-500 resize-none font-sans"
                />
              )}
            </div>
          </div>

          {/* 3. Character Count Selector */}
          <div className="p-3 bg-[#161B22] rounded-xl border border-[#30363D] space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-1.5">
                <Users className="w-4 h-4 text-purple-400" />
                <span>Character Count</span>
              </span>
              <span className="text-[10px] font-mono text-purple-300">
                {characterCount} Active Characters
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2">
              {[2, 3, 4].map((cnt) => (
                <button
                  key={cnt}
                  type="button"
                  onClick={() => handleSetCount(cnt)}
                  className={`py-1.5 text-xs font-bold rounded-lg border transition-all ${
                    characterCount === cnt
                      ? "bg-purple-600 border-purple-500 text-white shadow-md shadow-purple-900/30"
                      : "bg-[#0D1117] border-[#30363D] text-[#8B949E] hover:text-white hover:border-[#484F58]"
                  }`}
                >
                  {cnt} Characters
                </button>
              ))}
            </div>

            {/* Visual Canvas Mini-Map */}
            <div className="mt-2 pt-2 border-t border-[#21262D]">
              <div className="text-[10px] text-[#8B949E] font-medium mb-1 flex items-center justify-between">
                <span>Spatial Canvas Layout (Preview):</span>
                <span className="font-mono text-[9px]">{width} × {height}</span>
              </div>
              <div className="relative w-full h-14 bg-[#090C12] border border-[#30363D] rounded-lg overflow-hidden flex items-center">
                {characters.slice(0, characterCount).map((char, i) => {
                  const box = getComputedBox(char, i, characterCount);
                  const theme = CHAR_THEMES[i % CHAR_THEMES.length];
                  const leftPct = `${box[0] * 100}%`;
                  const widthPct = `${(box[2] - box[0]) * 100}%`;

                  return (
                    <div
                      key={i}
                      style={{
                        left: leftPct,
                        width: widthPct,
                        backgroundColor: theme.fill,
                        borderColor: theme.solid,
                      }}
                      className="absolute top-1 bottom-1 border rounded flex flex-col items-center justify-center transition-all duration-200"
                    >
                      <span className="text-[10px] font-bold text-white shadow-sm">
                        Char {i + 1}
                      </span>
                      {char.loraName && (
                        <span className="text-[8px] font-mono text-purple-200 px-1 bg-purple-950/70 rounded">
                          LoRA
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* 4. Character Cards Stack */}
          <div className="space-y-2.5">
            {characters.slice(0, characterCount).map((char, idx) => {
              const theme = CHAR_THEMES[idx % CHAR_THEMES.length];
              return (
                <div
                  key={idx}
                  className={`p-3 bg-[#161B22] rounded-xl border ${theme.border} ${theme.bg} space-y-2 transition-all`}
                >
                  {/* Card Header */}
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-bold ${theme.text} flex items-center space-x-1.5`}>
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: theme.solid }} />
                      <span>{theme.name}</span>
                    </span>

                    <div className="flex items-center space-x-1.5">
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#0D1117] text-purple-300 border border-[#30363D]">
                        X: {Math.round(getComputedBox(char, idx, characterCount)[0] * 100)}% – {Math.round(getComputedBox(char, idx, characterCount)[2] * 100)}%
                      </span>
                      <span className="text-[10px] text-[#8B949E]">Region:</span>
                      <select
                        value={char.regionPreset}
                        onChange={(e) =>
                          handleUpdateChar(idx, { regionPreset: e.target.value as any })
                        }
                        className="bg-[#0D1117] border border-[#30363D] text-[10px] text-white rounded px-2 py-0.5 focus:outline-none focus:border-purple-500 cursor-pointer font-mono"
                      >
                        <option value="auto">Auto Column</option>
                        <option value="left">Left Half (0% - 50%)</option>
                        <option value="right">Right Half (50% - 100%)</option>
                        <option value="left13">Left 1/3 (0% - 33%)</option>
                        <option value="center13">Center 1/3 (33% - 67%)</option>
                        <option value="right13">Right 1/3 (67% - 100%)</option>
                        <option value="custom">Custom Sliders</option>
                      </select>
                    </div>
                  </div>

                  {/* Character Prompt Text */}
                  <textarea
                    rows={2}
                    value={char.prompt}
                    onChange={(e) => handleUpdateChar(idx, { prompt: e.target.value })}
                    placeholder={`Prompt for ${theme.name} (e.g. 1girl, silver hair, red coat)...`}
                    className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg p-2 text-xs text-white placeholder-[#484F58] focus:outline-none focus:border-purple-500 resize-none font-sans"
                  />

                  {/* LoRA & Weight Controls */}
                  <div className="flex items-center space-x-2 pt-0.5">
                    <div className="flex-1">
                      <select
                        value={char.loraName}
                        onChange={(e) => handleUpdateChar(idx, { loraName: e.target.value })}
                        className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1 text-[11px] text-white focus:outline-none focus:border-purple-500 cursor-pointer"
                      >
                        <option value="">No LoRA (Regional Prompt Only)</option>
                        {availableLoras.map((l) => (
                          <option key={l.id} value={l.displayName}>
                            🧩 {l.displayName} [{l.category || "LoRA"}]
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="flex items-center space-x-1 shrink-0">
                      <span className="text-[10px] text-[#8B949E]">Weight:</span>
                      <input
                        type="number"
                        step={0.05}
                        min={-2.0}
                        max={2.0}
                        value={char.loraStrength}
                        onChange={(e) =>
                          handleUpdateChar(idx, { loraStrength: parseFloat(e.target.value) || 0.8 })
                        }
                        className="w-14 bg-[#0D1117] border border-[#30363D] rounded px-1.5 py-0.5 text-[11px] text-white text-center focus:outline-none focus:border-purple-500 font-mono"
                      />
                    </div>
                  </div>

                  {/* Custom Region Position Controls if "custom" selected */}
                  {char.regionPreset === "custom" && (
                    <div className="p-2 bg-[#090C12] rounded border border-[#30363D] space-y-1 text-[10px]">
                      <div className="flex items-center justify-between">
                        <span>X-Start: {Math.round(char.customBox[0] * 100)}%</span>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={Math.round(char.customBox[0] * 100)}
                          onChange={(e) => {
                            const val = parseInt(e.target.value) / 100;
                            handleUpdateChar(idx, {
                              customBox: [val, char.customBox[1], Math.max(val + 0.05, char.customBox[2]), char.customBox[3]],
                            });
                          }}
                          className="w-28 accent-purple-500"
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <span>X-End: {Math.round(char.customBox[2] * 100)}%</span>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={Math.round(char.customBox[2] * 100)}
                          onChange={(e) => {
                            const val = parseInt(e.target.value) / 100;
                            handleUpdateChar(idx, {
                              customBox: [Math.min(char.customBox[0], val - 0.05), char.customBox[1], val, char.customBox[3]],
                            });
                          }}
                          className="w-28 accent-purple-500"
                        />
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* 5. Shared Sampling Settings */}
          <div className="p-3 bg-[#161B22] rounded-xl border border-[#30363D] space-y-2.5">
            <div className="flex items-center space-x-2 text-xs font-bold text-white uppercase tracking-wider">
              <Sliders className="w-4 h-4 text-purple-400" />
              <span>Generation Settings</span>
            </div>

            {/* Aspect Ratio Presets */}
            <div className="grid grid-cols-5 gap-1.5">
              {(Object.keys(ASPECT_RATIOS) as AspectRatioPreset[]).map((ar) => (
                <button
                  key={ar}
                  type="button"
                  onClick={() => setAspectRatio(ar)}
                  className={`py-1 text-[11px] font-bold rounded border transition ${
                    aspectRatio === ar
                      ? "bg-purple-600 border-purple-500 text-white"
                      : "bg-[#0D1117] border-[#30363D] text-[#8B949E] hover:text-white"
                  }`}
                >
                  {ar}
                </button>
              ))}
            </div>

            {/* Steps & CFG */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <div className="flex justify-between text-[11px] text-[#8B949E] mb-1">
                  <span>Steps:</span>
                  <span className="font-mono text-white">{steps}</span>
                </div>
                <input
                  type="range"
                  min={15}
                  max={50}
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value))}
                  className="w-full accent-purple-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-[11px] text-[#8B949E] mb-1">
                  <span>CFG Scale:</span>
                  <span className="font-mono text-white">{cfgScale}</span>
                </div>
                <input
                  type="range"
                  min={2}
                  max={12}
                  step={0.5}
                  value={cfgScale}
                  onChange={(e) => setCfgScale(parseFloat(e.target.value))}
                  className="w-full accent-purple-500"
                />
              </div>
            </div>

            {/* Sampler & Scheduler */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-[#8B949E] block mb-1">Sampler</label>
                <select
                  value={sampler}
                  onChange={(e) => setSampler(e.target.value)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-purple-500 cursor-pointer"
                >
                  {AVAILABLE_SAMPLERS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-[10px] text-[#8B949E] block mb-1">Scheduler</label>
                <select
                  value={scheduler}
                  onChange={(e) => setScheduler(e.target.value)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-purple-500 cursor-pointer"
                >
                  {AVAILABLE_SCHEDULERS.map((sc) => (
                    <option key={sc} value={sc}>
                      {sc}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* 6. Primary Action: Generate / Stop */}
          <div className="pt-1 pb-4">
            {!isGenerating ? (
              <button
                type="button"
                onClick={handleGenerate}
                className="w-full py-3 px-4 bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold text-sm rounded-xl shadow-lg shadow-purple-950/40 border border-purple-400/30 flex items-center justify-center space-x-2 transition-all cursor-pointer"
              >
                <Play className="w-4 h-4 fill-current" />
                <span>GENERATE MULTI-CHARACTER ({characterCount} CHARACTERS)</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={handleCancel}
                className="w-full py-3 px-4 bg-red-600 hover:bg-red-500 text-white font-bold text-sm rounded-xl shadow-lg shadow-red-950/40 border border-red-400/30 flex items-center justify-center space-x-2 transition-all cursor-pointer"
              >
                <Square className="w-4 h-4 fill-current" />
                <span>CANCEL GENERATION</span>
              </button>
            )}
          </div>
        </div>

        {/* Right Output Column */}
        <div className="flex-1 bg-[#090C12] flex flex-col h-full min-w-0 overflow-hidden">
          {/* Output Toolbar */}
          <div className="h-11 px-5 border-b border-[#21262D] flex items-center justify-between shrink-0 bg-[#0D1117]/60">
            <div className="flex items-center space-x-2 text-xs font-semibold text-[#8B949E]">
              <ImageIcon className="w-4 h-4 text-purple-400" />
              <span>Canvas Output ({width} × {height})</span>
            </div>

            {result?.imageUrl && (
              <div className="flex items-center space-x-2">
                <button
                  type="button"
                  onClick={handleDownload}
                  className="px-3 py-1 bg-[#161B22] border border-[#30363D] hover:border-purple-500 text-white rounded text-xs flex items-center space-x-1.5 transition"
                >
                  <Download className="w-3.5 h-3.5 text-purple-400" />
                  <span>Save Image</span>
                </button>
              </div>
            )}
          </div>

          {/* Main Canvas Viewport */}
          <div className="flex-1 flex flex-col items-center justify-center p-6 min-h-0 overflow-hidden relative">
            {isGenerating && (
              <div className="absolute inset-0 bg-[#090C12]/80 backdrop-blur-sm z-20 flex flex-col items-center justify-center space-y-4 p-8">
                <div className="w-16 h-16 rounded-2xl bg-purple-950/60 border border-purple-600/50 flex items-center justify-center shadow-2xl">
                  <Users className="w-8 h-8 text-purple-400 animate-pulse" />
                </div>
                <div className="text-center space-y-2 max-w-md">
                  <p className="text-sm font-bold text-white tracking-wide">
                    Synthesizing {characterCount} Characters in Parallel
                  </p>
                  <p className="text-xs text-[#8B949E] font-mono">
                    {progressMessage || `Sampling Step ${progressStep} / ${progressTotal}`}
                  </p>
                  {/* Progress Bar */}
                  <div className="w-64 h-2 bg-[#161B22] rounded-full overflow-hidden border border-[#30363D] mx-auto">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-indigo-500 transition-all duration-300"
                      style={{ width: `${percent}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-purple-300 font-mono font-bold">
                    {percent}% complete
                  </p>
                </div>
              </div>
            )}

            {result?.imageUrl ? (
              <div className="max-w-full max-h-full flex items-center justify-center">
                <img
                  src={result.imageUrl}
                  alt="Multi-Character Generation Result"
                  className="max-w-full max-h-[calc(100vh-180px)] object-contain rounded-xl border border-[#21262D] shadow-2xl"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center text-center space-y-3 p-8 border-2 border-dashed border-[#21262D] rounded-2xl max-w-md">
                <div className="w-12 h-12 rounded-xl bg-[#161B22] border border-[#30363D] flex items-center justify-center text-[#484F58]">
                  <Users className="w-6 h-6 text-purple-400/50" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white">No Multi-Character Render Yet</h3>
                  <p className="text-xs text-[#8B949E] mt-1">
                    Select 2 to 4 characters, set their prompts and optional LoRAs, and click &quot;Generate Multi-Character&quot;.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Bottom Telemetry Bar */}
          <div className="h-10 px-5 border-t border-[#21262D] flex items-center justify-between shrink-0 bg-[#0D1117] text-xs font-mono text-[#8B949E]">
            <div className="flex items-center space-x-4">
              <span>Characters: <strong className="text-purple-300">{characterCount}</strong></span>
              {result && (
                <span>Time: <strong className="text-white">{(result.generationTimeMs / 1000).toFixed(2)}s</strong></span>
              )}
              {result && (
                <span>Seed: <strong className="text-white">{result.seed}</strong></span>
              )}
            </div>

            <div className="flex items-center space-x-2">
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-emerald-400 font-semibold">Ready</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
