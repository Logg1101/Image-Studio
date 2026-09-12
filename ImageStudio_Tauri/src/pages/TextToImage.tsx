import React, { useState } from "react";
import {
  Sparkles,
  Play,
  Square,
  Shuffle,
  Download,
  Copy,
  Check,
  AlertCircle,
  Loader2,
  RefreshCw,
  Image as ImageIcon,
  Sliders,
  Maximize2,
  Layers,
  Dices,
  Trash2,
  Brain,
  Plus,
  Ban,
  Minus,
} from "lucide-react";
import { useGeneration } from "../state/generationContext";
import {
  AVAILABLE_SAMPLERS,
  AVAILABLE_SCHEDULERS,
  generationService,
} from "../services/generationService";
import { AspectRatioPreset } from "../types";
import { LoraSelectorModal } from "../components/generation/LoraSelectorModal";
import { LoraCard } from "../components/generation/LoraCard";
import { AIPromptGenerator } from "../components/generation/AIPromptGenerator";

export const TextToImage: React.FC = () => {
  const {
    models,
    availableLoras,
    selectedModelId,
    setSelectedModelId,
    prompt,
    setPrompt,
    negativePrompt,
    setNegativePrompt,
    activeLoras,
    addLora,
    removeLora,
    updateLoraWeight,
    toggleLora,
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
    generate,
    cancel,
    randomizePrompt,
    clearPrompt,
    refreshModelsAndLoras,
  } = useGeneration();

  const [isLoraModalOpen, setIsLoraModalOpen] = useState(false);
  const [isRefreshingModels, setIsRefreshingModels] = useState(false);
  const [copiedPrompt, setCopiedPrompt] = useState(false);
  const [copiedSeed, setCopiedSeed] = useState(false);
  const [isUpscalingStandalone, setIsUpscalingStandalone] = useState(false);
  const [upscaleError, setUpscaleError] = useState<string | null>(null);
  const [isConvertingPrompt, setIsConvertingPrompt] = useState(false);
  const [promptConverterError, setPromptConverterError] = useState<string | null>(null);
  const [converterProfile, setConverterProfile] = useState<string>("illustrious_xl");

  const isGenerating = status === "loading_model" || status === "generating";
  
  // Total steps strictly defaults to the user's selected steps
  const displayTotal = steps > 0 ? steps : (progressTotal > 0 ? progressTotal : 30);
  const percent =
    progressStep > 0 && displayTotal > 0
      ? Math.min(100, Math.round((progressStep / displayTotal) * 100))
      : (status === "loading_model" ? 0 : 0);

  const ratios: AspectRatioPreset[] = ["1:1", "16:9", "9:16", "4:3", "3:4"];
  const upscaleMethods = ["None", "4x-RealCUGAN", "Tiled SDXL", "Lanczos", "Bicubic"];
  const upscaleFactors = [1.5, 2.0, 3.0, 4.0];

  const activePaths = new Set(activeLoras.map((l) => l.path));

  const handleRefreshModels = async () => {
    setIsRefreshingModels(true);
    await refreshModelsAndLoras();
    setTimeout(() => setIsRefreshingModels(false), 600);
  };

  const copyPromptText = () => {
    navigator.clipboard.writeText(prompt);
    setCopiedPrompt(true);
    setTimeout(() => setCopiedPrompt(false), 2000);
  };

  const handleAppendPrompt = (newTags: string) => {
    if (!prompt.trim()) {
      setPrompt(newTags);
      return;
    }
    const existingTags = prompt.split(",").map((t) => t.trim()).filter(Boolean);
    const incomingTags = newTags.split(",").map((t) => t.trim()).filter(Boolean);
    const existingSet = new Set(existingTags.map((t) => t.toLowerCase()));

    const toAdd = incomingTags.filter((t) => !existingSet.has(t.toLowerCase()));
    if (toAdd.length === 0) return;

    setPrompt(`${prompt.trim().replace(/,\s*$/, "")}, ${toAdd.join(", ")}`);
  };

  const copySeedText = () => {
    if (result?.seed !== undefined) {
      navigator.clipboard.writeText(result.seed.toString());
      setCopiedSeed(true);
      setTimeout(() => setCopiedSeed(false), 2000);
    }
  };

  const handleConvertPrompt = async () => {
    if (!prompt.trim() || isConvertingPrompt) return;
    setIsConvertingPrompt(true);
    setPromptConverterError(null);
    try {
      const res = await generationService.generatePromptTags({
        text: prompt,
        profile: converterProfile,
        style: "General",
        existing_prompt: "",
        mode: "replace",
      });
      if (res.success && res.prompt) {
        setPrompt(res.prompt);
        if (res.negative_prompt && (!negativePrompt.trim() || negativePrompt.includes("blurry, low quality"))) {
          setNegativePrompt(res.negative_prompt);
        }
      } else {
        setPromptConverterError(res.error || "LLM tag conversion failed");
      }
    } catch (e: any) {
      setPromptConverterError(e.message || "Failed to contact LLM backend");
    } finally {
      setIsConvertingPrompt(false);
    }
  };

  // Direct On-Demand Upscaler for rendered preview
  const handleOnDemandUpscale = async (model: string = "4x-realcugan", scale: number = 2) => {
    if (!result?.imageUrl) return;
    setIsUpscalingStandalone(true);
    setUpscaleError(null);
    try {
      const res = await fetch("http://127.0.0.1:8188/api/story/upscale_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_url: result.imageUrl,
          model_id: model.toLowerCase(),
          scale: scale,
        }),
      });
      if (!res.ok) {
        throw new Error(`Upscale API returned status ${res.status}`);
      }
      const data = await res.json();
      if (data.image_url) {
        result.imageUrl = data.image_url;
      }
    } catch (err: unknown) {
      setUpscaleError(err instanceof Error ? err.message : "Upscaling failed");
    } finally {
      setIsUpscalingStandalone(false);
    }
  };

  const activeModel = models.find((m) => m.id === selectedModelId);

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0B0E14] text-[#E2E8F0] overflow-hidden">
      {/* Top Header Bar - Clean title and active checkpoint badge (NO DUPLICATE STATS) */}
      <div className="h-11 px-5 bg-[#0D1117] border-b border-[#21262D] flex items-center justify-between shrink-0 select-none">
        <div className="flex items-center space-x-3">
          <Sparkles className="w-4 h-4 text-[#4F9CFF]" />
          <span className="font-bold text-xs tracking-wide bg-gradient-to-r from-blue-400 to-indigo-300 bg-clip-text text-transparent">
            TEXT TO IMAGE STUDIO
          </span>
          <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-blue-950/60 text-blue-300 border border-blue-800/50">
            Workstation
          </span>
        </div>

        {/* Checkpoint Status Indicator */}
        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="text-[#8B949E] text-[11px]">Checkpoint:</span>
          <span className="text-white font-semibold px-2 py-0.5 rounded bg-[#161B22] border border-[#30363D]">
            {activeModel?.name || selectedModelId || "SDXL"}
          </span>
          {activeModel?.architecture && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-950/60 border border-blue-800/40 text-blue-300 font-bold uppercase">
              {activeModel.architecture}
            </span>
          )}
        </div>
      </div>

      {/* Error notification banner */}
      {(error || upscaleError) && (
        <div className="px-6 py-2 bg-red-950/80 border-b border-red-800/80 text-red-200 text-xs flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
            <span>{error || upscaleError}</span>
          </div>
        </div>
      )}

      {/* Main Workspace Body */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Left Control Column */}
        <div className="w-[540px] border-r border-[#21262D] bg-[#0D1117] flex flex-col h-full overflow-y-auto custom-scrollbar p-4 space-y-3.5 shrink-0">
          {/* 1. Checkpoint Selector Dropdown */}
          <div className="space-y-1.5 p-3 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-bold text-blue-300 uppercase tracking-wider">
                <Brain className="w-4 h-4 text-blue-400" />
                <span>Model Checkpoint</span>
              </div>
              <button
                type="button"
                onClick={handleRefreshModels}
                title="Scan models/checkpoints folder"
                className="p-1 rounded text-blue-400 hover:text-blue-200 hover:bg-blue-900/40 transition flex items-center space-x-1 text-[10px]"
              >
                <RefreshCw className={`w-3 h-3 ${isRefreshingModels ? "animate-spin text-blue-400" : ""}`} />
                <span>Rescan</span>
              </button>
            </div>

            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-blue-500 font-medium cursor-pointer"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} [{m.architecture?.toUpperCase() || "MODEL"}] {m.description ? `(${m.description})` : ""}
                </option>
              ))}
            </select>
          </div>

          {/* 2. AI Prompt Generator (Natural Language to SDXL Tags) */}
          <AIPromptGenerator
            currentPrompt={prompt}
            onReplacePrompt={setPrompt}
            onAppendPrompt={handleAppendPrompt}
            onSetNegativePrompt={setNegativePrompt}
          />

          {/* 3. Positive Prompt Panel */}
          <div className="space-y-1.5 p-3.5 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-purple-300 uppercase tracking-wider flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                <span>Prompt Panel</span>
              </span>
              <div className="flex items-center space-x-1 text-[10px] font-mono text-[#8B949E]">
                <span>{prompt.length} chars</span>
              </div>
            </div>

            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={7}
              placeholder="Describe your desired image: character, subject, composition, pose, attire, lighting details, ambiance..."
              className="w-full min-h-[160px] bg-[#0D1117] border border-[#30363D] rounded-lg p-3 text-xs text-white font-mono leading-relaxed focus:outline-none focus:border-purple-500 resize-y custom-scrollbar"
            />

            <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
              <div className="flex flex-wrap items-center gap-1.5">
                {/* 1-Click LLM Tag Converter */}
                <button
                  type="button"
                  onClick={handleConvertPrompt}
                  disabled={isConvertingPrompt || !prompt.trim()}
                  className="px-2.5 py-1 rounded bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white text-[11px] font-bold flex items-center space-x-1.5 shadow transition disabled:opacity-50 cursor-pointer"
                  title="Convert natural language description into SDXL tags using local LLM"
                >
                  {isConvertingPrompt ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-purple-200" />
                      <span>Converting Tags...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-3.5 h-3.5 text-yellow-300" />
                      <span>Convert to Tags</span>
                    </>
                  )}
                </button>

                <select
                  value={converterProfile}
                  onChange={(e) => setConverterProfile(e.target.value)}
                  className="bg-[#0D1117] border border-[#30363D] text-[10px] text-purple-300 rounded px-1.5 py-1 font-mono cursor-pointer focus:outline-none focus:border-purple-500"
                  title="Target Model Profile for Tag Conversion"
                >
                  <option value="illustrious_xl">Illustrious XL</option>
                  <option value="sdxl_base">SDXL Base</option>
                  <option value="pony">Pony XL</option>
                  <option value="animagine_xl">Animagine XL</option>
                </select>

                <button
                  type="button"
                  onClick={randomizePrompt}
                  className="px-2.5 py-1 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] text-[#8993A7] hover:text-white text-[11px] font-semibold flex items-center space-x-1 transition"
                  title="Load example idea"
                >
                  <Dices className="w-3 h-3" />
                  <span>Idea</span>
                </button>
                <button
                  type="button"
                  onClick={clearPrompt}
                  className="p-1 rounded bg-[#0D1117] hover:bg-red-950/40 border border-[#30363D] hover:border-red-500/50 text-[#8993A7] hover:text-red-400 text-[11px] transition"
                  title="Clear prompt"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>

              <button
                type="button"
                onClick={copyPromptText}
                className="text-[11px] font-semibold text-gray-400 hover:text-white flex items-center space-x-1"
              >
                {copiedPrompt ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                <span>{copiedPrompt ? "Copied" : "Copy"}</span>
              </button>
            </div>

            {promptConverterError && (
              <div className="mt-2 p-2.5 rounded-lg bg-red-950/70 border border-red-800 text-red-200 text-xs flex items-start space-x-2">
                <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="font-semibold">Prompt Converter Error</p>
                  <p className="text-[11px] text-red-300/90">{promptConverterError}</p>
                </div>
              </div>
            )}
          </div>

          {/* 3. Negative Prompt Panel */}
          <div className="space-y-1.5 p-3.5 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-rose-300 uppercase tracking-wider flex items-center space-x-1.5">
                <Ban className="w-3.5 h-3.5 text-rose-400" />
                <span>Negative Prompt Panel</span>
              </span>
              <button
                type="button"
                onClick={() => setNegativePrompt("blurry, low quality, distorted, bad anatomy, deformed, artifacts, ugly, worst quality")}
                className="text-[10px] text-[#8B949E] hover:text-white"
              >
                Default Preset
              </button>
            </div>

            <textarea
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              rows={3}
              placeholder="Elements to suppress: blurry, worst quality, deformed limbs, bad hands..."
              className="w-full min-h-[75px] bg-[#0D1117] border border-[#30363D] rounded-lg p-2.5 text-xs text-[#8B949E] font-mono leading-relaxed focus:outline-none focus:border-rose-500 resize-y custom-scrollbar"
            />
          </div>

          {/* 4. LoRA Selector Rack */}
          <div className="space-y-2 p-3.5 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-cyan-300 uppercase tracking-wider flex items-center space-x-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                <span>LoRA Adapters ({activeLoras.length})</span>
              </span>
              <button
                type="button"
                onClick={() => setIsLoraModalOpen(true)}
                className="px-2.5 py-1 rounded bg-[#0D1117] hover:bg-cyan-950/40 border border-cyan-800/50 hover:border-cyan-400 text-cyan-300 text-[11px] font-bold flex items-center space-x-1 transition"
              >
                <Plus className="w-3 h-3" />
                <span>Browse All</span>
              </button>
            </div>

            {/* Categorized Quick Select Dropdown */}
            <select
              value=""
              onChange={(e) => {
                const selected = availableLoras.find((l) => l.id === e.target.value);
                if (selected) addLora(selected);
              }}
              className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg px-2.5 py-1.5 text-xs text-cyan-300 font-medium cursor-pointer focus:outline-none focus:border-cyan-500"
            >
              <option value="">+ Quick Pick LoRA by Category...</option>
              {Array.from(new Set(availableLoras.map((l) => l.category || "General"))).sort().map((cat) => (
                <optgroup key={cat} label={`📂 ${cat}`}>
                  {availableLoras
                    .filter((l) => (l.category || "General") === cat)
                    .map((lora) => (
                      <option key={lora.id} value={lora.id}>
                        {lora.displayName}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>

            <div className="space-y-1.5 max-h-40 overflow-y-auto custom-scrollbar">
              {activeLoras.length === 0 ? (
                <div className="p-2.5 bg-[#0D1117] rounded-lg border border-dashed border-[#30363D] text-center text-xs text-[#8B949E]">
                  No LoRAs attached. Click &ldquo;+ Add LoRA&rdquo; to attach from models/loras.
                </div>
              ) : (
                activeLoras.map((lora) => (
                  <LoraCard
                    key={lora.id}
                    lora={lora}
                    onWeightChange={updateLoraWeight}
                    onToggle={toggleLora}
                    onRemove={removeLora}
                  />
                ))
              )}
            </div>
          </div>

          {/* 5. Generation Settings Panel (Working Steps & CFG Counters, Samplers, Schedulers, Dimensions, Seed) */}
          <div className="p-3 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm space-y-3">
            <div className="flex items-center space-x-1.5 text-[#C9D1D9] text-xs font-bold uppercase tracking-wider">
              <Sliders className="w-3.5 h-3.5 text-purple-400" />
              <span>Generation Settings</span>
            </div>

            {/* Resolution Aspect Ratios */}
            <div>
              <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                Aspect Ratio Preset ({width} x {height})
              </label>
              <div className="grid grid-cols-5 gap-1">
                {ratios.map((r) => (
                  <button
                    key={r}
                    type="button"
                    onClick={() => setAspectRatio(r)}
                    className={`py-1 rounded text-xs font-bold transition ${
                      aspectRatio === r
                        ? "bg-purple-600 text-white shadow-sm"
                        : "bg-[#0D1117] text-[#8B949E] hover:text-white border border-[#30363D]"
                    }`}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>

            {/* Custom Width & Height */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] text-[#8B949E] mb-0.5">Width (px)</label>
                <input
                  type="number"
                  step={64}
                  value={width}
                  onChange={(e) => setDimensions(parseInt(e.target.value, 10) || 512, height)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1 text-xs text-white font-mono"
                />
              </div>
              <div>
                <label className="block text-[10px] text-[#8B949E] mb-0.5">Height (px)</label>
                <input
                  type="number"
                  step={64}
                  value={height}
                  onChange={(e) => setDimensions(width, parseInt(e.target.value, 10) || 512)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1 text-xs text-white font-mono"
                />
              </div>
            </div>

            {/* Sampler & Scheduler */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                  Sampler
                </label>
                <select
                  value={sampler}
                  onChange={(e) => setSampler(e.target.value)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500 font-medium"
                >
                  {AVAILABLE_SAMPLERS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                  Scheduler
                </label>
                <select
                  value={scheduler}
                  onChange={(e) => setScheduler(e.target.value)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500 font-medium"
                >
                  {AVAILABLE_SCHEDULERS.map((sch) => (
                    <option key={sch} value={sch}>
                      {sch}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Interactive Steps Counter & CFG Scale Counter (Numeric box + Increment/Decrement buttons + Slider) */}
            <div className="grid grid-cols-2 gap-3 pt-1">
              {/* Steps Counter */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px] text-[#8B949E] font-semibold uppercase">
                  <span>Steps</span>
                  <div className="flex items-center space-x-1">
                    <button
                      type="button"
                      onClick={() => setSteps(Math.max(1, steps - 1))}
                      className="w-5 h-5 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] flex items-center justify-center text-white"
                    >
                      <Minus className="w-3 h-3" />
                    </button>
                    <input
                      type="number"
                      min={1}
                      max={150}
                      value={steps}
                      onChange={(e) => setSteps(Math.max(1, Math.min(150, parseInt(e.target.value, 10) || 1)))}
                      className="w-10 bg-[#0D1117] border border-[#30363D] rounded text-center text-xs text-purple-400 font-mono font-bold py-0.5"
                    />
                    <button
                      type="button"
                      onClick={() => setSteps(Math.min(150, steps + 1))}
                      className="w-5 h-5 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] flex items-center justify-center text-white"
                    >
                      <Plus className="w-3 h-3" />
                    </button>
                  </div>
                </div>
                <input
                  type="range"
                  min="10"
                  max="60"
                  step="1"
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value))}
                  className="w-full accent-purple-500 cursor-pointer"
                />
              </div>

              {/* CFG Scale Counter */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px] text-[#8B949E] font-semibold uppercase">
                  <span>CFG Scale</span>
                  <div className="flex items-center space-x-1">
                    <button
                      type="button"
                      onClick={() => setCfgScale(Math.max(1, Number((cfgScale - 0.5).toFixed(1))))}
                      className="w-5 h-5 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] flex items-center justify-center text-white"
                    >
                      <Minus className="w-3 h-3" />
                    </button>
                    <input
                      type="number"
                      step={0.5}
                      min={1.0}
                      max={30.0}
                      value={cfgScale}
                      onChange={(e) => setCfgScale(Math.max(1, Math.min(30, parseFloat(e.target.value) || 1.0)))}
                      className="w-12 bg-[#0D1117] border border-[#30363D] rounded text-center text-xs text-purple-400 font-mono font-bold py-0.5"
                    />
                    <button
                      type="button"
                      onClick={() => setCfgScale(Math.min(30, Number((cfgScale + 0.5).toFixed(1))))}
                      className="w-5 h-5 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] flex items-center justify-center text-white"
                    >
                      <Plus className="w-3 h-3" />
                    </button>
                  </div>
                </div>
                <input
                  type="range"
                  min="1.0"
                  max="15.0"
                  step="0.5"
                  value={cfgScale}
                  onChange={(e) => setCfgScale(Number(e.target.value))}
                  className="w-full accent-purple-500 cursor-pointer"
                />
              </div>
            </div>

            {/* Seed Configuration */}
            <div>
              <div className="flex items-center justify-between text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                <span>Seed</span>
                <label className="flex items-center space-x-1 cursor-pointer text-xs lowercase">
                  <input
                    type="checkbox"
                    checked={isRandomSeed}
                    onChange={(e) => setIsRandomSeed(e.target.checked)}
                    className="accent-purple-500 rounded"
                  />
                  <span>randomize (-1)</span>
                </label>
              </div>
              <div className="flex space-x-1.5">
                <input
                  type="number"
                  disabled={isRandomSeed}
                  value={isRandomSeed ? -1 : seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="flex-1 bg-[#0D1117] border border-[#30363D] rounded px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-purple-500 disabled:opacity-50"
                />
                <button
                  type="button"
                  disabled={isRandomSeed}
                  onClick={() => setSeed(Math.floor(Math.random() * 2147483647))}
                  className="px-2.5 py-1.5 bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] text-[#8993A7] hover:text-white rounded text-xs disabled:opacity-40"
                  title="Roll new random seed"
                >
                  <Shuffle className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* 6. Neural Upscalers Section */}
            <div className="pt-2 border-t border-[#30363D] space-y-2">
              <div className="flex items-center space-x-1.5 text-xs font-bold text-cyan-300 uppercase tracking-wider">
                <Maximize2 className="w-3.5 h-3.5 text-cyan-400" />
                <span>High-Res Neural Upscaler</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[10px] text-[#8B949E] mb-0.5">Model</label>
                  <select
                    value={upscaleMethod}
                    onChange={(e) => setUpscaleMethod(e.target.value)}
                    className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                  >
                    {upscaleMethods.map((m) => (
                      <option key={m} value={m}>
                        {m === "None" ? "None (Disabled)" : m}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] text-[#8B949E] mb-0.5">Factor</label>
                  <select
                    value={upscaleFactor}
                    disabled={upscaleMethod === "None"}
                    onChange={(e) => setUpscaleFactor(parseFloat(e.target.value))}
                    className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 disabled:opacity-40"
                  >
                    {upscaleFactors.map((f) => (
                      <option key={f} value={f}>
                        {f}x
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* 7. Bottom Generation Action Button with Accurate Progress Bar */}
          <div className="sticky bottom-0 bg-[#0D1117] pt-2 pb-1 space-y-2">
            {/* Live Progress Bar during active generation */}
            {isGenerating && (
              <div className="space-y-1.5 p-2.5 bg-[#161B22] border border-[#30363D] rounded-lg shadow-md">
                <div className="flex items-center justify-between text-[11px] font-mono text-[#8B949E]">
                  <span className="truncate max-w-[280px] text-white">
                    {progressStep > 0
                      ? `Sampling step ${progressStep} / ${displayTotal}`
                      : progressMessage || "Preparing synthesis..."}
                  </span>
                  <span className="font-bold text-cyan-300">
                    {progressStep > 0 ? `${percent}%` : "In VRAM"}
                  </span>
                </div>
                <div className="w-full h-2 bg-[#0D1117] rounded-full overflow-hidden border border-[#30363D]">
                  {progressStep > 0 ? (
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 transition-all duration-200"
                      style={{ width: `${percent}%` }}
                    />
                  ) : (
                    <div className="h-full bg-blue-500 w-1/3 animate-pulse rounded-full" />
                  )}
                </div>
              </div>
            )}

            {isGenerating ? (
              <button
                type="button"
                onClick={() => cancel()}
                className="w-full py-3 bg-red-600 hover:bg-red-700 text-white text-xs font-bold uppercase rounded-lg shadow-lg flex items-center justify-center space-x-2 transition"
              >
                <Square className="w-4 h-4 fill-white" />
                <span>Cancel Generation ({progressStep > 0 ? `${percent}%` : "Loading..."})</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={() => generate()}
                className="w-full py-3 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white text-xs font-bold uppercase tracking-wider rounded-lg shadow-lg flex items-center justify-center space-x-2 transition"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>Generate Image</span>
              </button>
            )}
          </div>
        </div>

        {/* Right Dominant Preview Pane */}
        <div className="flex-1 bg-[#090C12] flex flex-col min-w-0 h-full overflow-hidden p-6">
          {isGenerating ? (
            /* Live Centerpiece Generation Overlay */
            <div className="flex-1 flex flex-col items-center justify-center">
              <div className="w-full max-w-md bg-[#161B22] border border-[#30363D] rounded-2xl shadow-2xl p-6 space-y-4 text-center">
                <div className="w-12 h-12 rounded-2xl bg-[#0D1117] border border-[#30363D] flex items-center justify-center text-blue-400 mx-auto shadow-inner">
                  <Loader2 className="w-6 h-6 animate-spin text-blue-400" />
                </div>

                <div className="space-y-1">
                  <h3 className="font-bold text-sm text-white tracking-wide flex items-center justify-center space-x-1.5">
                    <Sparkles className="w-4 h-4 text-emerald-400" />
                    <span>
                      {progressStep > 0
                        ? "Denoising Latents on GPU"
                        : "Preparing Pipeline & VRAM"}
                    </span>
                  </h3>
                  <p className="text-xs text-[#8B949E] font-mono leading-relaxed truncate px-2">
                    {progressMessage || `Generating with '${activeModel?.name || selectedModelId}' on GPU...`}
                  </p>
                </div>

                {/* Progress Bar */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex items-center justify-between text-xs font-mono text-[#8993A7]">
                    <span>Step {progressStep} / {displayTotal}</span>
                    <span className="font-bold text-white">
                      {progressStep > 0 ? `${percent}%` : "Loading..."}
                    </span>
                  </div>
                  <div className="w-full h-2.5 bg-[#0D1117] rounded-full overflow-hidden border border-[#30363D] p-0.5">
                    {progressStep > 0 ? (
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 transition-all duration-200"
                        style={{ width: `${percent}%` }}
                      />
                    ) : (
                      <div className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 w-1/2 animate-pulse rounded-full" />
                    )}
                  </div>
                </div>

                {/* Target Specs */}
                <div className="flex items-center justify-center space-x-2 pt-2 text-[10px] font-mono text-[#8B949E]">
                  <span className="px-2 py-0.5 rounded bg-[#0D1117] border border-[#30363D]">
                    {width} x {height}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-[#0D1117] border border-[#30363D]">
                    Steps: {displayTotal} | CFG: {cfgScale}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-[#0D1117] border border-[#30363D] truncate max-w-[150px]">
                    {activeModel?.name || selectedModelId || "SDXL"}
                  </span>
                </div>
              </div>
            </div>
          ) : result ? (
            /* Rendered Image Viewport */
            <div className="flex-1 flex flex-col min-h-0 items-center justify-center relative">
              {/* Output Image */}
              <div className="relative max-h-full max-w-full flex items-center justify-center rounded-xl overflow-hidden shadow-2xl border border-[#21262D] bg-[#000000]">
                <img
                  src={result.imageUrl}
                  alt={`Generated render seed ${result.seed}`}
                  className="max-h-[calc(100vh-180px)] max-w-full object-contain rounded-lg shadow-lg"
                />
              </div>

              {/* Action Toolbar on Bottom */}
              <div className="mt-3 flex flex-wrap items-center justify-between w-full max-w-2xl px-4 py-2 bg-[#161B22] border border-[#30363D] rounded-xl shadow-lg">
                <div className="flex items-center space-x-2 text-xs font-mono text-[#8B949E]">
                  <span className="bg-[#0D1117] px-2 py-1 rounded border border-[#30363D]">
                    Seed: {result.seed}
                  </span>
                  {result.generationTimeMs && (
                    <span className="bg-[#0D1117] px-2 py-1 rounded border border-[#30363D]">
                      {(result.generationTimeMs / 1000).toFixed(2)}s
                    </span>
                  )}
                  <span className="bg-[#0D1117] px-2 py-1 rounded border border-[#30363D]">
                    {width}x{height}
                  </span>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={copySeedText}
                    className="px-2.5 py-1.5 bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] text-[#8993A7] hover:text-white rounded-lg text-xs font-semibold flex items-center space-x-1 transition"
                  >
                    {copiedSeed ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>Seed</span>
                  </button>

                  <button
                    type="button"
                    disabled={isUpscalingStandalone}
                    onClick={() => handleOnDemandUpscale("4x-realcugan", 2)}
                    className="px-3 py-1.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-bold flex items-center space-x-1.5 shadow-md disabled:opacity-50 transition"
                  >
                    {isUpscalingStandalone ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Maximize2 className="w-3.5 h-3.5" />
                    )}
                    <span>{isUpscalingStandalone ? "Upscaling..." : "Upscale 2x"}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      if (result.imageUrl) {
                        const link = document.createElement("a");
                        link.href = result.imageUrl;
                        link.download = `imagestudio_${result.seed}.png`;
                        link.click();
                      }
                    }}
                    className="px-3.5 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-bold flex items-center space-x-1.5 shadow-md transition"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download</span>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            /* Idle Placeholder Viewport */
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-3">
              <div className="w-16 h-16 rounded-2xl bg-[#161B22] border border-[#30363D] flex items-center justify-center text-[#8B949E] shadow-inner">
                <ImageIcon className="w-8 h-8 opacity-40" />
              </div>
              <div className="space-y-1 max-w-sm">
                <h3 className="font-bold text-sm text-white">No Image Generated Yet</h3>
                <p className="text-xs text-[#8B949E] leading-relaxed">
                  Select your checkpoint, adjust your steps and CFG, enter your prompt, then click &ldquo;Generate Image&rdquo;.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* LoRA Selector Modal Popover */}
      <LoraSelectorModal
        isOpen={isLoraModalOpen}
        onClose={() => setIsLoraModalOpen(false)}
        availableLoras={availableLoras}
        activeLoraPaths={activePaths}
        onSelectLora={addLora}
      />
    </div>
  );
};
