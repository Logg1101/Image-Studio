import React, { useState } from "react";
import {
  Sparkles,
  Play,
  Square,
  Shuffle,
  RotateCcw,
  Download,
  Copy,
  Check,
  AlertCircle,
  Loader2,
  RefreshCw,
  Layers,
  Image as ImageIcon,
  Sliders,
  Lock,
  Smile,
  Maximize,
} from "lucide-react";
import { useStoryStudio } from "../state/storyStudioContext";

export const DegenerateStudio: React.FC = () => {
  const {
    catalogues,
    characters,
    isLoadingCatalogues,
    refreshCharacters,
    selectedCharacter,
    setSelectedCharacter,
    selectedExpression,
    setSelectedExpression,
    selectedClothesType,
    setSelectedClothesType,
    selectedClothesColor,
    setSelectedClothesColor,
    selectedClothesDetails,
    toggleClothesDetail,
    selectedPose,
    setSelectedPose,
    selectedPov,
    setSelectedPov,
    selectedBondageType,
    setSelectedBondageType,
    selectedBondage,
    setSelectedBondage,
    selectedBondageHands,
    setSelectedBondageHands,
    selectedBondageLegs,
    setSelectedBondageLegs,
    selectedBondageAccessories,
    setSelectedBondageAccessories,
    selectedGag,
    setSelectedGag,
    selectedScene,
    setSelectedScene,
    resolution,
    setResolution,
    sampler,
    setSampler,
    scheduler,
    setScheduler,
    steps,
    setSteps,
    cfg,
    setCfg,
    seed,
    setSeed,
    batchCount,
    setBatchCount,
    upscaleMethod,
    setUpscaleMethod,
    upscaleFactor,
    setUpscaleFactor,
    positivePrompt,
    setPositivePrompt,
    negativePrompt,
    setNegativePrompt,
    isGeneratingPrompt,
    refreshPrompt,
    isGenerating,
    isUpscaling,
    progressStep,
    progressTotal,
    progressPercent,
    progressMessage,
    batchProgress,
    outputImages,
    selectedImageIndex,
    setSelectedImageIndex,
    errorMessage,
    generateSingleImage,
    generateBatchImages,
    upscaleCurrentImage,
    randomizeSelections,
    clearSelections,
    cancelGeneration,
  } = useStoryStudio();

  const [copiedPrompt, setCopiedPrompt] = useState(false);

  const copyPromptText = () => {
    navigator.clipboard.writeText(positivePrompt);
    setCopiedPrompt(true);
    setTimeout(() => setCopiedPrompt(false), 2000);
  };

  const activeChar = characters.find((c) => c.id === selectedCharacter);
  const currentImage = outputImages[selectedImageIndex] || null;

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0B0E14] text-[#E2E8F0] overflow-hidden">
      {/* Top Header Bar */}
      <div className="h-12 px-6 bg-[#0D1117] border-b border-[#21262D] flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-3">
          <Sparkles className="w-5 h-5 text-[#8A2BE2]" />
          <span className="font-bold text-sm tracking-wide bg-gradient-to-r from-purple-400 to-indigo-300 bg-clip-text text-transparent">
            DEGENERATE STUDIO
          </span>
          <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-purple-950/60 text-purple-300 border border-purple-800/50">
            Hassaku XL Illustrious
          </span>
        </div>

        {/* Global Quick Action Controls */}
        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={() => randomizeSelections()}
            title="Randomize character, outfit, pose & scene"
            className="flex items-center space-x-1.5 px-3 py-1 bg-[#161B22] hover:bg-[#21262D] border border-[#30363D] hover:border-purple-500/50 text-[#C9D1D9] text-xs font-semibold rounded-md transition shadow-sm"
          >
            <Shuffle className="w-3.5 h-3.5 text-purple-400" />
            <span>Randomize</span>
          </button>
          <button
            type="button"
            onClick={() => clearSelections()}
            title="Reset to default selections"
            className="flex items-center space-x-1.5 px-3 py-1 bg-[#161B22] hover:bg-[#21262D] border border-[#30363D] hover:border-gray-500 text-[#8B949E] hover:text-white text-xs font-semibold rounded-md transition shadow-sm"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Error notification banner */}
      {errorMessage && (
        <div className="px-6 py-2 bg-red-950/80 border-b border-red-800/80 text-red-200 text-xs flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        </div>
      )}

      {/* Main Workspace Body */}
      {isLoadingCatalogues ? (
        <div className="flex-1 flex flex-col items-center justify-center space-y-3">
          <Loader2 className="w-8 h-8 text-purple-500 animate-spin" />
          <span className="text-xs font-semibold text-gray-400">Loading Catalogues & Character LoRAs...</span>
        </div>
      ) : (
        <div className="flex-1 flex min-h-0 overflow-hidden">
          {/* Left Control Column */}
          <div className="w-[480px] border-r border-[#21262D] bg-[#0D1117] flex flex-col h-full overflow-y-auto custom-scrollbar p-5 space-y-5 shrink-0">
            {/* Character Selector (Live LoRA Scanner) */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center space-x-1.5">
                  <span className="text-xs font-semibold text-purple-300 uppercase tracking-wider">
                    Character (Live LoRA Discovery)
                  </span>
                  <button
                    type="button"
                    onClick={() => refreshCharacters()}
                    title="Re-scan models/loras folder"
                    className="p-1 rounded text-purple-400 hover:text-purple-200 hover:bg-purple-900/40 transition"
                  >
                    <RefreshCw className="w-3 h-3" />
                  </button>
                </div>
                {activeChar && (
                  <span className="text-[10px] text-gray-400 font-mono">
                    Weight: {activeChar.default_weight}
                  </span>
                )}
              </div>
              <select
                value={selectedCharacter}
                onChange={(e) => setSelectedCharacter(e.target.value)}
                className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
              >
                {characters.map((char) => (
                  <option key={char.id} value={char.id}>
                    {char.name} {char.description ? `— ${char.description.slice(0, 35)}...` : ""}
                  </option>
                ))}
              </select>
              {activeChar?.description && (
                <p className="text-[10px] text-[#8B949E] mt-1 italic">
                  {activeChar.description}
                </p>
              )}
            </div>

            {/* Facial Expression (16 Expressions) */}
            <div>
              <label className="block text-xs font-semibold text-[#8B949E] uppercase tracking-wider mb-1.5 flex items-center space-x-1.5">
                <Smile className="w-3.5 h-3.5 text-pink-400" />
                <span>Facial Expression (16 Presets)</span>
              </label>
              <select
                value={selectedExpression}
                onChange={(e) => setSelectedExpression(e.target.value)}
                className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
              >
                {catalogues?.expressions?.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Clothes Type (30 Outfits) & Color */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-[#8B949E] uppercase tracking-wider mb-1.5">
                  Outfit (30 Styles)
                </label>
                <select
                  value={selectedClothesType}
                  onChange={(e) => setSelectedClothesType(e.target.value)}
                  className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
                >
                  {catalogues?.clothes_types.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-[#8B949E] uppercase tracking-wider mb-1.5">
                  Clothes Color
                </label>
                <select
                  value={selectedClothesColor}
                  onChange={(e) => setSelectedClothesColor(e.target.value)}
                  className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
                >
                  {catalogues?.clothes_colors.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Multi-Select Clothes Details & Accessories */}
            <div>
              <label className="block text-xs font-semibold text-[#8B949E] uppercase tracking-wider mb-1.5 flex items-center justify-between">
                <span>Accessories & Details (Multi-Select)</span>
                <span className="text-[10px] text-purple-400 font-mono">
                  {selectedClothesDetails.length} selected
                </span>
              </label>
              <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto p-1.5 bg-[#090C12] rounded-lg border border-[#21262D]">
                {catalogues?.clothes_details.map((detail) => {
                  const isSelected = selectedClothesDetails.includes(detail.id);
                  return (
                    <button
                      key={detail.id}
                      type="button"
                      onClick={() => toggleClothesDetail(detail.id)}
                      className={`px-2 py-1 rounded-md text-[10px] font-medium transition-all ${
                        isSelected
                          ? "bg-purple-600 text-white shadow-sm border border-purple-400"
                          : "bg-[#161B22] text-[#8B949E] hover:text-white border border-[#30363D]"
                      }`}
                    >
                      {detail.name}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Pose & POV (10 Poses, 10 POVs) */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-[#8B949E] uppercase tracking-wider mb-1.5">
                  Pose (10 Options)
                </label>
                <select
                  value={selectedPose}
                  onChange={(e) => setSelectedPose(e.target.value)}
                  className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
                >
                  {catalogues?.poses.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-[#8B949E] uppercase tracking-wider mb-1.5">
                  POV / Camera Angle (10 Presets)
                </label>
                <select
                  value={selectedPov}
                  onChange={(e) => setSelectedPov(e.target.value)}
                  className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
                >
                  {catalogues?.povs.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Dedicated Bondage & Restraint Studio (6 Modular Dropdowns) */}
            <div className="p-3.5 bg-gradient-to-br from-purple-950/30 via-[#161B22] to-purple-950/20 rounded-xl border border-purple-800/40 shadow-lg space-y-3">
              <div className="flex items-center justify-between pb-1.5 border-b border-purple-900/30">
                <div className="flex items-center space-x-2 text-purple-300 text-xs font-bold uppercase tracking-wider">
                  <div className="p-1 rounded bg-purple-900/40 text-purple-300">
                    <Lock className="w-3.5 h-3.5" />
                  </div>
                  <span>Bondage & Restraint Studio</span>
                </div>
                <span className="text-[10px] text-purple-400/80 bg-purple-900/30 px-2 py-0.5 rounded-full font-mono">
                  Modular Restraint Rig
                </span>
              </div>

              {/* Row 1: Material & Full Scenario */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-semibold text-purple-300 uppercase tracking-wider mb-1">
                    1. Restraint Material / Type
                  </label>
                  <select
                    value={selectedBondageType}
                    onChange={(e) => setSelectedBondageType(e.target.value)}
                    className="w-full bg-[#0D1117] border border-purple-900/60 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
                  >
                    {catalogues?.bondage_types?.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[11px] font-semibold text-purple-300 uppercase tracking-wider mb-1">
                    2. Bondage Scenario & Rig
                  </label>
                  <select
                    value={selectedBondage}
                    onChange={(e) => setSelectedBondage(e.target.value)}
                    className="w-full bg-[#0D1117] border border-purple-900/60 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
                  >
                    {catalogues?.bondage_styles?.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Row 2: Hands & Legs */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-semibold text-purple-300 uppercase tracking-wider mb-1">
                    3. Hand & Arm Position
                  </label>
                  <select
                    value={selectedBondageHands}
                    onChange={(e) => setSelectedBondageHands(e.target.value)}
                    className="w-full bg-[#0D1117] border border-purple-900/60 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
                  >
                    {catalogues?.bondage_hands?.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[11px] font-semibold text-purple-300 uppercase tracking-wider mb-1">
                    4. Leg & Feet Position
                  </label>
                  <select
                    value={selectedBondageLegs}
                    onChange={(e) => setSelectedBondageLegs(e.target.value)}
                    className="w-full bg-[#0D1117] border border-purple-900/60 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
                  >
                    {catalogues?.bondage_legs?.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Row 3: Gag & Accessories */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-semibold text-purple-300 uppercase tracking-wider mb-1">
                    5. Gag & Mouth Restraint
                  </label>
                  <select
                    value={selectedGag}
                    onChange={(e) => setSelectedGag(e.target.value)}
                    className="w-full bg-[#0D1117] border border-purple-900/60 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
                  >
                    {catalogues?.gag_types?.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[11px] font-semibold text-purple-300 uppercase tracking-wider mb-1">
                    6. Hardware & Accessories
                  </label>
                  <select
                    value={selectedBondageAccessories}
                    onChange={(e) => setSelectedBondageAccessories(e.target.value)}
                    className="w-full bg-[#0D1117] border border-purple-900/60 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
                  >
                    {catalogues?.bondage_accessories?.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Scene Situation */}
            <div>
              <label className="block text-xs font-semibold text-[#8B949E] uppercase tracking-wider mb-1.5">
                Scene Situation
              </label>
              <select
                value={selectedScene}
                onChange={(e) => setSelectedScene(e.target.value)}
                className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
              >
                {catalogues?.scenes.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Advanced Generation Settings */}
            <details className="group p-3 bg-[#161B22] rounded-lg border border-[#30363D]">
              <summary className="flex items-center justify-between cursor-pointer list-none">
                <div className="flex items-center space-x-1.5 text-[#C9D1D9] text-xs font-semibold uppercase tracking-wider">
                  <Sliders className="w-3.5 h-3.5 text-purple-400" />
                  <span>Advanced Settings</span>
                </div>
                <div className="text-[#8B949E] group-open:rotate-180 transition-transform">
                  ▼
                </div>
              </summary>
              <div className="mt-3 space-y-3">
                {/* Resolution */}
                <div>
                  <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                    Resolution Preset
                  </label>
                  <select
                    value={resolution}
                    onChange={(e) => setResolution(e.target.value)}
                    className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
                  >
                    {catalogues?.resolutions.map((res) => (
                      <option key={res.id} value={res.id}>
                        {res.name} ({res.aspect_ratio})
                      </option>
                    ))}
                  </select>
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
                      className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
                    >
                      {catalogues?.samplers.map((s) => (
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
                      className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
                    >
                      {catalogues?.schedulers.map((sch) => (
                        <option key={sch} value={sch}>
                          {sch}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Steps & CFG */}
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <div className="flex justify-between text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                      <span>Steps</span>
                      <span className="font-mono text-purple-400">{steps}</span>
                    </div>
                    <input
                      type="range"
                      min="15"
                      max="60"
                      step="1"
                      value={steps}
                      onChange={(e) => setSteps(Number(e.target.value))}
                      className="w-full accent-purple-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                      <span>CFG Scale</span>
                      <span className="font-mono text-purple-400">{cfg.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="15"
                      step="0.5"
                      value={cfg}
                      onChange={(e) => setCfg(Number(e.target.value))}
                      className="w-full accent-purple-500"
                    />
                  </div>
                </div>

                {/* Seed & Batch Count */}
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                      Seed (-1 = Random)
                    </label>
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                      className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                      Batch Count
                    </label>
                    <div className="flex space-x-1">
                      {[1, 2, 3, 4].map((count) => (
                        <button
                          key={count}
                          type="button"
                          onClick={() => setBatchCount(count)}
                          className={`flex-1 py-1.5 rounded text-xs font-bold transition ${
                            batchCount === count
                              ? "bg-purple-600 text-white shadow-sm"
                              : "bg-[#0D1117] text-[#8B949E] hover:text-white border border-[#30363D]"
                          }`}
                        >
                          {count}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* High-Res Upscaler Section */}
                <div className="pt-2 border-t border-[#30363D] space-y-2">
                  <div className="flex items-center space-x-1.5 text-xs font-semibold text-cyan-300 uppercase tracking-wider">
                    <Maximize className="w-3.5 h-3.5 text-cyan-400" />
                    <span>High-Res Upscaler</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                        Upscale Model
                      </label>
                      <select
                        value={upscaleMethod}
                        onChange={(e) => setUpscaleMethod(e.target.value)}
                        className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
                      >
                        <option value="None">None (Disabled)</option>
                        <option value="4x-RealCUGAN">4x-RealCUGAN</option>
                        <option value="Tiled SDXL">Tiled SDXL</option>
                        <option value="Lanczos">Lanczos</option>
                        <option value="Bicubic">Bicubic</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                        Scale Factor
                      </label>
                      <select
                        value={upscaleFactor}
                        disabled={upscaleMethod === "None"}
                        onChange={(e) => setUpscaleFactor(parseFloat(e.target.value))}
                        className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500 disabled:opacity-40"
                      >
                        <option value="1.5">1.5x</option>
                        <option value="2.0">2.0x (HD)</option>
                        <option value="3.0">3.0x (3K)</option>
                        <option value="4.0">4.0x (4K Ultra)</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            </details>

            {/* Prompt Editor Panel (Live AI Prompt Agent & Manual Editor) */}
            <div className="space-y-3 pt-1 border-t border-[#21262D]">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-purple-300 uppercase tracking-wider flex items-center space-x-1">
                  <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                  <span>Prompt Editor Panel</span>
                </span>
                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={() => refreshPrompt()}
                    disabled={isGeneratingPrompt}
                    title="Auto-synthesize prompt from current selections"
                    className="text-[11px] font-semibold text-purple-400 hover:text-purple-300 flex items-center space-x-1 disabled:opacity-50"
                  >
                    <RefreshCw
                      className={`w-3 h-3 ${isGeneratingPrompt ? "animate-spin text-purple-400" : ""}`}
                    />
                    <span>Re-Synthesize</span>
                  </button>
                  <button
                    type="button"
                    onClick={copyPromptText}
                    title="Copy Positive Prompt"
                    className="text-[11px] font-semibold text-gray-400 hover:text-white flex items-center space-x-1"
                  >
                    {copiedPrompt ? (
                      <Check className="w-3 h-3 text-emerald-400" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                    <span>{copiedPrompt ? "Copied" : "Copy"}</span>
                  </button>
                </div>
              </div>

              {/* Positive Prompt Textarea */}
              <div>
                <label className="block text-[11px] font-semibold text-[#8B949E] uppercase mb-1">
                  Positive Prompt (Hassaku XL Illustrious Tags)
                </label>
                <textarea
                  value={positivePrompt}
                  onChange={(e) => setPositivePrompt(e.target.value)}
                  rows={7}
                  className="w-full min-h-[160px] bg-[#161B22] border border-[#30363D] rounded-lg p-3 text-xs text-white font-mono leading-relaxed focus:outline-none focus:border-purple-500 custom-scrollbar resize-y"
                  placeholder="Masterpiece, 1girl, positive prompt tags..."
                />
              </div>

              {/* Negative Prompt Textarea */}
              <div>
                <label className="block text-[11px] font-semibold text-[#8B949E] uppercase mb-1">
                  Negative Prompt
                </label>
                <textarea
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  rows={3}
                  className="w-full min-h-[75px] bg-[#161B22] border border-[#30363D] rounded-lg p-2.5 text-xs text-[#8B949E] font-mono leading-relaxed focus:outline-none focus:border-purple-500 custom-scrollbar resize-y"
                  placeholder="low quality, worst quality, deformed..."
                />
              </div>
            </div>

            {/* Bottom Generation Buttons */}
            <div className="pt-2 sticky bottom-0 bg-[#0D1117] pb-1 space-y-2">
              {isGenerating ? (
                <button
                  type="button"
                  onClick={() => cancelGeneration()}
                  className="w-full py-3 bg-red-600 hover:bg-red-700 text-white text-xs font-bold uppercase rounded-lg shadow-lg flex items-center justify-center space-x-2 transition"
                >
                  <Square className="w-4 h-4 fill-white" />
                  <span>
                    Cancel Generation{" "}
                    {batchProgress ? `(${batchProgress.current}/${batchProgress.total})` : `(${progressPercent}%)`}
                  </span>
                </button>
              ) : batchCount > 1 ? (
                <button
                  type="button"
                  onClick={() => generateBatchImages()}
                  className="w-full py-3 bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 text-white text-xs font-bold uppercase tracking-wider rounded-lg shadow-lg flex items-center justify-center space-x-2 transition"
                >
                  <Layers className="w-4 h-4" />
                  <span>Generate Batch ({batchCount} Images)</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => generateSingleImage()}
                  className="w-full py-3 bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 text-white text-xs font-bold uppercase tracking-wider rounded-lg shadow-lg flex items-center justify-center space-x-2 transition"
                >
                  <Play className="w-4 h-4 fill-white" />
                  <span>Generate Single Image</span>
                </button>
              )}
            </div>
          </div>

          {/* Right Dominant Output Canvas Viewport */}
          <div className="flex-1 bg-[#090C12] flex flex-col min-w-0 h-full overflow-hidden p-6">
            {currentImage ? (
              <div className="flex-1 flex flex-col min-h-0 items-center justify-center relative">
                {/* Main Render Image */}
                <div className="relative max-h-full max-w-full flex items-center justify-center rounded-xl overflow-hidden shadow-2xl border border-[#21262D] bg-[#000000]">
                  <img
                    src={currentImage.imageUrl}
                    alt={`Generated render seed ${currentImage.seed}`}
                    className={`max-h-[calc(100vh-210px)] max-w-full object-contain rounded-lg transition-all ${
                      isGenerating ? "filter blur-sm opacity-40 scale-[0.98]" : ""
                    }`}
                  />

                  {/* Real-time Generation Progress Overlay when generating with an existing image */}
                  {isGenerating && (
                    <div className="absolute inset-0 bg-black/80 backdrop-blur-md flex flex-col items-center justify-center p-6 z-20 animate-in fade-in duration-200">
                      <div className="flex flex-col items-center space-y-4 max-w-md w-full px-6 bg-[#0D1117]/95 border border-purple-500/50 rounded-2xl p-6 shadow-2xl">
                        <div className="relative">
                          <div className="w-14 h-14 rounded-2xl bg-purple-950/80 border border-purple-700 flex items-center justify-center text-purple-300 shadow-xl">
                            <Loader2 className="w-7 h-7 text-purple-400 animate-spin" />
                          </div>
                          <Sparkles className="w-4 h-4 text-purple-200 absolute -top-1 -right-1 animate-pulse" />
                        </div>

                        <div className="w-full text-center space-y-2">
                          <div className="flex items-center justify-between text-xs font-mono font-semibold">
                            <span className="text-purple-300">
                              {batchProgress
                                ? `Batch Image ${batchProgress.current} of ${batchProgress.total}`
                                : progressStep > 0
                                ? `Step ${progressStep} / ${progressTotal}`
                                : "Sampling on GPU..."}
                            </span>
                            <span className="text-emerald-400 font-bold px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-700/60">
                              {progressPercent}%
                            </span>
                          </div>

                          {/* Animated Gradient Progress Bar */}
                          <div className="w-full h-3 bg-[#161B22] rounded-full overflow-hidden border border-[#30363D] shadow-inner">
                            <div
                              className="h-full bg-gradient-to-r from-purple-600 via-indigo-500 to-cyan-400 rounded-full transition-all duration-300 ease-out shadow-lg"
                              style={{ width: `${Math.max(progressPercent, 4)}%` }}
                            />
                          </div>

                          <p className="text-xs text-gray-300 font-medium">
                            {progressMessage || "Sampling latent steps..."}
                          </p>
                        </div>

                        <button
                          type="button"
                          onClick={() => cancelGeneration()}
                          className="px-4 py-1.5 bg-red-950/80 hover:bg-red-900 border border-red-700 text-red-200 text-xs font-semibold rounded-lg transition flex items-center space-x-1.5 shadow-lg"
                        >
                          <Square className="w-3.5 h-3.5 fill-red-400" />
                          <span>Cancel Generation</span>
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Top Right Quick Actions & Upscale Shortcuts */}
                  <div className="absolute top-3 right-3 flex items-center space-x-2 bg-black/75 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 opacity-90 hover:opacity-100 transition shadow-lg">
                    {/* Quick 2x Upscale */}
                    <button
                      type="button"
                      disabled={isUpscaling}
                      onClick={() => upscaleCurrentImage(2, "4x-realcugan")}
                      title="Neural 2x Upscale with 4x-RealCUGAN"
                      className="flex items-center space-x-1 px-2.5 py-1 bg-cyan-950/80 hover:bg-cyan-900 border border-cyan-700/60 rounded text-[11px] font-semibold text-cyan-200 transition disabled:opacity-50"
                    >
                      {isUpscaling ? (
                        <Loader2 className="w-3 h-3 animate-spin text-cyan-400" />
                      ) : (
                        <Sparkles className="w-3 h-3 text-cyan-400" />
                      )}
                      <span>2x Upscale</span>
                    </button>

                    {/* Quick 4x Upscale */}
                    <button
                      type="button"
                      disabled={isUpscaling}
                      onClick={() => upscaleCurrentImage(4, "4x-realcugan")}
                      title="Neural 4x 4K Upscale with 4x-RealCUGAN"
                      className="flex items-center space-x-1 px-2.5 py-1 bg-purple-950/80 hover:bg-purple-900 border border-purple-700/60 rounded text-[11px] font-semibold text-purple-200 transition disabled:opacity-50"
                    >
                      {isUpscaling ? (
                        <Loader2 className="w-3 h-3 animate-spin text-purple-400" />
                      ) : (
                        <Maximize className="w-3 h-3 text-purple-400" />
                      )}
                      <span>4x Ultra</span>
                    </button>

                    <a
                      href={currentImage.imageUrl}
                      download={`text2img_${currentImage.seed}.png`}
                      title="Download full-resolution image"
                      className="p-1 text-gray-300 hover:text-white transition"
                    >
                      <Download className="w-4 h-4" />
                    </a>
                  </div>

                  {/* Seed / Prompt Metadata Overlay */}
                  <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-4 flex items-end justify-between">
                    <div>
                      <span className="text-[11px] font-mono text-purple-300 font-semibold">
                        Seed: {currentImage.seed}
                      </span>
                      <p className="text-[10px] text-gray-300 line-clamp-1 max-w-md">
                        {currentImage.positivePrompt}
                      </p>
                    </div>
                    <span className="text-[10px] text-gray-400 font-mono">
                      {resolution} • {sampler}
                    </span>
                  </div>
                </div>

                {/* Batch Thumbnail Carousel (when output images > 1) */}
                {outputImages.length > 1 && (
                  <div className="w-full max-w-xl mt-4 flex items-center space-x-2 overflow-x-auto py-2 custom-scrollbar shrink-0">
                    {outputImages.map((img, idx) => (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => setSelectedImageIndex(idx)}
                        className={`w-14 h-14 rounded-lg overflow-hidden shrink-0 border-2 transition ${
                          selectedImageIndex === idx
                            ? "border-purple-500 shadow-md scale-105"
                            : "border-transparent opacity-60 hover:opacity-100"
                        }`}
                      >
                        <img
                          src={img.imageUrl}
                          alt={`Thumbnail ${idx}`}
                          className="w-full h-full object-cover"
                        />
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-[#21262D] rounded-2xl p-10 text-center">
                {isGenerating ? (
                  <div className="flex flex-col items-center space-y-5 max-w-md w-full px-6">
                    <div className="relative">
                      <div className="w-16 h-16 rounded-2xl bg-purple-950/60 border border-purple-800/60 flex items-center justify-center text-purple-400 shadow-xl">
                        <Loader2 className="w-8 h-8 text-purple-400 animate-spin" />
                      </div>
                      <Sparkles className="w-4 h-4 text-purple-200 absolute -top-1 -right-1 animate-pulse" />
                    </div>

                    <div className="w-full text-center space-y-2">
                      <div className="flex items-center justify-between text-xs font-mono font-semibold">
                        <span className="text-purple-300">
                          {batchProgress
                            ? `Batch Image ${batchProgress.current} of ${batchProgress.total}`
                            : progressStep > 0
                            ? `Step ${progressStep} / ${progressTotal}`
                            : "Initializing Hassaku Checkpoint..."}
                        </span>
                        <span className="text-emerald-400 font-bold px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/40">
                          {progressPercent}%
                        </span>
                      </div>

                      {/* Animated Gradient Progress Bar */}
                      <div className="w-full h-2.5 bg-[#161B22] rounded-full overflow-hidden border border-[#30363D] shadow-inner">
                        <div
                          className="h-full bg-gradient-to-r from-purple-600 via-indigo-500 to-cyan-400 rounded-full transition-all duration-300 ease-out shadow-lg"
                          style={{ width: `${Math.max(progressPercent, 4)}%` }}
                        />
                      </div>

                      <p className="text-xs text-[#8B949E] font-medium">
                        {progressMessage || "Generating with Hassaku XL Illustrious on RTX GPU..."}
                      </p>
                    </div>

                    <button
                      type="button"
                      onClick={() => cancelGeneration()}
                      className="px-4 py-1.5 bg-red-950/60 hover:bg-red-900 border border-red-800/60 text-red-300 text-xs font-semibold rounded-lg transition flex items-center space-x-1.5"
                    >
                      <Square className="w-3.5 h-3.5 fill-red-400" />
                      <span>Cancel Generation</span>
                    </button>
                  </div>
                ) : (
                  <div className="flex flex-col items-center space-y-3 max-w-md">
                    <div className="w-16 h-16 rounded-2xl bg-purple-950/40 border border-purple-800/40 flex items-center justify-center text-purple-400">
                      <ImageIcon className="w-8 h-8" />
                    </div>
                    <h3 className="text-base font-bold text-white">
                      Text to Image Studio Ready
                    </h3>
                    <p className="text-xs text-gray-400 leading-relaxed">
                      Select character LoRA, facial expression, outfit style, accessories, lighting atmosphere, pose, POV, bondage or gag restraints, or customize your prompt in the editor panel to generate high-fidelity illustrations.
                    </p>
                    <button
                      type="button"
                      onClick={() => generateSingleImage()}
                      className="mt-2 px-5 py-2.5 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white text-xs font-bold uppercase rounded-lg shadow-md transition flex items-center space-x-2"
                    >
                      <Play className="w-3.5 h-3.5 fill-white" />
                      <span>Start Generation</span>
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
export default DegenerateStudio;

