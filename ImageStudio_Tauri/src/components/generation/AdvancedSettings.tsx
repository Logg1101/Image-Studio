import React, { useState } from "react";
import { ChevronDown, Sliders, Shuffle, Maximize } from "lucide-react";
import { useGeneration } from "../../state/generationContext";
import { AspectRatioPreset } from "../../types";
import {
  AVAILABLE_SAMPLERS,
  AVAILABLE_SCHEDULERS,
} from "../../services/generationService";

export const AdvancedSettings: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const {
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
  } = useGeneration();

  const ratios: AspectRatioPreset[] = ["1:1", "16:9", "9:16", "4:3", "3:4"];
  const upscaleMethods = ["None", "4x-RealCUGAN", "Tiled SDXL", "Lanczos", "Bicubic"];
  const upscaleFactors = [1.5, 2.0, 3.0, 4.0];

  return (
    <div className="space-y-2 pt-1 border-t border-[#252C3A]/60">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between text-[11px] font-bold text-[#8993A7] hover:text-[#E8ECF4] tracking-wider uppercase transition-colors"
      >
        <span className="flex items-center space-x-1.5">
          <Sliders className="w-3.5 h-3.5 text-[#7C6CFF]" />
          <span>Advanced Settings</span>
        </span>
        <ChevronDown
          className={`w-3.5 h-3.5 transition-transform duration-200 ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {isOpen && (
        <div className="p-3 bg-[#151A24] border border-[#252C3A] rounded-card space-y-4">
          {/* Aspect Ratio Buttons */}
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-[#8993A7] uppercase tracking-wider">
              Aspect Ratio
            </label>
            <div className="grid grid-cols-5 gap-1.5">
              {ratios.map((r) => {
                const isSelected = aspectRatio === r;
                return (
                  <button
                    key={r}
                    type="button"
                    onClick={() => setAspectRatio(r)}
                    className={`
                      py-1.5 rounded-control text-xs font-semibold transition-all
                      ${
                        isSelected
                          ? "bg-[#7C6CFF] text-white shadow-sm"
                          : "bg-[#0F131B] border border-[#252C3A] hover:border-[#353F54] text-[#8993A7] hover:text-[#E8ECF4]"
                      }
                    `}
                  >
                    {r}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Width & Height Inputs */}
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <label className="text-[10px] font-bold text-[#8993A7] uppercase">
                Width (px)
              </label>
              <input
                type="number"
                step={64}
                min={256}
                max={2048}
                value={width}
                onChange={(e) => setDimensions(parseInt(e.target.value, 10) || 512, height)}
                className="w-full px-2.5 py-1.5 bg-[#0F131B] border border-[#252C3A] rounded-control font-mono text-xs text-[#E8ECF4]"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[10px] font-bold text-[#8993A7] uppercase">
                Height (px)
              </label>
              <input
                type="number"
                step={64}
                min={256}
                max={2048}
                value={height}
                onChange={(e) => setDimensions(width, parseInt(e.target.value, 10) || 512)}
                className="w-full px-2.5 py-1.5 bg-[#0F131B] border border-[#252C3A] rounded-control font-mono text-xs text-[#E8ECF4]"
              />
            </div>
          </div>

          {/* Steps & CFG */}
          <div className="space-y-3">
            {/* Steps */}
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[10px] font-bold text-[#8993A7] uppercase">
                  Sampling Steps
                </span>
                <span className="font-mono font-bold text-[#E8ECF4]">{steps}</span>
              </div>
              <input
                type="range"
                min={10}
                max={60}
                step={1}
                value={steps}
                onChange={(e) => setSteps(parseInt(e.target.value, 10))}
                className="w-full h-1.5 bg-[#0F131B] rounded appearance-none cursor-pointer accent-[#7C6CFF]"
              />
            </div>

            {/* CFG Guidance */}
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[10px] font-bold text-[#8993A7] uppercase">
                  CFG Guidance
                </span>
                <span className="font-mono font-bold text-[#E8ECF4]">
                  {cfgScale.toFixed(1)}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={15}
                step={0.5}
                value={cfgScale}
                onChange={(e) => setCfgScale(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-[#0F131B] rounded appearance-none cursor-pointer accent-[#7C6CFF]"
              />
            </div>
          </div>

          {/* Sampler & Scheduler */}
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <label className="text-[10px] font-bold text-[#8993A7] uppercase">
                Sampler
              </label>
              <select
                value={sampler}
                onChange={(e) => setSampler(e.target.value)}
                className="w-full px-2 py-1.5 bg-[#0F131B] border border-[#252C3A] rounded-control text-xs text-[#E8ECF4] font-medium"
              >
                {AVAILABLE_SAMPLERS.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label className="text-[10px] font-bold text-[#8993A7] uppercase">
                Scheduler
              </label>
              <select
                value={scheduler}
                onChange={(e) => setScheduler(e.target.value)}
                className="w-full px-2 py-1.5 bg-[#0F131B] border border-[#252C3A] rounded-control text-xs text-[#E8ECF4] font-medium"
              >
                {AVAILABLE_SCHEDULERS.map((sc) => (
                  <option key={sc} value={sc}>
                    {sc}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* High-Res Upscaler Section */}
          <div className="space-y-2 pt-2 border-t border-[#252C3A]/60">
            <div className="flex items-center space-x-1.5 text-[10px] font-bold text-[#8993A7] uppercase tracking-wider">
              <Maximize className="w-3.5 h-3.5 text-[#35D6C5]" />
              <span>High-Res Upscaler</span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <label className="text-[10px] font-semibold text-[#8993A7]">
                  Upscale Method
                </label>
                <select
                  value={upscaleMethod}
                  onChange={(e) => setUpscaleMethod(e.target.value)}
                  className="w-full px-2 py-1.5 bg-[#0F131B] border border-[#252C3A] rounded-control text-xs text-[#E8ECF4] font-medium"
                >
                  {upscaleMethods.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-[10px] font-semibold text-[#8993A7]">
                  Scale Factor
                </label>
                <select
                  value={upscaleFactor}
                  disabled={upscaleMethod === "None"}
                  onChange={(e) => setUpscaleFactor(parseFloat(e.target.value))}
                  className="w-full px-2 py-1.5 bg-[#0F131B] border border-[#252C3A] rounded-control text-xs text-[#E8ECF4] font-medium disabled:opacity-40"
                >
                  {upscaleFactors.map((f) => (
                    <option key={f} value={f}>
                      {f}x ({Math.round(width * f)} ? {Math.round(height * f)})
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Seed Input */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-[10px] font-bold text-[#8993A7] uppercase">
                Seed
              </label>
              <label className="flex items-center space-x-1.5 text-xs text-[#E8ECF4] cursor-pointer">
                <input
                  type="checkbox"
                  checked={isRandomSeed}
                  onChange={(e) => setIsRandomSeed(e.target.checked)}
                  className="rounded bg-[#0F131B] border-[#252C3A] text-[#7C6CFF]"
                />
                <span className="text-[10px] font-semibold text-[#8993A7]">Random</span>
              </label>
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="number"
                disabled={isRandomSeed}
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
                className={`
                  flex-1 px-2.5 py-1.5 bg-[#0F131B] border border-[#252C3A] rounded-control font-mono text-xs
                  ${isRandomSeed ? "text-[#555E70] opacity-60" : "text-[#E8ECF4]"}
                `}
              />
              <button
                type="button"
                disabled={isRandomSeed}
                onClick={() => setSeed(Math.floor(Math.random() * 2147483647))}
                className="p-1.5 rounded-control bg-[#0F131B] border border-[#252C3A] hover:border-[#353F54] text-[#8993A7] hover:text-[#E8ECF4] disabled:opacity-40"
                title="Randomize seed"
              >
                <Shuffle className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
