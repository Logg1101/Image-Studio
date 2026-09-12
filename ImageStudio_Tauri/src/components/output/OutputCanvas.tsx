import React from "react";
import {
  Download,
  Wand2,
  Maximize2,
  ImageIcon,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Loader2,
} from "lucide-react";
import { useGeneration } from "../../state/generationContext";

export const OutputCanvas: React.FC = () => {
  const {
    result,
    status,
    progressStep,
    progressTotal,
    progressMessage,
    error,
    width,
    height,
    steps,
    selectedModelId,
  } = useGeneration();

  const isGenerating = status === "loading_model" || status === "generating";
  const displayTotal = progressTotal > 0 ? progressTotal : steps;
  const percent =
    displayTotal > 0
      ? Math.min(100, Math.round((progressStep / displayTotal) * 100))
      : 0;

  return (
    <div className="flex-1 h-full bg-[#090C12] border border-[#252C3A] rounded-card flex flex-col overflow-hidden relative shadow-inner">
      {/* Top Action / Metadata Bar */}
      <div className="h-11 px-4 bg-[#0F131B] border-b border-[#252C3A] flex items-center justify-between shrink-0 select-none">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-bold text-[#E8ECF4] flex items-center space-x-1.5">
            <ImageIcon className="w-4 h-4 text-[#4F9CFF]" />
            <span>Output Viewport</span>
          </span>

          {result && (
            <span className="text-[10px] font-mono text-[#8993A7] bg-[#151A24] px-2 py-0.5 rounded border border-[#252C3A]">
              Seed: {result.seed} ? {(result.generationTimeMs / 1000).toFixed(2)}s
            </span>
          )}

          {isGenerating && (
            <span className="text-[10px] font-mono font-bold text-[#35D6C5] bg-[#151A24] px-2 py-0.5 rounded border border-[#252C3A] flex items-center space-x-1 animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-[#35D6C5]" />
              <span>Rendering GPU Latents...</span>
            </span>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-1.5">
          <button
            type="button"
            disabled={!result || isGenerating}
            onClick={() => {
              if (result?.imageUrl) {
                const link = document.createElement("a");
                link.href = result.imageUrl;
                link.download = `imagestudio_${result.seed}.png`;
                link.click();
              }
            }}
            className="px-2.5 py-1 rounded-control bg-[#151A24] hover:bg-[#1D2433] border border-[#252C3A] text-xs font-semibold text-[#E8ECF4] flex items-center space-x-1 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            title="Download image"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Save</span>
          </button>

          <button
            type="button"
            disabled={!result || isGenerating}
            className="px-2.5 py-1 rounded-control bg-[#151A24] hover:bg-[#1D2433] border border-[#252C3A] text-xs font-semibold text-[#8993A7] hover:text-[#E8ECF4] flex items-center space-x-1 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            title="Send to Alchemy Editor (Phase 3)"
          >
            <Wand2 className="w-3.5 h-3.5" />
            <span>Edit</span>
          </button>

          <button
            type="button"
            disabled={!result || isGenerating}
            className="px-2.5 py-1 rounded-control bg-[#151A24] hover:bg-[#1D2433] border border-[#252C3A] text-xs font-semibold text-[#8993A7] hover:text-[#E8ECF4] flex items-center space-x-1 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            title="Upscale Image (Phase 3)"
          >
            <Maximize2 className="w-3.5 h-3.5" />
            <span>Upscale</span>
          </button>
        </div>
      </div>

      {/* Main Canvas Viewport */}
      <div className="flex-1 flex items-center justify-center p-6 relative overflow-hidden">
        {isGenerating ? (
          /* Prominent Live Progress Centerpiece Overlay */
          <div className="w-full max-w-md bg-[#0F131B] border border-[#252C3A] rounded-card shadow-2xl p-6 space-y-4 text-center select-none animate-in fade-in duration-200">
            <div className="w-12 h-12 rounded-2xl bg-[#151A24] border border-[#252C3A] flex items-center justify-center text-[#7C6CFF] mx-auto shadow-inner">
              <Loader2 className="w-6 h-6 animate-spin text-[#7C6CFF]" />
            </div>

            <div className="space-y-1">
              <h3 className="font-bold text-sm text-[#E8ECF4] tracking-wide flex items-center justify-center space-x-1.5">
                <Cpu className="w-4 h-4 text-[#35D6C5]" />
                <span>GPU Synthesis in Progress</span>
              </h3>
              <p className="text-xs text-[#8993A7] font-mono leading-relaxed truncate px-2">
                {progressMessage}
              </p>
            </div>

            {/* Visual Progress Bar */}
            <div className="space-y-1.5 pt-1">
              <div className="flex items-center justify-between text-xs font-mono text-[#8993A7]">
                <span>
                  Step {progressStep} / {displayTotal}
                </span>
                <span className="font-bold text-[#E8ECF4]">{percent}%</span>
              </div>
              <div className="w-full h-2.5 bg-[#151A24] rounded-full overflow-hidden border border-[#252C3A] p-0.5">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-[#7C6CFF] via-[#4F9CFF] to-[#35D6C5] transition-all duration-200 shadow-sm"
                  style={{ width: `${percent}%` }}
                />
              </div>
            </div>

            {/* Target Spec Badges */}
            <div className="flex items-center justify-center space-x-2 pt-2 text-[10px] font-mono text-[#8993A7]">
              <span className="px-2 py-0.5 rounded bg-[#151A24] border border-[#252C3A]">
                {width} ? {height}
              </span>
              <span className="px-2 py-0.5 rounded bg-[#151A24] border border-[#252C3A] truncate max-w-[200px]">
                {selectedModelId}
              </span>
            </div>
          </div>
        ) : result ? (
          <div className="relative max-w-full max-h-full flex items-center justify-center">
            <img
              src={result.imageUrl}
              alt="Generated output"
              className="max-w-full max-h-[calc(100vh-210px)] object-contain rounded-lg border border-[#252C3A] shadow-2xl transition-all"
            />
          </div>
        ) : (
          /* Calm Empty State */
          <div className="flex flex-col items-center justify-center text-center space-y-3 p-8 max-w-sm">
            <div className="w-16 h-16 rounded-2xl bg-[#151A24] border border-[#252C3A] flex items-center justify-center text-[#555E70] shadow-inner">
              <Sparkles className="w-8 h-8" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-[#E8ECF4] tracking-wide">
                OUTPUT CANVAS
              </h3>
              <p className="text-xs text-[#8993A7] mt-1 leading-relaxed">
                Click &ldquo;GENERATE IMAGE&rdquo; to begin real-time GPU synthesis.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Status Bar Footer */}
      <div className="h-8 px-4 bg-[#0F131B] border-t border-[#252C3A] flex items-center justify-between text-[11px] font-mono text-[#8993A7] select-none">
        <div className="flex items-center space-x-2 truncate">
          {status === "complete" ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-[#10B981] shrink-0" />
          ) : status === "error" ? (
            <AlertCircle className="w-3.5 h-3.5 text-[#FF5C6C] shrink-0" />
          ) : isGenerating ? (
            <span className="w-2 h-2 rounded-full bg-[#35D6C5] animate-ping shrink-0" />
          ) : (
            <span className="w-2 h-2 rounded-full bg-[#10B981] shrink-0" />
          )}
          <span className="truncate">
            {error ? `Error: ${error}` : progressMessage}
          </span>
        </div>

        {result && !isGenerating && (
          <span className="text-[10px] text-[#35D6C5] shrink-0">
            ? Ready
          </span>
        )}
      </div>
    </div>
  );
};
