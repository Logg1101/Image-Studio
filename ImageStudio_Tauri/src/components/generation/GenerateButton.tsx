import React from "react";
import { Sparkles, Square, Loader2 } from "lucide-react";
import { useGeneration } from "../../state/generationContext";

export const GenerateButton: React.FC = () => {
  const { status, generate, cancel, progressStep, progressTotal, progressMessage } =
    useGeneration();

  const isGenerating = status === "loading_model" || status === "generating";

  const percent =
    progressTotal > 0
      ? Math.min(100, Math.round((progressStep / progressTotal) * 100))
      : 0;

  return (
    <div className="space-y-2 pt-2 border-t border-[#252C3A]">
      {/* Progress Bar (Visible during active render) */}
      {isGenerating && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[10px] font-mono text-[#8993A7]">
            <span className="truncate">{progressMessage}</span>
            <span>{percent}%</span>
          </div>
          <div className="w-full h-1.5 bg-[#151A24] rounded-full overflow-hidden border border-[#252C3A]">
            <div
              className="h-full bg-gradient-to-r from-[#7C6CFF] to-[#35D6C5] transition-all duration-150"
              style={{ width: `${percent}%` }}
            />
          </div>
        </div>
      )}

      {/* Buttons */}
      <div className="flex items-center space-x-2">
        <button
          type="button"
          onClick={generate}
          disabled={isGenerating}
          className="flex-1 h-11 rounded-button btn-primary-gradient text-white text-xs font-bold tracking-wide flex items-center justify-center space-x-2 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isGenerating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Generating...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>GENERATE IMAGE</span>
            </>
          )}
        </button>

        {isGenerating && (
          <button
            type="button"
            onClick={cancel}
            className="h-11 px-4 rounded-button btn-danger-action text-xs font-bold flex items-center space-x-1.5"
            title="Stop generation"
          >
            <Square className="w-3.5 h-3.5 fill-current" />
            <span>Stop</span>
          </button>
        )}
      </div>
    </div>
  );
};
