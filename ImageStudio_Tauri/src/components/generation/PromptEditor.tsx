import React from "react";
import { Sparkles, Dices, Trash2 } from "lucide-react";
import { useGeneration } from "../../state/generationContext";

export const PromptEditor: React.FC = () => {
  const { prompt, setPrompt, enhancePrompt, randomizePrompt, clearPrompt } =
    useGeneration();

  const charCount = prompt.length;
  const tokenEstimate = Math.round(prompt.trim().split(/\s+/).filter(Boolean).length * 1.3);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[11px] font-bold text-[#8993A7] tracking-wider uppercase">
        <span className="flex items-center space-x-1.5 text-[#E8ECF4]">
          <span>Prompt Workspace</span>
        </span>
        <span className="text-[10px] font-mono text-[#8993A7]">
          ~{tokenEstimate} tokens ({charCount} chars)
        </span>
      </div>

      <div className="relative">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the image you want to create: subject, character, pose, lighting, atmospheric style, art details..."
          rows={5}
          className="w-full px-3.5 py-2.5 bg-[#151A24] border border-[#252C3A] hover:border-[#353F54] focus:border-[#7C6CFF] rounded-card text-xs text-[#E8ECF4] placeholder-[#555E70] leading-relaxed resize-none transition-all custom-scrollbar"
        />
      </div>

      {/* Prompt Tools */}
      <div className="flex items-center space-x-1.5">
        <button
          type="button"
          onClick={enhancePrompt}
          className="px-2.5 py-1 rounded-control bg-[#151A24] hover:bg-[#1D2433] border border-[#252C3A] hover:border-[#35D6C5] text-[#35D6C5] text-[11px] font-semibold flex items-center space-x-1 transition-all"
        >
          <Sparkles className="w-3 h-3" />
          <span>AI Enhance</span>
        </button>

        <button
          type="button"
          onClick={randomizePrompt}
          className="px-2.5 py-1 rounded-control bg-[#151A24] hover:bg-[#1D2433] border border-[#252C3A] text-[#8993A7] hover:text-[#E8ECF4] text-[11px] font-semibold flex items-center space-x-1 transition-all"
        >
          <Dices className="w-3 h-3" />
          <span>Randomize</span>
        </button>

        <button
          type="button"
          onClick={clearPrompt}
          className="px-2 py-1 rounded-control hover:bg-[#151A24] text-[#8993A7] hover:text-[#FF5C6C] text-[11px] font-semibold flex items-center space-x-1 transition-all"
          title="Clear prompt"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
};
