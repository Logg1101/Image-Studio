import React, { useState } from "react";
import { ChevronDown, Ban } from "lucide-react";
import { useGeneration } from "../../state/generationContext";

export const NegativePrompt: React.FC = () => {
  const { negativePrompt, setNegativePrompt } = useGeneration();
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="space-y-1.5">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between text-[11px] font-bold text-[#8993A7] hover:text-[#E8ECF4] tracking-wider uppercase transition-colors"
      >
        <span className="flex items-center space-x-1.5">
          <Ban className="w-3.5 h-3.5 text-[#FF5C6C]" />
          <span>Negative Prompt (Optional)</span>
        </span>
        <ChevronDown
          className={`w-3.5 h-3.5 transition-transform duration-200 ${
            isExpanded ? "rotate-180" : ""
          }`}
        />
      </button>

      {isExpanded && (
        <textarea
          value={negativePrompt}
          onChange={(e) => setNegativePrompt(e.target.value)}
          placeholder="Elements to suppress (e.g. blurry, low quality, bad anatomy, deformed, artifacts)..."
          rows={3}
          className="w-full px-3 py-2 bg-[#151A24] border border-[#252C3A] hover:border-[#353F54] focus:border-[#7C6CFF] rounded-card text-xs text-[#E8ECF4] placeholder-[#555E70] leading-relaxed resize-none transition-all custom-scrollbar"
        />
      )}
    </div>
  );
};
