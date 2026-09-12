import React from "react";
import { GripVertical, X } from "lucide-react";
import { ActiveLora } from "../../types";

interface LoraCardProps {
  lora: ActiveLora;
  onWeightChange: (id: string, weight: number) => void;
  onToggle: (id: string) => void;
  onRemove: (id: string) => void;
}

export const LoraCard: React.FC<LoraCardProps> = ({
  lora,
  onWeightChange,
  onToggle,
  onRemove,
}) => {
  return (
    <div
      className={`
        group relative px-3 py-2 bg-[#151A24] border rounded-card transition-all
        ${lora.enabled ? "border-[#252C3A] hover:border-[#353F54]" : "border-[#1D2433] opacity-60"}
      `}
      title={`File: ${lora.filename}\nPath: ${lora.path}`}
    >
      {/* Top Row: Grip + Checkbox + Clean Name + Weight Spinner + Remove */}
      <div className="flex items-center space-x-2">
        <GripVertical className="w-3.5 h-3.5 text-[#555E70] cursor-grab active:cursor-grabbing shrink-0" />

        <input
          type="checkbox"
          checked={lora.enabled}
          onChange={() => onToggle(lora.id)}
          className="w-3.5 h-3.5 rounded bg-[#0F131B] border-[#252C3A] text-[#7C6CFF] focus:ring-0 focus:ring-offset-0 cursor-pointer"
        />

        <div className="flex-1 truncate font-semibold text-xs text-[#E8ECF4] flex items-center space-x-1.5">
          <span className="text-[11px]">🧩</span>
          <span className="truncate">{lora.displayName}</span>
        </div>

        {/* Weight Spinner */}
        <input
          type="number"
          step={0.05}
          min={-2.0}
          max={2.0}
          value={lora.weight}
          disabled={!lora.enabled}
          onChange={(e) => onWeightChange(lora.id, parseFloat(e.target.value) || 0)}
          className="w-14 px-1.5 py-0.5 bg-[#0F131B] border border-[#252C3A] rounded text-right font-mono text-[11px] font-bold text-[#E8ECF4] focus:border-[#7C6CFF]"
        />

        {/* Remove Button */}
        <button
          type="button"
          onClick={() => onRemove(lora.id)}
          className="p-1 rounded hover:bg-[#1D2433] text-[#8993A7] hover:text-[#FF5C6C] transition-colors"
          title="Remove LoRA"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Bottom Row: Inline Slider */}
      <div className="mt-2 pl-6 pr-1 flex items-center">
        <input
          type="range"
          min={-200}
          max={200}
          value={Math.round(lora.weight * 100)}
          disabled={!lora.enabled}
          onChange={(e) => onWeightChange(lora.id, parseInt(e.target.value, 10) / 100)}
          className="w-full h-1 bg-[#0F131B] rounded-lg appearance-none cursor-pointer accent-[#7C6CFF]"
        />
      </div>
    </div>
  );
};
