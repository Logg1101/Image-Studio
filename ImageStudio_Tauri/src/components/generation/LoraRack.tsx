import React, { useState } from "react";
import { Plus } from "lucide-react";
import { useGeneration } from "../../state/generationContext";
import { LoraCard } from "./LoraCard";
import { LoraSelectorModal } from "./LoraSelectorModal";

export const LoraRack: React.FC = () => {
  const {
    availableLoras,
    activeLoras,
    addLora,
    removeLora,
    updateLoraWeight,
    toggleLora,
  } = useGeneration();

  const [isModalOpen, setIsModalOpen] = useState(false);

  const activePaths = new Set(activeLoras.map((l) => l.path));

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center justify-between text-[11px] font-bold text-[#8993A7] tracking-wider uppercase">
        <span className="flex items-center space-x-1.5">
          <span>🎨 Active LoRAs ({activeLoras.length})</span>
        </span>
        <button
          type="button"
          onClick={() => setIsModalOpen(true)}
          className="px-2.5 py-1 rounded-control bg-[#151A24] hover:bg-[#1D2433] border border-[#252C3A] hover:border-[#7C6CFF] text-[#4F9CFF] hover:text-[#E8ECF4] text-[11px] font-bold flex items-center space-x-1 transition-all"
        >
          <Plus className="w-3 h-3" />
          <span>Add LoRA</span>
        </button>
      </div>

      {/* LoRA Stack */}
      <div className="space-y-2">
        {activeLoras.length === 0 ? (
          <div className="px-3 py-3 rounded-card bg-[#151A24]/40 border border-dashed border-[#252C3A] text-center text-xs text-[#8993A7]">
            No LoRA adapters loaded. Click &ldquo;+ Add LoRA&rdquo; to attach.
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

      {/* Searchable Modal Popover */}
      <LoraSelectorModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        availableLoras={availableLoras}
        activeLoraPaths={activePaths}
        onSelectLora={addLora}
      />
    </div>
  );
};
