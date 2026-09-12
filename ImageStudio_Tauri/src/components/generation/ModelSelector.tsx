import React, { useState } from "react";
import { Brain, RotateCw, ChevronDown, Check } from "lucide-react";
import { useGeneration } from "../../state/generationContext";

export const ModelSelector: React.FC = () => {
  const { models, selectedModelId, setSelectedModelId, refreshModelsAndLoras } =
    useGeneration();
  const [isOpen, setIsOpen] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const activeModel = models.find((m) => m.id === selectedModelId) || models[0];

  const handleRefresh = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsRefreshing(true);
    await refreshModelsAndLoras();
    setTimeout(() => setIsRefreshing(false), 500);
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[11px] font-bold text-[#8993A7] tracking-wider uppercase">
        <span className="flex items-center space-x-1.5">
          <Brain className="w-3.5 h-3.5 text-[#7C6CFF]" />
          <span>Active Checkpoint</span>
        </span>
        <button
          onClick={handleRefresh}
          className="p-1 rounded hover:bg-[#151A24] text-[#8993A7] hover:text-[#E8ECF4] transition-colors"
          title="Rescan model checkpoints"
        >
          <RotateCw className={`w-3 h-3 ${isRefreshing ? "animate-spin text-[#7C6CFF]" : ""}`} />
        </button>
      </div>

      <div className="relative">
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="w-full h-10 px-3 bg-[#151A24] hover:bg-[#1D2433] border border-[#252C3A] hover:border-[#353F54] rounded-control flex items-center justify-between transition-all"
        >
          <div className="flex items-center space-x-2 truncate">
            <span className="w-2 h-2 rounded-full bg-[#10B981]" />
            <span className="text-xs font-semibold text-[#E8ECF4] truncate">
              {activeModel?.name || "Select Checkpoint"}
            </span>
            {activeModel?.architecture && (
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#0F131B] border border-[#252C3A] text-[#35D6C5] shrink-0">
                {activeModel.architecture.toUpperCase()}
              </span>
            )}
          </div>
          <ChevronDown className={`w-4 h-4 text-[#8993A7] transition-transform ${isOpen ? "rotate-180" : ""}`} />
        </button>

        {isOpen && (
          <>
            <div
              className="fixed inset-0 z-40"
              onClick={() => setIsOpen(false)}
            />
            <div className="absolute top-full left-0 w-full mt-1.5 bg-[#0F131B] border border-[#252C3A] rounded-card shadow-2xl z-50 p-1 space-y-1">
              {models.map((model) => {
                const isSelected = model.id === selectedModelId;
                return (
                  <button
                    key={model.id}
                    onClick={() => {
                      setSelectedModelId(model.id);
                      setIsOpen(false);
                    }}
                    className={`w-full px-3 py-2 rounded-lg text-left text-xs flex items-center justify-between transition-colors ${
                      isSelected
                        ? "bg-[#1A2234] border border-[#7C6CFF] text-white"
                        : "hover:bg-[#151A24] text-[#E8ECF4]"
                    }`}
                  >
                    <div>
                      <div className="font-semibold">{model.name}</div>
                      {model.description && (
                        <div className="text-[10px] text-[#8993A7] mt-0.5 line-clamp-1">
                          {model.description}
                        </div>
                      )}
                    </div>
                    {isSelected && <Check className="w-3.5 h-3.5 text-[#7C6CFF] shrink-0 ml-2" />}
                  </button>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
};
