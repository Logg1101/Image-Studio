import React, { useState, useEffect, useRef } from "react";
import { Search, X, Layers, Check } from "lucide-react";
import { LoraInfo } from "../../types";

interface LoraSelectorModalProps {
  isOpen: boolean;
  onClose: () => void;
  availableLoras: LoraInfo[];
  activeLoraPaths: Set<string>;
  onSelectLora: (lora: LoraInfo) => void;
}

export const LoraSelectorModal: React.FC<LoraSelectorModalProps> = ({
  isOpen,
  onClose,
  availableLoras,
  activeLoraPaths,
  onSelectLora,
}) => {
  const [query, setQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Extract unique categories from available LoRAs
  const categories = React.useMemo(() => {
    const cats = new Set<string>();
    availableLoras.forEach((l) => {
      if (l.category) cats.add(l.category);
    });
    return Array.from(cats).sort();
  }, [availableLoras]);

  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedCategory("all");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  const filtered = availableLoras.filter((l) => {
    const matchesQuery =
      l.displayName.toLowerCase().includes(query.toLowerCase()) ||
      l.filename.toLowerCase().includes(query.toLowerCase()) ||
      (l.category && l.category.toLowerCase().includes(query.toLowerCase()));

    const matchesCategory =
      selectedCategory === "all" ||
      (l.category && l.category.toLowerCase() === selectedCategory.toLowerCase());

    return matchesQuery && matchesCategory;
  });

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev < filtered.length - 1 ? prev + 1 : prev));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (filtered[selectedIndex]) {
        onSelectLora(filtered[selectedIndex]);
        onClose();
      }
    } else if (e.key === "Escape") {
      onClose();
    }
  };

  const getCategoryColor = (cat?: string) => {
    switch (cat?.toLowerCase()) {
      case "characters":
        return "bg-purple-950/60 text-purple-300 border-purple-800/40";
      case "lighting":
        return "bg-amber-950/60 text-amber-300 border-amber-800/40";
      case "textures":
      case "clothing":
        return "bg-emerald-950/60 text-emerald-300 border-emerald-800/40";
      case "styles":
        return "bg-cyan-950/60 text-cyan-300 border-cyan-800/40";
      case "poses":
        return "bg-blue-950/60 text-blue-300 border-blue-800/40";
      case "concepts":
        return "bg-rose-950/60 text-rose-300 border-rose-800/40";
      default:
        return "bg-gray-800/60 text-gray-300 border-gray-700/40";
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-xs p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-[#0F131B] border border-[#252C3A] rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Modal Header */}
        <div className="px-4 py-3 border-b border-[#252C3A] flex items-center justify-between">
          <div className="flex items-center space-x-2 font-bold text-sm text-[#E8ECF4]">
            <Layers className="w-4 h-4 text-[#7C6CFF]" />
            <span>Select LoRA Adapter</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-[#151A24] text-[#8993A7] hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Search Bar & Category Filter Pills */}
        <div className="p-3 border-b border-[#252C3A] bg-[#090C12] space-y-2">
          <div className="relative flex items-center">
            <Search className="w-4 h-4 text-[#8993A7] absolute left-3 pointer-events-none" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setSelectedIndex(0);
              }}
              placeholder="🔍 Search LoRAs by name, tags, or folder..."
              className="w-full pl-9 pr-3 py-2 bg-[#151A24] border border-[#252C3A] focus:border-[#7C6CFF] rounded-lg text-xs text-[#E8ECF4] placeholder-[#555E70]"
            />
          </div>

          {/* Category Filter Pills */}
          <div className="flex items-center space-x-1.5 overflow-x-auto py-1 custom-scrollbar">
            <button
              type="button"
              onClick={() => {
                setSelectedCategory("all");
                setSelectedIndex(0);
              }}
              className={`px-2.5 py-1 rounded-md text-[11px] font-bold transition shrink-0 ${
                selectedCategory === "all"
                  ? "bg-[#7C6CFF] text-white shadow-sm"
                  : "bg-[#151A24] text-[#8993A7] hover:text-white border border-[#252C3A]"
              }`}
            >
              All ({availableLoras.length})
            </button>
            {categories.map((cat) => {
              const count = availableLoras.filter(
                (l) => l.category?.toLowerCase() === cat.toLowerCase()
              ).length;
              return (
                <button
                  key={cat}
                  type="button"
                  onClick={() => {
                    setSelectedCategory(cat.toLowerCase());
                    setSelectedIndex(0);
                  }}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-bold transition shrink-0 flex items-center space-x-1 ${
                    selectedCategory === cat.toLowerCase()
                      ? "bg-[#7C6CFF] text-white shadow-sm"
                      : "bg-[#151A24] text-[#8993A7] hover:text-white border border-[#252C3A]"
                  }`}
                >
                  <span>{cat}</span>
                  <span className="text-[9px] opacity-70">({count})</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Results List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1 custom-scrollbar">
          {filtered.length === 0 ? (
            <div className="py-10 text-center text-xs text-[#8993A7]">
              No LoRA checkpoints found matching your criteria.
            </div>
          ) : (
            filtered.map((lora, idx) => {
              const isSelected = idx === selectedIndex;
              const isAlreadyActive = activeLoraPaths.has(lora.path);

              return (
                <button
                  key={lora.id}
                  type="button"
                  onClick={() => {
                    onSelectLora(lora);
                    onClose();
                  }}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`
                    w-full px-3 py-2.5 rounded-lg text-left text-xs flex items-center justify-between transition-colors
                    ${
                      isSelected
                        ? "bg-[#1A2234] border border-[#7C6CFF] text-white"
                        : "hover:bg-[#151A24] text-[#E8ECF4] border border-transparent"
                    }
                  `}
                >
                  <div className="flex-1 min-w-0 pr-3">
                    <div className="flex items-center space-x-2 font-bold text-xs">
                      <span>🧩</span>
                      <span className="truncate">{lora.displayName}</span>
                      {lora.category && (
                        <span
                          className={`text-[9px] font-mono px-1.5 py-0.2 rounded border uppercase font-bold ${getCategoryColor(
                            lora.category
                          )}`}
                        >
                          {lora.category}
                        </span>
                      )}
                      {lora.rank && (
                        <span className="text-[10px] font-mono px-1 rounded bg-[#0F131B] text-[#35D6C5] border border-[#252C3A]">
                          dim {lora.rank}
                        </span>
                      )}
                    </div>
                    <div className="text-[10px] text-[#8993A7] font-mono mt-0.5 truncate">
                      {lora.filename}
                    </div>
                  </div>

                  {isAlreadyActive && (
                    <span className="flex items-center space-x-1 text-[10px] font-semibold text-[#10B981] shrink-0">
                      <Check className="w-3 h-3" />
                      <span>Loaded</span>
                    </span>
                  )}
                </button>
              );
            })
          )}
        </div>

        {/* Footer info */}
        <div className="px-4 py-2 border-t border-[#252C3A] bg-[#090C12] text-[10px] font-mono text-[#8993A7] flex items-center justify-between">
          <span>↑↓ to navigate • Enter to select • Esc to close</span>
          <span>
            Showing {filtered.length} of {availableLoras.length} available
          </span>
        </div>
      </div>
    </div>
  );
};
