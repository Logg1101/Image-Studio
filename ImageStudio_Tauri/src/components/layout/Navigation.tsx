import React from "react";
import { Sparkles, Users, Archive, Flame, Camera, Paintbrush, Scissors } from "lucide-react";
import { NavModule } from "../../types";

interface NavigationProps {
  activeModule: NavModule;
  onSelectModule: (module: NavModule) => void;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeModule,
  onSelectModule,
}) => {
  const tabs = [
    {
      id: "text2img" as NavModule,
      label: "Text → Image",
      icon: Sparkles,
      enabled: true,
    },
    {
      id: "multi_character" as NavModule,
      label: "Multi-Character",
      icon: Users,
      enabled: true,
      badge: "Regional • Anti-Bleed",
    },
    {
      id: "directed_t2i" as NavModule,
      label: "Directed Txt2Img",
      icon: Camera,
      enabled: true,
      badge: "Pose • POV • Outfit",
    },
    {
      id: "degenerate" as NavModule,
      label: "Degenerate Studio",
      icon: Flame,
      enabled: true,
      badge: "Prompt Rig",
    },
    {
      id: "inpaint" as NavModule,
      label: "Inpainting Lab",
      icon: Paintbrush,
      enabled: true,
      badge: "Face Lock",
    },
    {
      id: "retouch" as NavModule,
      label: "Retouch & Clean",
      icon: Scissors,
      enabled: true,
      badge: "Eraser • Cutout",
    },
    {
      id: "vault" as NavModule,
      label: "Vault",
      icon: Archive,
      enabled: true,
      badge: "History",
    },
  ];

  return (
    <nav className="h-11 px-4 bg-[#090C12] border-b border-[#252C3A] flex items-center space-x-2 shrink-0 select-none">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeModule === tab.id;

        return (
          <button
            key={tab.id}
            onClick={() => tab.enabled && onSelectModule(tab.id)}
            disabled={!tab.enabled}
            className={`
              relative flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150
              ${
                isActive
                  ? "bg-[#151A24] text-[#7C6CFF] border border-[#7C6CFF] shadow-sm"
                  : tab.enabled
                  ? "text-[#8993A7] hover:text-[#E8ECF4] hover:bg-[#151A24] border border-transparent"
                  : "text-[#555E70] cursor-not-allowed opacity-60 border border-transparent"
              }
            `}
          >
            <Icon className={`w-3.5 h-3.5 ${isActive ? "text-[#7C6CFF]" : ""}`} />
            <span>{tab.label}</span>
            {tab.badge && (
              <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#0F131B] text-[#555E70] border border-[#252C3A]">
                {tab.badge}
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
};
