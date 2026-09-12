import React, { useState } from "react";
import { TopBar } from "./TopBar";
import { Navigation } from "./Navigation";
import { TextToImage } from "../../pages/TextToImage";
import { MultiCharacter } from "../../pages/MultiCharacter";
import { DirectedTextToImage } from "../../pages/DirectedTextToImage";
import { DegenerateStudio } from "../../pages/DegenerateStudio";
import { InpaintStudio } from "../../pages/InpaintStudio";
import { RetouchStudio } from "../../pages/RetouchStudio";
import { Vault } from "../../pages/Vault";
import { NavModule } from "../../types";

export const AppShell: React.FC = () => {
  const [activeModule, setActiveModule] = useState<NavModule>("text2img");

  return (
    <div className="w-full h-full flex flex-col bg-[#090C12] text-[#E8ECF4] overflow-hidden select-none">
      {/* Top Header */}
      <TopBar />

      {/* Global Navigation */}
      <Navigation
        activeModule={activeModule}
        onSelectModule={setActiveModule}
      />

      {/* Primary Workspace View */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {activeModule === "text2img" && <TextToImage />}
        {activeModule === "multi_character" && <MultiCharacter />}
        {activeModule === "directed_t2i" && <DirectedTextToImage />}
        {activeModule === "degenerate" && <DegenerateStudio />}
        {activeModule === "inpaint" && <InpaintStudio />}
        {activeModule === "retouch" && <RetouchStudio />}
        {activeModule === "vault" && (
          <Vault onNavigateToText2Img={() => setActiveModule("text2img")} />
        )}
      </div>
    </div>
  );
};
