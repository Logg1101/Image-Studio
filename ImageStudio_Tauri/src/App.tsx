import React from "react";
import { GenerationProvider } from "./state/generationContext";
import { StoryStudioProvider } from "./state/storyStudioContext";
import { AppShell } from "./components/layout/AppShell";

export const App: React.FC = () => {
  return (
    <GenerationProvider>
      <StoryStudioProvider>
        <AppShell />
      </StoryStudioProvider>
    </GenerationProvider>
  );
};

export default App;
