import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import {
  StoryCataloguesResponse,
  CharacterItem,
  SingleGenParams,
  StoryLineItem
} from "../types/storyStudio";
import { storyStudioService } from "../services/storyStudioService";

interface StoryStudioContextType {
  mode: "single" | "story";
  setMode: (mode: "single" | "story") => void;

  // Catalogues & Characters
  catalogues: StoryCataloguesResponse | null;
  characters: CharacterItem[];
  isLoadingCatalogues: boolean;
  refreshCharacters: () => Promise<void>;

  // Single / Batch selections
  selectedCharacter: string;
  setSelectedCharacter: (id: string) => void;
  selectedExpression: string;
  setSelectedExpression: (id: string) => void;
  selectedClothesType: string;
  setSelectedClothesType: (id: string) => void;
  selectedClothesColor: string;
  setSelectedClothesColor: (id: string) => void;
  selectedClothesDetails: string[];
  toggleClothesDetail: (id: string) => void;
  selectedLighting: string;
  setSelectedLighting: (id: string) => void;
  selectedPose: string;
  setSelectedPose: (id: string) => void;
  selectedPov: string;
  setSelectedPov: (id: string) => void;
  selectedBondageType: string;
  setSelectedBondageType: (id: string) => void;
  selectedBondage: string;
  setSelectedBondage: (id: string) => void;
  selectedBondageHands: string;
  setSelectedBondageHands: (id: string) => void;
  selectedBondageLegs: string;
  setSelectedBondageLegs: (id: string) => void;
  selectedBondageAccessories: string;
  setSelectedBondageAccessories: (id: string) => void;
  selectedGag: string;
  setSelectedGag: (id: string) => void;
  selectedScene: string;
  setSelectedScene: (id: string) => void;
  selectedStyle: string;
  setSelectedStyle: (id: string) => void;

  // Settings
  resolution: string;
  setResolution: (res: string) => void;
  sampler: string;
  setSampler: (sampler: string) => void;
  scheduler: string;
  setScheduler: (scheduler: string) => void;
  steps: number;
  setSteps: (steps: number) => void;
  cfg: number;
  setCfg: (cfg: number) => void;
  seed: number;
  setSeed: (seed: number) => void;
  batchCount: number;
  setBatchCount: (count: number) => void;
  upscaleMethod: string;
  setUpscaleMethod: (m: string) => void;
  upscaleFactor: number;
  setUpscaleFactor: (f: number) => void;

  // Prompts & Generation
  positivePrompt: string;
  setPositivePrompt: (p: string) => void;
  negativePrompt: string;
  setNegativePrompt: (p: string) => void;
  isGeneratingPrompt: boolean;
  refreshPrompt: () => Promise<void>;

  // Execution state
  isGenerating: boolean;
  isUpscaling: boolean;
  progressStep: number;
  progressTotal: number;
  progressPercent: number;
  progressMessage: string;
  batchProgress: { current: number; total: number } | null;
  outputImages: Array<{ imageUrl: string; seed: number; positivePrompt: string; path?: string }>;
  selectedImageIndex: number;
  setSelectedImageIndex: (idx: number) => void;
  errorMessage: string | null;

  // Actions
  generateSingleImage: () => Promise<void>;
  generateBatchImages: () => Promise<void>;
  upscaleCurrentImage: (scale?: number, modelId?: string) => Promise<void>;
  randomizeSelections: () => Promise<void>;
  clearSelections: () => void;
  cancelGeneration: () => Promise<void>;

  // Story Mode
  storyName: string;
  setStoryName: (name: string) => void;
  fullStoryText: string;
  setFullStoryText: (text: string) => void;
  isDecomposingStory: boolean;
  decomposeStoryToPrompts: () => Promise<void>;
  storyLines: StoryLineItem[];
  addStoryLine: (text?: string) => void;
  updateStoryLineText: (index: number, text: string) => void;
  updateScenePrompt: (index: number, positive: string, negative?: string) => void;
  deleteStoryLine: (index: number) => void;
  moveStoryLine: (fromIndex: number, toIndex: number) => void;
  storyIsRunning: boolean;
  startStoryGeneration: () => Promise<void>;
  pauseStoryGeneration: () => void;
  stopStoryGeneration: () => void;
  regenerateLine: (index: number) => Promise<void>;
}

const StoryStudioContext = createContext<StoryStudioContextType | undefined>(undefined);

export const StoryStudioProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [mode, setMode] = useState<"single" | "story">("single");

  // Catalogues & Characters
  const [catalogues, setCatalogues] = useState<StoryCataloguesResponse | null>(null);
  const [characters, setCharacters] = useState<CharacterItem[]>([]);
  const [isLoadingCatalogues, setIsLoadingCatalogues] = useState(true);

  // Selections
  const [selectedCharacter, setSelectedCharacter] = useState<string>("");
  const [selectedExpression, setSelectedExpression] = useState<string>("neutral");
  const [selectedClothesType, setSelectedClothesType] = useState<string>("maid");
  const [selectedClothesColor, setSelectedClothesColor] = useState<string>("white");
  const [selectedClothesDetails, setSelectedClothesDetails] = useState<string[]>(["lace", "ribbons"]);
  const [selectedLighting, setSelectedLighting] = useState<string>("natural_sunlight");
  const [selectedPose, setSelectedPose] = useState<string>("standing");
  const [selectedPov, setSelectedPov] = useState<string>("three_quarter_front");
  const [selectedBondageType, setSelectedBondageType] = useState<string>("none");
  const [selectedBondage, setSelectedBondage] = useState<string>("none");
  const [selectedBondageHands, setSelectedBondageHands] = useState<string>("none");
  const [selectedBondageLegs, setSelectedBondageLegs] = useState<string>("none");
  const [selectedBondageAccessories, setSelectedBondageAccessories] = useState<string>("none");
  const [selectedGag, setSelectedGag] = useState<string>("none");
  const [selectedScene, setSelectedScene] = useState<string>("getting_ready");
  const [selectedStyle, setSelectedStyle] = useState<string>("romantic_anime");

  // Settings
  const [resolution, setResolution] = useState<string>("1024x1536");
  const [sampler, setSampler] = useState<string>("Euler a");
  const [scheduler, setScheduler] = useState<string>("Normal");
  const [steps, setSteps] = useState<number>(30);
  const [cfg, setCfg] = useState<number>(5.5);
  const [seed, setSeed] = useState<number>(-1);
  const [batchCount, setBatchCount] = useState<number>(1);
  const [upscaleMethod, setUpscaleMethod] = useState<string>("None");
  const [upscaleFactor, setUpscaleFactor] = useState<number>(2.0);

  // Prompts
  const [positivePrompt, setPositivePrompt] = useState<string>("");
  const [negativePrompt, setNegativePrompt] = useState<string>("");
  const [isGeneratingPrompt, setIsGeneratingPrompt] = useState(false);

  // Execution & Progress State
  const [isGenerating, setIsGenerating] = useState(false);
  const [isUpscaling, setIsUpscaling] = useState(false);
  const [progressStep, setProgressStep] = useState<number>(0);
  const [progressTotal, setProgressTotal] = useState<number>(30);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [progressMessage, setProgressMessage] = useState<string>("Ready");
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
  const [outputImages, setOutputImages] = useState<Array<{ imageUrl: string; seed: number; positivePrompt: string; path?: string }>>([]);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Story Mode State
  const [storyName, setStoryName] = useState<string>("Mansion Encounter");
  const [fullStoryText, setFullStoryText] = useState<string>(
    "She arrives at the grand mansion and nervously looks around.\n" +
    "She enters her new private bedroom with soft warm lighting.\n" +
    "She discovers an elegant outfit waiting for her on the bed.\n" +
    "She changes into the outfit and looks at herself in the full-length mirror.\n" +
    "She hears someone approaching and turns toward the door in surprise."
  );
  const [isDecomposingStory, setIsDecomposingStory] = useState(false);
  const [storyLines, setStoryLines] = useState<StoryLineItem[]>([]);
  const [storyIsRunning, setStoryIsRunning] = useState(false);
  const [storyPaused, setStoryPaused] = useState(false);

  // Real-time Progress Polling during generation (100ms interval for ultra-smooth UI updates)
  useEffect(() => {
    if (!isGenerating && !storyIsRunning) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://127.0.0.1:8188/api/progress");
        if (res.ok) {
          const p = await res.json();
          if (p) {
            if (typeof p.step === "number") setProgressStep(p.step);
            if (typeof p.total === "number" && p.total > 0) setProgressTotal(p.total);
            if (typeof p.percent === "number") setProgressPercent(p.percent);
            if (p.message) setProgressMessage(p.message);
          }
        }
      } catch {
        // ignore polling errors
      }
    }, 100);

    return () => clearInterval(interval);
  }, [isGenerating, storyIsRunning]);

  // Load Catalogues on startup
  useEffect(() => {
    async function loadData() {
      setIsLoadingCatalogues(true);
      try {
        const [catData, charData] = await Promise.all([
          storyStudioService.fetchCatalogues(),
          storyStudioService.fetchCharacters(),
        ]);
        setCatalogues(catData);
        setCharacters(charData);

        if (charData.length > 0) {
          if (!selectedCharacter || !charData.some((c) => c.id === selectedCharacter)) {
            setSelectedCharacter(charData[0].id);
          }
        }
        if (catData.samplers && catData.samplers.length > 0) setSampler(catData.default_sampler || "Euler a");
        if (catData.schedulers && catData.schedulers.length > 0) setScheduler(catData.default_scheduler || "Normal");
      } catch (err: unknown) {
        console.error("Failed to load Story Studio initial data:", err);
      } finally {
        setIsLoadingCatalogues(false);
      }
    }
    loadData();
  }, []);

  const refreshCharacters = async () => {
    try {
      const charData = await storyStudioService.fetchCharacters();
      setCharacters(charData);
      if (charData.length > 0) {
        if (!selectedCharacter || !charData.some((c) => c.id === selectedCharacter)) {
          setSelectedCharacter(charData[0].id);
        }
      }
    } catch (err: unknown) {
      console.error("Failed to refresh characters:", err);
    }
  };

  const toggleClothesDetail = (id: string) => {
    setSelectedClothesDetails((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]
    );
  };

  const getWidthHeight = useCallback((): { width: number; height: number } => {
    if (resolution && resolution.includes("x")) {
      const parts = resolution.split("x");
      const w = parseInt(parts[0], 10);
      const h = parseInt(parts[1], 10);
      if (!isNaN(w) && !isNaN(h)) return { width: w, height: h };
    }
    return { width: 1024, height: 1536 };
  }, [resolution]);

  // Prompt Agent synthesis
  const refreshPrompt = useCallback(async () => {
    setIsGeneratingPrompt(true);
    setErrorMessage(null);
    try {
      const charObj = characters.find((c) => c.id === selectedCharacter);
      const exprObj = catalogues?.expressions?.find((e) => e.id === selectedExpression);
      const clothesTypeObj = catalogues?.clothes_types.find((c) => c.id === selectedClothesType);
      const poseObj = catalogues?.poses.find((p) => p.id === selectedPose);
      const povObj = catalogues?.povs.find((p) => p.id === selectedPov);
      const bondageTypeObj = catalogues?.bondage_types?.find((b) => b.id === selectedBondageType);
      const bondageObj = catalogues?.bondage_styles?.find((b) => b.id === selectedBondage);
      const bondageHandsObj = catalogues?.bondage_hands?.find((b) => b.id === selectedBondageHands);
      const bondageLegsObj = catalogues?.bondage_legs?.find((b) => b.id === selectedBondageLegs);
      const bondageAccessoriesObj = catalogues?.bondage_accessories?.find((b) => b.id === selectedBondageAccessories);
      const gagObj = catalogues?.gag_types?.find((g) => g.id === selectedGag);
      const sceneObj = catalogues?.scenes?.find((s) => s.id === selectedScene);

      const promptPayload = {
        character_id: selectedCharacter,
        character_name: charObj?.name,
        character_trigger: charObj?.trigger,
        character_description: charObj?.description,
        expression: selectedExpression,
        expression_prompt: exprObj?.prompt,
        clothes_type: selectedClothesType,
        clothes_type_prompt: clothesTypeObj?.prompt,
        clothes_color: selectedClothesColor,
        clothes_details: selectedClothesDetails,
        pose: selectedPose,
        pose_prompt: poseObj?.prompt,
        pov: selectedPov,
        pov_prompt: povObj?.prompt,
        bondage_type: selectedBondageType,
        bondage_type_prompt: bondageTypeObj?.prompt,
        bondage_style: selectedBondage,
        bondage_prompt: bondageObj?.prompt,
        bondage_hands: selectedBondageHands,
        bondage_hands_prompt: bondageHandsObj?.prompt,
        bondage_legs: selectedBondageLegs,
        bondage_legs_prompt: bondageLegsObj?.prompt,
        bondage_accessories: selectedBondageAccessories,
        bondage_accessories_prompt: bondageAccessoriesObj?.prompt,
        gag_type: selectedGag,
        gag_prompt: gagObj?.prompt,
        scene: selectedScene,
        scene_description: sceneObj?.prompt_context,
      };

      const res = await storyStudioService.previewPromptAgent(promptPayload);
      let updatedPrompt = res.positive_prompt;
      if (updatedPrompt && !updatedPrompt.includes("realistic fabric texture, realistic lighting, cinematic")) {
        updatedPrompt = updatedPrompt.trim();
        if (updatedPrompt.endsWith(",")) {
            updatedPrompt += " realistic fabric texture, realistic lighting, cinematic";
        } else {
            updatedPrompt += ", realistic fabric texture, realistic lighting, cinematic";
        }
      }
      setPositivePrompt(updatedPrompt);
      setNegativePrompt(res.negative_prompt);
    } catch (err: unknown) {
      console.error("Error refreshing prompt:", err);
    } finally {
      setIsGeneratingPrompt(false);
    }
  }, [
    characters,
    catalogues,
    selectedCharacter,
    selectedExpression,
    selectedClothesType,
    selectedClothesColor,
    selectedClothesDetails,
    selectedPose,
    selectedPov,
    selectedBondageType,
    selectedBondage,
    selectedBondageHands,
    selectedBondageLegs,
    selectedBondageAccessories,
    selectedGag,
    selectedScene,
  ]);

  // Auto-refresh prompt when key dropdowns change
  useEffect(() => {
    if (catalogues && characters.length > 0) {
      refreshPrompt();
    }
  }, [
    selectedCharacter,
    selectedExpression,
    selectedClothesType,
    selectedClothesColor,
    selectedClothesDetails,
    selectedPose,
    selectedPov,
    selectedBondageType,
    selectedBondage,
    selectedBondageHands,
    selectedBondageLegs,
    selectedBondageAccessories,
    selectedGag,
    selectedScene,
  ]);

  // Single Generation
  const generateSingleImage = async () => {
    setIsGenerating(true);
    setErrorMessage(null);
    setBatchProgress(null);
    setProgressStep(0);
    setProgressPercent(0);
    setProgressMessage("Starting Hassaku XL Illustrious generation...");
    try {
      const { width, height } = getWidthHeight();
      const params: SingleGenParams = {
        character_id: selectedCharacter,
        expression: selectedExpression,
        clothes_type: selectedClothesType,
        clothes_color: selectedClothesColor,
        clothes_details: selectedClothesDetails,
        pose: selectedPose,
        pov: selectedPov,
        bondage_type: selectedBondageType,
        bondage_style: selectedBondage,
        bondage_hands: selectedBondageHands,
        bondage_legs: selectedBondageLegs,
        bondage_accessories: selectedBondageAccessories,
        gag_type: selectedGag,
        scene: selectedScene,
        positive_prompt: positivePrompt,
        negative_prompt: negativePrompt,
        width,
        height,
        steps,
        cfg,
        sampler,
        scheduler,
        seed,
        upscale_method: upscaleMethod === "None" ? undefined : upscaleMethod,
        upscale_factor: upscaleFactor,
      };

      const res = await storyStudioService.generateSingle(params);
      setOutputImages((prev) => [
        {
          imageUrl: res.image_url,
          seed: res.seed,
          positivePrompt: res.positive_prompt,
          path: res.image_path,
        },
        ...prev,
      ]);
      setSelectedImageIndex(0);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(msg);
    } finally {
      setIsGenerating(false);
    }
  };

  // Batch Generation
  const generateBatchImages = async () => {
    setIsGenerating(true);
    setErrorMessage(null);
    setBatchProgress({ current: 0, total: batchCount });
    setProgressStep(0);
    setProgressPercent(0);
    setProgressMessage(`Starting Batch generation (1 of ${batchCount})...`);

    try {
      const { width, height } = getWidthHeight();
      const params: SingleGenParams = {
        character_id: selectedCharacter,
        expression: selectedExpression,
        clothes_type: selectedClothesType,
        clothes_color: selectedClothesColor,
        clothes_details: selectedClothesDetails,
        pose: selectedPose,
        pov: selectedPov,
        bondage_type: selectedBondageType,
        bondage_style: selectedBondage,
        bondage_hands: selectedBondageHands,
        bondage_legs: selectedBondageLegs,
        bondage_accessories: selectedBondageAccessories,
        gag_type: selectedGag,
        scene: selectedScene,
        positive_prompt: positivePrompt,
        negative_prompt: negativePrompt,
        width,
        height,
        steps,
        cfg,
        sampler,
        scheduler,
        seed,
        upscale_method: upscaleMethod === "None" ? undefined : upscaleMethod,
        upscale_factor: upscaleFactor,
      };

      for (let i = 0; i < batchCount; i++) {
        setBatchProgress({ current: i + 1, total: batchCount });
        setProgressMessage(`Rendering batch image ${i + 1} of ${batchCount}...`);
        const res = await storyStudioService.generateSingle({
          ...params,
          seed: seed === -1 ? -1 : seed + i,
        });
        setOutputImages((prev) => [
          {
            imageUrl: res.image_url,
            seed: res.seed,
            positivePrompt: res.positive_prompt,
            path: res.image_path,
          },
          ...prev,
        ]);
      }
      setSelectedImageIndex(0);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(msg);
    } finally {
      setIsGenerating(false);
      setBatchProgress(null);
    }
  };

  // On-demand upscale current image
  const upscaleCurrentImage = async (scale: number = 2, modelId: string = "4x-realcugan") => {
    const current = outputImages[selectedImageIndex];
    if (!current) return;
    setIsUpscaling(true);
    setErrorMessage(null);
    try {
      const res = await storyStudioService.upscaleImage({
        image_url: current.imageUrl,
        scale,
        model_id: modelId,
      });
      setOutputImages((prev) => [
        {
          imageUrl: res.image_url,
          seed: current.seed,
          positivePrompt: `${current.positivePrompt} (Upscaled ${res.scale}x ${res.model_id})`,
          path: current.path,
        },
        ...prev,
      ]);
      setSelectedImageIndex(0);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(msg);
    } finally {
      setIsUpscaling(false);
    }
  };

  // Randomize
  const randomizeSelections = async () => {
    if (!catalogues) return;
    if (characters.length > 0) {
      const randomChar = characters[Math.floor(Math.random() * characters.length)];
      setSelectedCharacter(randomChar.id);
    }
    if (catalogues.expressions && catalogues.expressions.length > 0) {
      setSelectedExpression(catalogues.expressions[Math.floor(Math.random() * catalogues.expressions.length)].id);
    }
    if (catalogues.clothes_types && catalogues.clothes_types.length > 0) {
      setSelectedClothesType(catalogues.clothes_types[Math.floor(Math.random() * catalogues.clothes_types.length)].id);
    }
    if (catalogues.clothes_colors && catalogues.clothes_colors.length > 0) {
      setSelectedClothesColor(catalogues.clothes_colors[Math.floor(Math.random() * catalogues.clothes_colors.length)].id);
    }
    if (catalogues.clothes_details && catalogues.clothes_details.length > 0) {
      const shuffled = [...catalogues.clothes_details].sort(() => 0.5 - Math.random());
      setSelectedClothesDetails(shuffled.slice(0, 2).map((d) => d.id));
    }
    if (catalogues.poses && catalogues.poses.length > 0) {
      setSelectedPose(catalogues.poses[Math.floor(Math.random() * catalogues.poses.length)].id);
    }
    if (catalogues.povs && catalogues.povs.length > 0) {
      setSelectedPov(catalogues.povs[Math.floor(Math.random() * catalogues.povs.length)].id);
    }
    if (catalogues.bondage_types && catalogues.bondage_types.length > 0) {
      setSelectedBondageType(catalogues.bondage_types[Math.floor(Math.random() * catalogues.bondage_types.length)].id);
    }
    if (catalogues.bondage_styles && catalogues.bondage_styles.length > 0) {
      setSelectedBondage(catalogues.bondage_styles[Math.floor(Math.random() * catalogues.bondage_styles.length)].id);
    }
    if (catalogues.bondage_hands && catalogues.bondage_hands.length > 0) {
      setSelectedBondageHands(catalogues.bondage_hands[Math.floor(Math.random() * catalogues.bondage_hands.length)].id);
    }
    if (catalogues.bondage_legs && catalogues.bondage_legs.length > 0) {
      setSelectedBondageLegs(catalogues.bondage_legs[Math.floor(Math.random() * catalogues.bondage_legs.length)].id);
    }
    if (catalogues.bondage_accessories && catalogues.bondage_accessories.length > 0) {
      setSelectedBondageAccessories(catalogues.bondage_accessories[Math.floor(Math.random() * catalogues.bondage_accessories.length)].id);
    }
    if (catalogues.gag_types && catalogues.gag_types.length > 0) {
      setSelectedGag(catalogues.gag_types[Math.floor(Math.random() * catalogues.gag_types.length)].id);
    }
    if (catalogues.scenes && catalogues.scenes.length > 0) {
      setSelectedScene(catalogues.scenes[Math.floor(Math.random() * catalogues.scenes.length)].id);
    }
  };

  const clearSelections = () => {
    setSelectedExpression("neutral");
    setSelectedClothesType("maid");
    setSelectedClothesColor("white");
    setSelectedClothesDetails(["lace"]);
    setSelectedPose("standing");
    setSelectedPov("three_quarter_front");
    setSelectedBondageType("none");
    setSelectedBondage("none");
    setSelectedBondageHands("none");
    setSelectedBondageLegs("none");
    setSelectedBondageAccessories("none");
    setSelectedGag("none");
    setSelectedScene("first_meeting");
    setSteps(30);
    setCfg(5.5);
    setSeed(-1);
    setBatchCount(1);
    setUpscaleMethod("None");
    setUpscaleFactor(2.0);
  };

  const cancelGeneration = async () => {
    await storyStudioService.cancel();
    setIsGenerating(false);
    setBatchProgress(null);
  };

  // Story Mode Operations
  const addStoryLine = (text: string = "") => {
    setStoryLines((prev) => [
      ...prev,
      { index: prev.length, text: text || "New story progression line...", status: "Pending" },
    ]);
  };

  const updateStoryLineText = (index: number, text: string) => {
    setStoryLines((prev) =>
      prev.map((line, i) => (i === index ? { ...line, text } : line))
    );
  };

  const deleteStoryLine = (index: number) => {
    setStoryLines((prev) =>
      prev.filter((_, i) => i !== index).map((line, i) => ({ ...line, index: i }))
    );
  };

  const decomposeStoryToPrompts = async () => {
    if (!fullStoryText || !fullStoryText.trim()) return;
    setIsDecomposingStory(true);
    setErrorMessage(null);
    try {
      const { width, height } = getWidthHeight();
      const res = await storyStudioService.decomposeStory({
        story_text: fullStoryText,
        character_id: selectedCharacter,
        default_settings: {
          clothes_type: selectedClothesType,
          clothes_color: selectedClothesColor,
          clothes_details: selectedClothesDetails,
          style: selectedStyle,
          width,
          height,
          steps,
          cfg,
          sampler,
          scheduler,
          seed,
        },
      });

      const parsedScenes: StoryLineItem[] = res.scenes.map((s, idx) => ({
        index: idx,
        text: (s.text as string) || "",
        positive_prompt: (s.positive_prompt as string) || "",
        negative_prompt: (s.negative_prompt as string) || "",
        directive: (s.directive as Record<string, unknown>) || {},
        status: "Pending" as const,
        image_url: undefined,
        seed: undefined,
      }));

      setStoryLines(parsedScenes);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(msg);
    } finally {
      setIsDecomposingStory(false);
    }
  };

  const updateScenePrompt = (index: number, positive: string, negative?: string) => {
    setStoryLines((prev) =>
      prev.map((line, i) =>
        i === index
          ? {
              ...line,
              positive_prompt: positive,
              negative_prompt: negative !== undefined ? negative : line.negative_prompt,
            }
          : line
      )
    );
  };

  const moveStoryLine = (fromIndex: number, toIndex: number) => {
    if (toIndex < 0 || toIndex >= storyLines.length) return;
    setStoryLines((prev) => {
      const next = [...prev];
      const [item] = next.splice(fromIndex, 1);
      next.splice(toIndex, 0, item);
      return next.map((line, i) => ({ ...line, index: i }));
    });
  };

  const startStoryGeneration = async () => {
    if (storyIsRunning) return;
    setStoryIsRunning(true);
    setStoryPaused(false);
    setErrorMessage(null);

    try {
      const { width, height } = getWidthHeight();
      const proj = await storyStudioService.createProject({
        name: storyName,
        story_text: fullStoryText,
        scenes: storyLines.map((l) => ({
          index: l.index,
          text: l.text,
          positive_prompt: l.positive_prompt,
          negative_prompt: l.negative_prompt,
          directive: l.directive,
          status: l.status,
          image_filename: l.image_filename,
          seed: l.seed,
        })),
        character: selectedCharacter,
        default_settings: {
          clothes_type: selectedClothesType,
          clothes_color: selectedClothesColor,
          style: selectedStyle,
          width,
          height,
          steps,
          cfg,
          sampler,
          scheduler,
          seed,
        },
      });

      for (let i = 0; i < storyLines.length; i++) {
        if (storyPaused) break;
        if (storyLines[i].status === "Completed" && storyLines[i].image_url) {
          continue; // Skip already completed lines if resuming
        }

        // Set status to Generating
        setStoryLines((prev) =>
          prev.map((l, idx) => (idx === i ? { ...l, status: "Generating" } : l))
        );

        const res = await storyStudioService.generateStoryLine(proj.name, i);

        // Update with completed result
        setStoryLines((prev) =>
          prev.map((l, idx) =>
            idx === i
              ? {
                  ...l,
                  status: "Completed",
                  image_url: res.image_url,
                  seed: res.seed,
                  prompt_data: {
                    positive_prompt: res.positive_prompt,
                    negative_prompt: res.negative_prompt,
                    directive: res.directive,
                  },
                }
              : l
          )
        );
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(msg);
    } finally {
      setStoryIsRunning(false);
    }
  };

  const pauseStoryGeneration = () => {
    setStoryPaused(true);
    setStoryIsRunning(false);
  };

  const stopStoryGeneration = () => {
    setStoryPaused(true);
    setStoryIsRunning(false);
    storyStudioService.cancel();
  };

  const regenerateLine = async (index: number) => {
    setErrorMessage(null);
    try {
      setStoryLines((prev) =>
        prev.map((l, idx) => (idx === index ? { ...l, status: "Generating" } : l))
      );
      const res = await storyStudioService.generateStoryLine(storyName, index);
      setStoryLines((prev) =>
        prev.map((l, idx) =>
          idx === index
            ? {
                ...l,
                status: "Completed",
                image_url: res.image_url,
                seed: res.seed,
                prompt_data: {
                  positive_prompt: res.positive_prompt,
                  negative_prompt: res.negative_prompt,
                  directive: res.directive,
                },
              }
            : l
        )
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(msg);
      setStoryLines((prev) =>
        prev.map((l, idx) => (idx === index ? { ...l, status: "Failed", error: msg } : l))
      );
    }
  };

  return (
    <StoryStudioContext.Provider
      value={{
        mode,
        setMode,
        catalogues,
        characters,
        isLoadingCatalogues,
        refreshCharacters,
        selectedCharacter,
        setSelectedCharacter,
        selectedExpression,
        setSelectedExpression,
        selectedClothesType,
        setSelectedClothesType,
        selectedClothesColor,
        setSelectedClothesColor,
        selectedClothesDetails,
        toggleClothesDetail,
        selectedLighting,
        setSelectedLighting,
        selectedPose,
        setSelectedPose,
        selectedPov,
        setSelectedPov,
        selectedBondageType,
        setSelectedBondageType,
        selectedBondage,
        setSelectedBondage,
        selectedBondageHands,
        setSelectedBondageHands,
        selectedBondageLegs,
        setSelectedBondageLegs,
        selectedBondageAccessories,
        setSelectedBondageAccessories,
        selectedGag,
        setSelectedGag,
        selectedScene,
        setSelectedScene,
        selectedStyle,
        setSelectedStyle,
        resolution,
        setResolution,
        sampler,
        setSampler,
        scheduler,
        setScheduler,
        steps,
        setSteps,
        cfg,
        setCfg,
        seed,
        setSeed,
        batchCount,
        setBatchCount,
        upscaleMethod,
        setUpscaleMethod,
        upscaleFactor,
        setUpscaleFactor,
        positivePrompt,
        setPositivePrompt,
        negativePrompt,
        setNegativePrompt,
        isGeneratingPrompt,
        refreshPrompt,
        isGenerating,
        isUpscaling,
        progressStep,
        progressTotal,
        progressPercent,
        progressMessage,
        batchProgress,
        outputImages,
        selectedImageIndex,
        setSelectedImageIndex,
        errorMessage,
        generateSingleImage,
        generateBatchImages,
        upscaleCurrentImage,
        randomizeSelections,
        clearSelections,
        cancelGeneration,
        storyName,
        setStoryName,
        fullStoryText,
        setFullStoryText,
        isDecomposingStory,
        decomposeStoryToPrompts,
        storyLines,
        addStoryLine,
        updateStoryLineText,
        updateScenePrompt,
        deleteStoryLine,
        moveStoryLine,
        storyIsRunning,
        startStoryGeneration,
        pauseStoryGeneration,
        stopStoryGeneration,
        regenerateLine,
      }}
    >
      {children}
    </StoryStudioContext.Provider>
  );
};

export const useStoryStudio = (): StoryStudioContextType => {
  const context = useContext(StoryStudioContext);
  if (!context) {
    throw new Error("useStoryStudio must be used within a StoryStudioProvider");
  }
  return context;
};
