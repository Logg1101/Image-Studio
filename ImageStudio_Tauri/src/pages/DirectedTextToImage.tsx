import React, { useState } from "react";
import {
  Play,
  Square,
  Shuffle,
  Download,
  Copy,
  Check,
  AlertCircle,
  Loader2,
  RefreshCw,
  Image as ImageIcon,
  Sliders,
  Maximize2,
  Layers,
  Dices,
  Trash2,
  Brain,
  Plus,
  Ban,
  Minus,
  Camera,
  Shirt,
  Compass,
} from "lucide-react";
import { useGeneration } from "../state/generationContext";
import {
  AVAILABLE_SAMPLERS,
  AVAILABLE_SCHEDULERS,
} from "../services/generationService";
import { AspectRatioPreset } from "../types";
import { LoraSelectorModal } from "../components/generation/LoraSelectorModal";
import { LoraCard } from "../components/generation/LoraCard";

// Comprehensive catalog of Poses (28 distinct options)
const EXTENDED_POSES = [
  { id: "none", name: "None / Default", prompt: "" },
  { id: "standing", name: "Standing Gracefully", prompt: "standing gracefully, elegant posture, natural stance" },
  { id: "sitting", name: "Sitting on Chair", prompt: "sitting gracefully on chair, elegant relaxed poise" },
  { id: "kneeling", name: "Kneeling (Seiza)", prompt: "kneeling on floor, seiza pose, demure posture" },
  { id: "walking", name: "Walking Forward", prompt: "walking forward gently, flowing movement, graceful stride" },
  { id: "looking_over_shoulder", name: "Looking Over Shoulder", prompt: "looking back over her shoulder, seductive glance back, arched back" },
  { id: "hands_behind_back", name: "Hands Behind Back", prompt: "hands held behind back, playful demure stance, chest forward" },
  { id: "holding_hands_together", name: "Holding Hands Together", prompt: "holding hands together in front of chest, bashful finger clasp, shy gesture" },
  { id: "sitting_on_bed", name: "Sitting on Bed", prompt: "sitting on soft bed, legs crossed gently, soft bedding around" },
  { id: "leaning_against_wall", name: "Leaning Against Wall", prompt: "leaning back against wall, relaxed tilt, one leg bent" },
  { id: "lying_down", name: "Lying on Back (Supine)", prompt: "lying on back, relaxed pose, comfortable posture, head turned to camera" },
  { id: "lying_prone", name: "Lying on Stomach (Prone)", prompt: "lying on stomach, propped on elbows, chin resting on hands, legs kicked up" },
  { id: "stretching", name: "Stretching Arms Up", prompt: "stretching arms high overhead, relaxed morning stretch, arched back" },
  { id: "crouching", name: "Crouching / Squatting", prompt: "crouching low, dynamic squat, arms resting on knees" },
  { id: "all_fours", name: "All Fours (Hands & Knees)", prompt: "on all fours, hands and knees, arched back, playful submissive posture" },
  { id: "arched_back", name: "Arched Back", prompt: "pronounced arched back, curvature, chest thrust forward, alluring stance" },
  { id: "hands_on_hips", name: "Hands on Hips", prompt: "hands placed firmly on hips, confident sassy posture, head tilted" },
  { id: "arms_crossed", name: "Arms Crossed", prompt: "arms crossed under chest, confident stance, subtle smirk" },
  { id: "leaning_forward", name: "Leaning Forward (Cleavage)", prompt: "leaning forward towards camera, intimate proximity, cleavage emphasis" },
  { id: "floating", name: "Floating / Levitation", prompt: "floating weightlessly, zero gravity, drifting in air, flowing hair and dress" },
  { id: "dynamic_action", name: "Dynamic Action Leap", prompt: "dynamic action pose, mid-air leap, flowing clothing, motion blur particles" },
  { id: "dancing", name: "Dancing / Pirouette", prompt: "dancing pose, graceful pirouette, arms aloft, flowing silhouette" },
  { id: "cat_pose", name: "Cat Pose (Feline Stance)", prompt: "playful cat pose, curled paws, feline posture, head tilted cutely" },
  { id: "w_sitting", name: "W-Sitting (Legs Folded Out)", prompt: "w-sitting on floor, legs spread and folded beside hips, demure anime posture" },
  { id: "reaching_out", name: "Reaching Hand to Viewer", prompt: "reaching one hand forward towards camera, longing expression, depth perspective" },
  { id: "tiptoe", name: "Tiptoe Stance", prompt: "standing on tiptoes, ballerina poise, extended slender legs" },
  { id: "shibari_binding", name: "Shibari Rope Binding", prompt: "bound in intricate japanese hemp shibari ropes, decorative chest diamond harness, restrained posture" },
  { id: "rope_bondage", name: "Rope Bondage Restraints", prompt: "tightly bound with heavy ropes, arms bound behind back, restricted stance" },
];

// Comprehensive catalog of POVs & Camera Angles (22 distinct options)
const EXTENDED_POVS = [
  { id: "none", name: "None / Default", prompt: "" },
  { id: "front_view", name: "Front View (Centered)", prompt: "front view, centered portrait, direct eye contact" },
  { id: "back_view", name: "Back View (From Behind)", prompt: "back view, from behind, delicate back profile" },
  { id: "side_view", name: "Side Profile View", prompt: "profile view, side angle, dramatic silhouette" },
  { id: "three_quarter_front", name: "3/4 Front View", prompt: "three-quarter front view, dynamic portrait angle" },
  { id: "three_quarter_back", name: "3/4 Back View", prompt: "three-quarter back view, partial profile from behind" },
  { id: "from_above", name: "From Above (High Angle)", prompt: "high angle shot, looking down from above, top-down perspective" },
  { id: "from_below", name: "From Below (Low Angle)", prompt: "low angle shot, looking up from below, majestic upward perspective" },
  { id: "extreme_high_angle", name: "Bird's-Eye View (90° Overhead)", prompt: "extreme high angle, direct overhead bird's-eye view, 90-degree top-down shot" },
  { id: "extreme_low_angle", name: "Worm's-Eye View (Ground Level)", prompt: "extreme low angle shot, ground level perspective, towering upward view" },
  { id: "over_the_shoulder", name: "Over the Shoulder", prompt: "over the shoulder shot, cinematic framing, first-person perspective" },
  { id: "close_up", name: "Close-up Portrait (Face & Eyes)", prompt: "close-up portrait, detailed facial features, expressive eyes, intimate focus" },
  { id: "extreme_close_up", name: "Extreme Close-up (Macro Eyes & Lips)", prompt: "extreme close-up macro shot, focused on eyes and lips, shallow depth of field" },
  { id: "full_body", name: "Full Body Shot", prompt: "full body shot, complete head-to-toe view, wide perspective" },
  { id: "medium_shot", name: "Medium Shot (Waist-Up)", prompt: "medium shot, waist-up framing, natural portrait balance" },
  { id: "cowboy_shot", name: "Cowboy Shot (Thigh-Up)", prompt: "cowboy shot, thigh-up cinematic framing, character focus" },
  { id: "dutch_angle", name: "Dutch Angle (Tilted Dynamic)", prompt: "dutch angle, tilted camera roll, dynamic cinematic framing" },
  { id: "fisheye_lens", name: "Fisheye Lens Distortion", prompt: "fisheye lens perspective, barrel distortion, wide dramatic curvature" },
  { id: "selfie_angle", name: "Handheld Selfie Angle", prompt: "handheld selfie perspective, high smartphone camera angle, arm extending forward" },
  { id: "looking_up_at_viewer", name: "Looking Up at Viewer", prompt: "looking up at viewer, submissive gaze, upturned face, looking up from floor" },
  { id: "looking_down_at_viewer", name: "Looking Down at Viewer", prompt: "looking down at viewer with commanding gaze, tall perspective, looking down" },
  { id: "wide_cinematic", name: "Cinematic Wide Panorama", prompt: "wide cinematic shot, expansive background, anamorphic widescreen framing" },
];

// Comprehensive catalog of Clothing Styles (Separated individual options, no bundling)
const EXTENDED_CLOTHES = [
  { id: "none", name: "None / Default", prompt: "" },
  { id: "maid", name: "Maid Outfit", prompt: "maid outfit, maid apron, frilled maid headdress, maid dress, white apron" },
  { id: "school_sailor", name: "School Uniform (Sailor Suit)", prompt: "japanese school sailor uniform, sailor collar, pleated skirt, necktie" },
  { id: "school_blazer", name: "School Uniform (Blazer)", prompt: "school blazer, collared dress shirt, school crest, plaid pleated skirt, school tie" },
  { id: "casual_sundress", name: "Casual Summer Sundress", prompt: "light casual sundress, sleeveless breezy dress, floral print, thin straps" },
  { id: "evening_gown", name: "Formal Evening Gown", prompt: "formal evening gown, long elegant dress, luxury velvet fabric, high side slit" },
  { id: "cocktail_dress", name: "Backless Cocktail Dress", prompt: "glamorous backless cocktail dress, tight satin dress, elegant hemline" },
  { id: "kimono", name: "Traditional Silk Kimono", prompt: "traditional japanese kimono, wide silk obi sash, ornate patterned silk, furisode sleeves" },
  { id: "yukata", name: "Summer Yukata", prompt: "cotton summer yukata, floral patterns, lightweight summer kimono, simple obi" },
  { id: "gothic_lolita", name: "Gothic Lolita Dress", prompt: "gothic lolita dress, dark lace frills, victorian corset, layered petticoat, bonnet" },
  { id: "office_lady", name: "Office Lady Business Suit", prompt: "business suit, tailored pencil skirt, collared blouse, necktie, formal blazer" },
  { id: "lingerie", name: "Sensual Lace Lingerie", prompt: "sensual delicate lace lingerie, matching satin bra and panties, sheer lace trim" },
  { id: "babydoll", name: "Sheer Babydoll Nightgown", prompt: "delicate sheer babydoll nightgown, flowing ruffled hem, translucent silk sleepwear" },
  { id: "bunny_suit", name: "Bunny Girl Leotard", prompt: "glossy bunny suit leotard, bunny ears headband, bow tie, collar, wrist cuffs, fishnets" },
  { id: "nurse", name: "Nurse Uniform", prompt: "fitted white nurse uniform, nurse cap with cross emblem, nurse dress, stethoscope" },
  { id: "bikini", name: "Two-Piece Bikini", prompt: "two-piece string bikini, halter neck swimsuit top, side-tie bikini bottoms, beachwear" },
  { id: "one_piece_swimsuit", name: "One-Piece Swimsuit", prompt: "classic one-piece swimsuit, high-leg cut, sleek form-fitting swimwear" },
  { id: "micro_bikini", name: "Micro String Bikini", prompt: "extreme micro string bikini, minimal coverage, thin string ties, revealing beachwear" },
  { id: "competition_swimsuit", name: "Competition Racing Swimsuit", prompt: "glossy competition swimsuit, hydrodynamic high-neck racing swimsuit, speedo style" },
  { id: "gym_uniform", name: "Gym Uniform (T-shirt & Shorts)", prompt: "japanese gym uniform, white athletic t-shirt, jersey shorts, gym socks" },
  { id: "bloomers", name: "Athletic Navy Bloomers", prompt: "classic navy blue athletic bloomers, retro gym bloomers, fitted high-waist shorts" },
  { id: "cyberpunk_techwear", name: "Cyberpunk Tactical Techwear", prompt: "cyberpunk techwear, tactical webbing harness, glowing neon LED trim, cargo utility straps" },
  { id: "cheongsam", name: "Chinese Cheongsam / Qipao", prompt: "chinese cheongsam qipao dress, high side slit, golden dragon embroidery, mandarin collar" },
  { id: "police_uniform", name: "Police Officer Uniform", prompt: "police uniform, peaked officer cap, tailored buttoned shirt with badge, utility belt" },
  { id: "wedding_dress", name: "Bridal Wedding Gown", prompt: "bridal wedding gown, white lace veil, intricate floral embroidery, satin corset bodice" },
  { id: "witch_robes", name: "Witch Robes & Hat", prompt: "mystical witch robes, tall pointed witch hat, wide flowing sleeves, arcane symbols" },
  { id: "latex_catsuit", name: "High-Gloss Latex Catsuit", prompt: "shiny black latex catsuit, skintight bodysuit, high-gloss specular reflections, front zipper" },
  { id: "biker_jacket", name: "Leather Biker Jacket & Skirt", prompt: "black leather motorcycle biker jacket, metal studs, zippered pockets, mini skirt" },
  { id: "oversized_hoodie", name: "Oversized Hoodie & Thigh-Highs", prompt: "cozy oversized hoodie, extra long sleeves covering hands, black thigh-high socks" },
  { id: "cheerleader", name: "Cheerleader Uniform", prompt: "colorful cheerleader uniform, cropped top, pleated cheerleader mini-skirt, pompoms" },
  { id: "miko", name: "Miko (Shrine Maiden Robes)", prompt: "traditional miko attire, white kosode robe, bright scarlet red hakama pleated trousers" },
];

// Comprehensive catalog of Outfit Colors (22 options)
const EXTENDED_COLORS = [
  { id: "none", name: "Any / Default", prompt: "" },
  { id: "white", name: "Pure White", prompt: "pure white" },
  { id: "black", name: "Jet Black", prompt: "jet black" },
  { id: "red", name: "Crimson Red", prompt: "crimson red" },
  { id: "ruby", name: "Ruby Wine Red", prompt: "deep ruby wine red" },
  { id: "blue", name: "Royal Blue", prompt: "vibrant royal blue" },
  { id: "sky_blue", name: "Sky Blue", prompt: "soft sky blue" },
  { id: "navy", name: "Navy Blue", prompt: "deep navy blue" },
  { id: "green", name: "Emerald Green", prompt: "rich emerald green" },
  { id: "mint", name: "Mint Pastel Green", prompt: "pastel mint green" },
  { id: "olive", name: "Olive Drab", prompt: "military olive green" },
  { id: "pink", name: "Pastel Pink", prompt: "delicate pastel baby pink" },
  { id: "hot_pink", name: "Hot Magenta Pink", prompt: "vibrant hot magenta pink" },
  { id: "lavender", name: "Lavender Violet", prompt: "soft lavender violet" },
  { id: "purple", name: "Deep Royal Purple", prompt: "regal deep purple" },
  { id: "gold", name: "Champagne Gold", prompt: "luxurious champagne gold" },
  { id: "silver", name: "Silver / Platinum", prompt: "metallic silver platinum" },
  { id: "charcoal", name: "Charcoal Grey", prompt: "dark charcoal grey" },
  { id: "cream", name: "Warm Cream / Beige", prompt: "warm cream beige" },
  { id: "turquoise", name: "Turquoise / Cyan", prompt: "radiant turquoise cyan" },
  { id: "rose_gold", name: "Rose Gold", prompt: "metallic rose gold" },
  { id: "burgundy", name: "Burgundy Maroon", prompt: "deep burgundy maroon" },
];

export const DirectedTextToImage: React.FC = () => {
  const {
    models,
    availableLoras,
    selectedModelId,
    setSelectedModelId,
    prompt,
    setPrompt,
    negativePrompt,
    setNegativePrompt,
    activeLoras,
    addLora,
    removeLora,
    updateLoraWeight,
    toggleLora,
    width,
    height,
    aspectRatio,
    setAspectRatio,
    setDimensions,
    steps,
    setSteps,
    cfgScale,
    setCfgScale,
    sampler,
    setSampler,
    scheduler,
    setScheduler,
    seed,
    setSeed,
    isRandomSeed,
    setIsRandomSeed,
    upscaleMethod,
    setUpscaleMethod,
    upscaleFactor,
    setUpscaleFactor,
    status,
    progressStep,
    progressTotal,
    progressMessage,
    result,
    error,
    generate,
    cancel,
    randomizePrompt,
    clearPrompt,
    refreshModelsAndLoras,
  } = useGeneration();

  const [isLoraModalOpen, setIsLoraModalOpen] = useState(false);
  const [isRefreshingModels, setIsRefreshingModels] = useState(false);
  const [copiedPrompt, setCopiedPrompt] = useState(false);
  const [copiedSeed, setCopiedSeed] = useState(false);
  const [isUpscalingStandalone, setIsUpscalingStandalone] = useState(false);
  const [upscaleError, setUpscaleError] = useState<string | null>(null);

  // Dynamic Directing Dropdowns State
  const [selectedPose, setSelectedPose] = useState<string>("none");
  const [selectedPov, setSelectedPov] = useState<string>("none");
  const [selectedClothes, setSelectedClothes] = useState<string>("none");
  const [selectedColor, setSelectedColor] = useState<string>("none");

  // Track the text actively inserted by each dropdown to allow clean in-place replacement
  const [activePoseSnippet, setActivePoseSnippet] = useState<string>("");
  const [activePovSnippet, setActivePovSnippet] = useState<string>("");
  const [activeClothesSnippet, setActiveClothesSnippet] = useState<string>("");

  // Helper function to update prompt text smoothly by replacing or appending
  const syncPromptSegment = (oldSnippet: string, newSnippet: string) => {
    let current = prompt;
    if (oldSnippet && current.includes(oldSnippet)) {
      if (newSnippet) {
        current = current.replace(oldSnippet, newSnippet);
      } else {
        // Remove cleanly
        current = current.replace(new RegExp(`,\\s*${escapeRegex(oldSnippet)}`, "g"), "");
        current = current.replace(new RegExp(`${escapeRegex(oldSnippet)},?\\s*`, "g"), "");
      }
    } else if (newSnippet) {
      const trimmed = current.trim();
      current = trimmed ? `${trimmed}, ${newSnippet}` : newSnippet;
    }
    setPrompt(cleanCommas(current));
  };

  const escapeRegex = (string: string) => {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  };

  const cleanCommas = (text: string) => {
    return text
      .replace(/,\s*,+/g, ",")
      .replace(/^,\s*/, "")
      .replace(/,\s*$/, "")
      .trim();
  };

  // Immediate Dropdown Handlers - Updates Prompt in real-time
  const handlePoseSelect = (newPoseId: string) => {
    setSelectedPose(newPoseId);
    const poseObj = EXTENDED_POSES.find((p) => p.id === newPoseId);
    const newText = poseObj && poseObj.id !== "none" ? poseObj.prompt : "";
    syncPromptSegment(activePoseSnippet, newText);
    setActivePoseSnippet(newText);
  };

  const handlePovSelect = (newPovId: string) => {
    setSelectedPov(newPovId);
    const povObj = EXTENDED_POVS.find((p) => p.id === newPovId);
    const newText = povObj && povObj.id !== "none" ? povObj.prompt : "";
    syncPromptSegment(activePovSnippet, newText);
    setActivePovSnippet(newText);
  };

  const handleClothingSelect = (clothesId: string, colorId: string) => {
    setSelectedClothes(clothesId);
    setSelectedColor(colorId);

    const clothesObj = EXTENDED_CLOTHES.find((c) => c.id === clothesId);
    const colorObj = EXTENDED_COLORS.find((col) => col.id === colorId);

    let newText = "";
    if (clothesObj && clothesObj.id !== "none" && clothesObj.prompt) {
      const colorPrefix = colorObj && colorObj.id !== "none" && colorObj.prompt ? `${colorObj.prompt} ` : "";
      newText = `${colorPrefix}${clothesObj.prompt}`;
    } else if (colorObj && colorObj.id !== "none" && colorObj.prompt) {
      newText = `${colorObj.prompt} outfit`;
    }

    syncPromptSegment(activeClothesSnippet, newText);
    setActiveClothesSnippet(newText);
  };

  const isGenerating = status === "loading_model" || status === "generating";
  
  const displayTotal = steps > 0 ? steps : (progressTotal > 0 ? progressTotal : 30);
  const percent =
    progressStep > 0 && displayTotal > 0
      ? Math.min(100, Math.round((progressStep / displayTotal) * 100))
      : 0;

  const ratios: AspectRatioPreset[] = ["1:1", "16:9", "9:16", "4:3", "3:4"];
  const upscaleMethods = ["None", "4x-RealCUGAN", "Tiled SDXL", "Lanczos", "Bicubic"];
  const upscaleFactors = [1.5, 2.0, 3.0, 4.0];

  const activePaths = new Set(activeLoras.map((l) => l.path));

  const handleRefreshModels = async () => {
    setIsRefreshingModels(true);
    await refreshModelsAndLoras();
    setTimeout(() => setIsRefreshingModels(false), 600);
  };

  const copyPromptText = () => {
    navigator.clipboard.writeText(prompt);
    setCopiedPrompt(true);
    setTimeout(() => setCopiedPrompt(false), 2000);
  };

  const copySeedText = () => {
    if (result?.seed !== undefined) {
      navigator.clipboard.writeText(result.seed.toString());
      setCopiedSeed(true);
      setTimeout(() => setCopiedSeed(false), 2000);
    }
  };

  // Direct On-Demand Upscaler for rendered preview
  const handleOnDemandUpscale = async (model: string = "4x-realcugan", scale: number = 2) => {
    if (!result?.imageUrl) return;
    setIsUpscalingStandalone(true);
    setUpscaleError(null);
    try {
      const res = await fetch("http://127.0.0.1:8188/api/story/upscale_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_url: result.imageUrl,
          model_id: model.toLowerCase(),
          scale: scale,
        }),
      });
      if (!res.ok) {
        throw new Error(`Upscale API returned status ${res.status}`);
      }
      const data = await res.json();
      if (data.image_url) {
        result.imageUrl = data.image_url;
      }
    } catch (err: unknown) {
      setUpscaleError(err instanceof Error ? err.message : "Upscaling failed");
    } finally {
      setIsUpscalingStandalone(false);
    }
  };

  const activeModel = models.find((m) => m.id === selectedModelId);

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0B0E14] text-[#E2E8F0] overflow-hidden">
      {/* Top Header Bar - Clean title and active checkpoint badge */}
      <div className="h-11 px-5 bg-[#0D1117] border-b border-[#21262D] flex items-center justify-between shrink-0 select-none">
        <div className="flex items-center space-x-3">
          <Camera className="w-4 h-4 text-[#35D6C5]" />
          <span className="font-bold text-xs tracking-wide bg-gradient-to-r from-teal-400 to-indigo-300 bg-clip-text text-transparent">
            DIRECTED TEXT → IMAGE
          </span>
          <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-teal-950/60 text-teal-300 border border-teal-800/50">
            Pose • POV • Outfit
          </span>
        </div>

        {/* Checkpoint Status Indicator */}
        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="text-[#8B949E] text-[11px]">Checkpoint:</span>
          <span className="text-white font-semibold px-2 py-0.5 rounded bg-[#161B22] border border-[#30363D]">
            {activeModel?.name || selectedModelId || "SDXL"}
          </span>
          {activeModel?.architecture && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-950/60 border border-blue-800/40 text-blue-300 font-bold uppercase">
              {activeModel.architecture}
            </span>
          )}
        </div>
      </div>

      {/* Error notification banner */}
      {(error || upscaleError) && (
        <div className="px-6 py-2 bg-red-950/80 border-b border-red-800/80 text-red-200 text-xs flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
            <span>{error || upscaleError}</span>
          </div>
        </div>
      )}

      {/* Main Workspace Body */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Left Control Column */}
        <div className="w-[540px] border-r border-[#21262D] bg-[#0D1117] flex flex-col h-full overflow-y-auto custom-scrollbar p-4 space-y-3.5 shrink-0">
          {/* 1. Checkpoint Selector Dropdown */}
          <div className="space-y-1.5 p-3 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-bold text-blue-300 uppercase tracking-wider">
                <Brain className="w-4 h-4 text-blue-400" />
                <span>Model Checkpoint</span>
              </div>
              <button
                type="button"
                onClick={handleRefreshModels}
                title="Scan models/checkpoints folder"
                className="p-1 rounded text-blue-400 hover:text-blue-200 hover:bg-blue-900/40 transition flex items-center space-x-1 text-[10px]"
              >
                <RefreshCw className={`w-3 h-3 ${isRefreshingModels ? "animate-spin text-blue-400" : ""}`} />
                <span>Rescan</span>
              </button>
            </div>

            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-blue-500 font-medium cursor-pointer"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} [{m.architecture?.toUpperCase() || "MODEL"}] {m.description ? `(${m.description})` : ""}
                </option>
              ))}
            </select>
          </div>

          {/* 2. Positive Prompt Panel (NO AI ENHANCE - Pure manual prompt workspace) */}
          <div className="space-y-1.5 p-3.5 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-purple-300 uppercase tracking-wider flex items-center space-x-1.5">
                <Compass className="w-3.5 h-3.5 text-purple-400" />
                <span>Prompt Panel</span>
              </span>
              <div className="flex items-center space-x-1 text-[10px] font-mono text-[#8B949E]">
                <span>{prompt.length} chars</span>
              </div>
            </div>

            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={7}
              placeholder="Enter your prompt here. Dropdown selections below will automatically append/update in this box..."
              className="w-full min-h-[160px] bg-[#0D1117] border border-[#30363D] rounded-lg p-3 text-xs text-white font-mono leading-relaxed focus:outline-none focus:border-purple-500 resize-y custom-scrollbar"
            />

            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center space-x-1.5">
                <button
                  type="button"
                  onClick={randomizePrompt}
                  className="px-2.5 py-1 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] text-[#8993A7] hover:text-white text-[11px] font-semibold flex items-center space-x-1 transition"
                >
                  <Dices className="w-3 h-3" />
                  <span>Idea</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    clearPrompt();
                    setActivePoseSnippet("");
                    setActivePovSnippet("");
                    setActiveClothesSnippet("");
                    setSelectedPose("none");
                    setSelectedPov("none");
                    setSelectedClothes("none");
                    setSelectedColor("none");
                  }}
                  className="p-1 rounded bg-[#0D1117] hover:bg-red-950/40 border border-[#30363D] hover:border-red-500/50 text-[#8993A7] hover:text-red-400 text-[11px] transition"
                  title="Clear prompt & reset dropdowns"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>

              <button
                type="button"
                onClick={copyPromptText}
                className="text-[11px] font-semibold text-gray-400 hover:text-white flex items-center space-x-1"
              >
                {copiedPrompt ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                <span>{copiedPrompt ? "Copied" : "Copy"}</span>
              </button>
            </div>
          </div>

          {/* 3. Directing Controls: Pose, POV, Outfit & Color (Instantly appends to prompt) */}
          <div className="p-3.5 bg-gradient-to-br from-[#121B28] via-[#161B22] to-[#121B28] rounded-xl border border-teal-800/40 shadow-sm space-y-3">
            <div className="flex items-center justify-between pb-1 border-b border-teal-900/30">
              <div className="flex items-center space-x-1.5 text-teal-300 text-xs font-bold uppercase tracking-wider">
                <Camera className="w-3.5 h-3.5 text-teal-400" />
                <span>Directing Dropdowns (Appends to Prompt)</span>
              </div>
              <span className="text-[10px] text-teal-400/80 font-mono">Live Sync</span>
            </div>

            {/* Pose & POV Grid */}
            <div className="grid grid-cols-2 gap-2.5">
              {/* Pose Dropdown */}
              <div>
                <label className="block text-[11px] font-semibold text-[#8B949E] uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Pose ({EXTENDED_POSES.length - 1})</span>
                </label>
                <select
                  value={selectedPose}
                  onChange={(e) => handlePoseSelect(e.target.value)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-teal-500 font-medium cursor-pointer"
                >
                  {EXTENDED_POSES.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* POV / Camera Angle Dropdown */}
              <div>
                <label className="block text-[11px] font-semibold text-[#8B949E] uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>POV Angle ({EXTENDED_POVS.length - 1})</span>
                </label>
                <select
                  value={selectedPov}
                  onChange={(e) => handlePovSelect(e.target.value)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-teal-500 font-medium cursor-pointer"
                >
                  {EXTENDED_POVS.map((pv) => (
                    <option key={pv.id} value={pv.id}>
                      {pv.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Outfit Style & Color Grid */}
            <div className="grid grid-cols-2 gap-2.5">
              {/* Individual Outfit Style Dropdown */}
              <div>
                <label className="block text-[11px] font-semibold text-[#8B949E] uppercase tracking-wider mb-1 flex items-center space-x-1">
                  <Shirt className="w-3 h-3 text-teal-400" />
                  <span>Outfit ({EXTENDED_CLOTHES.length - 1})</span>
                </label>
                <select
                  value={selectedClothes}
                  onChange={(e) => handleClothingSelect(e.target.value, selectedColor)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-teal-500 font-medium cursor-pointer"
                >
                  {EXTENDED_CLOTHES.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Extended Outfit Color Dropdown */}
              <div>
                <label className="block text-[11px] font-semibold text-[#8B949E] uppercase tracking-wider mb-1">
                  Color ({EXTENDED_COLORS.length - 1})
                </label>
                <select
                  value={selectedColor}
                  onChange={(e) => handleClothingSelect(selectedClothes, e.target.value)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-teal-500 font-medium cursor-pointer"
                >
                  {EXTENDED_COLORS.map((col) => (
                    <option key={col.id} value={col.id}>
                      {col.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Visual preview of currently active directing tags */}
            {(activePoseSnippet || activePovSnippet || activeClothesSnippet) && (
              <div className="pt-1 flex flex-wrap gap-1.5 items-center">
                <span className="text-[10px] text-teal-400/80 font-mono">Injected:</span>
                {activePoseSnippet && (
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-teal-950/70 border border-teal-800/60 text-teal-200 truncate max-w-[200px]" title={activePoseSnippet}>
                    Pose: {activePoseSnippet}
                  </span>
                )}
                {activePovSnippet && (
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-950/70 border border-blue-800/60 text-blue-200 truncate max-w-[200px]" title={activePovSnippet}>
                    POV: {activePovSnippet}
                  </span>
                )}
                {activeClothesSnippet && (
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950/70 border border-purple-800/60 text-purple-200 truncate max-w-[200px]" title={activeClothesSnippet}>
                    Outfit: {activeClothesSnippet}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* 4. Negative Prompt Panel */}
          <div className="space-y-1.5 p-3.5 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-rose-300 uppercase tracking-wider flex items-center space-x-1.5">
                <Ban className="w-3.5 h-3.5 text-rose-400" />
                <span>Negative Prompt Panel</span>
              </span>
              <button
                type="button"
                onClick={() => setNegativePrompt("blurry, low quality, distorted, bad anatomy, deformed, artifacts, ugly, worst quality")}
                className="text-[10px] text-[#8B949E] hover:text-white"
              >
                Default Preset
              </button>
            </div>

            <textarea
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              rows={3}
              placeholder="Elements to suppress: blurry, worst quality, deformed limbs, bad hands..."
              className="w-full min-h-[75px] bg-[#0D1117] border border-[#30363D] rounded-lg p-2.5 text-xs text-[#8B949E] font-mono leading-relaxed focus:outline-none focus:border-rose-500 resize-y custom-scrollbar"
            />
          </div>

          {/* 5. LoRA Selector Rack */}
          <div className="space-y-2 p-3.5 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-cyan-300 uppercase tracking-wider flex items-center space-x-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                <span>LoRA Adapters ({activeLoras.length})</span>
              </span>
              <button
                type="button"
                onClick={() => setIsLoraModalOpen(true)}
                className="px-2.5 py-1 rounded bg-[#0D1117] hover:bg-cyan-950/40 border border-cyan-800/50 hover:border-cyan-400 text-cyan-300 text-[11px] font-bold flex items-center space-x-1 transition"
              >
                <Plus className="w-3 h-3" />
                <span>Browse All</span>
              </button>
            </div>

            {/* Categorized Quick Select Dropdown */}
            <select
              value=""
              onChange={(e) => {
                const selected = availableLoras.find((l) => l.id === e.target.value);
                if (selected) addLora(selected);
              }}
              className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg px-2.5 py-1.5 text-xs text-cyan-300 font-medium cursor-pointer focus:outline-none focus:border-cyan-500"
            >
              <option value="">+ Quick Pick LoRA by Category...</option>
              {Array.from(new Set(availableLoras.map((l) => l.category || "General"))).sort().map((cat) => (
                <optgroup key={cat} label={`📂 ${cat}`}>
                  {availableLoras
                    .filter((l) => (l.category || "General") === cat)
                    .map((lora) => (
                      <option key={lora.id} value={lora.id}>
                        {lora.displayName}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>

            <div className="space-y-1.5 max-h-40 overflow-y-auto custom-scrollbar">
              {activeLoras.length === 0 ? (
                <div className="p-2.5 bg-[#0D1117] rounded-lg border border-dashed border-[#30363D] text-center text-xs text-[#8B949E]">
                  No LoRAs attached. Click &ldquo;+ Add LoRA&rdquo; to attach from models/loras.
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
          </div>

          {/* 6. Generation Settings Panel (Working Steps & CFG Counters, Samplers, Schedulers, Dimensions, Seed) */}
          <div className="p-3 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm space-y-3">
            <div className="flex items-center space-x-1.5 text-[#C9D1D9] text-xs font-bold uppercase tracking-wider">
              <Sliders className="w-3.5 h-3.5 text-purple-400" />
              <span>Generation Settings</span>
            </div>

            {/* Resolution Aspect Ratios */}
            <div>
              <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                Aspect Ratio Preset ({width} x {height})
              </label>
              <div className="grid grid-cols-5 gap-1">
                {ratios.map((r) => (
                  <button
                    key={r}
                    type="button"
                    onClick={() => setAspectRatio(r)}
                    className={`py-1 rounded text-xs font-bold transition ${
                      aspectRatio === r
                        ? "bg-purple-600 text-white shadow-sm"
                        : "bg-[#0D1117] text-[#8B949E] hover:text-white border border-[#30363D]"
                    }`}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>

            {/* Custom Width & Height */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] text-[#8B949E] mb-0.5">Width (px)</label>
                <input
                  type="number"
                  step={64}
                  value={width}
                  onChange={(e) => setDimensions(parseInt(e.target.value, 10) || 512, height)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1 text-xs text-white font-mono"
                />
              </div>
              <div>
                <label className="block text-[10px] text-[#8B949E] mb-0.5">Height (px)</label>
                <input
                  type="number"
                  step={64}
                  value={height}
                  onChange={(e) => setDimensions(width, parseInt(e.target.value, 10) || 512)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1 text-xs text-white font-mono"
                />
              </div>
            </div>

            {/* Sampler & Scheduler */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                  Sampler
                </label>
                <select
                  value={sampler}
                  onChange={(e) => setSampler(e.target.value)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500 font-medium cursor-pointer"
                >
                  {AVAILABLE_SAMPLERS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                  Scheduler
                </label>
                <select
                  value={scheduler}
                  onChange={(e) => setScheduler(e.target.value)}
                  className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500 font-medium cursor-pointer"
                >
                  {AVAILABLE_SCHEDULERS.map((sch) => (
                    <option key={sch} value={sch}>
                      {sch}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Interactive Steps Counter & CFG Scale Counter */}
            <div className="grid grid-cols-2 gap-3 pt-1">
              {/* Steps Counter */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px] text-[#8B949E] font-semibold uppercase">
                  <span>Steps</span>
                  <div className="flex items-center space-x-1">
                    <button
                      type="button"
                      onClick={() => setSteps(Math.max(1, steps - 1))}
                      className="w-5 h-5 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] flex items-center justify-center text-white"
                    >
                      <Minus className="w-3 h-3" />
                    </button>
                    <input
                      type="number"
                      min={1}
                      max={150}
                      value={steps}
                      onChange={(e) => setSteps(Math.max(1, Math.min(150, parseInt(e.target.value, 10) || 1)))}
                      className="w-10 bg-[#0D1117] border border-[#30363D] rounded text-center text-xs text-purple-400 font-mono font-bold py-0.5"
                    />
                    <button
                      type="button"
                      onClick={() => setSteps(Math.min(150, steps + 1))}
                      className="w-5 h-5 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] flex items-center justify-center text-white"
                    >
                      <Plus className="w-3 h-3" />
                    </button>
                  </div>
                </div>
                <input
                  type="range"
                  min="10"
                  max="60"
                  step="1"
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value))}
                  className="w-full accent-purple-500 cursor-pointer"
                />
              </div>

              {/* CFG Scale Counter */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px] text-[#8B949E] font-semibold uppercase">
                  <span>CFG Scale</span>
                  <div className="flex items-center space-x-1">
                    <button
                      type="button"
                      onClick={() => setCfgScale(Math.max(1, Number((cfgScale - 0.5).toFixed(1))))}
                      className="w-5 h-5 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] flex items-center justify-center text-white"
                    >
                      <Minus className="w-3 h-3" />
                    </button>
                    <input
                      type="number"
                      step={0.5}
                      min={1.0}
                      max={30.0}
                      value={cfgScale}
                      onChange={(e) => setCfgScale(Math.max(1, Math.min(30, parseFloat(e.target.value) || 1.0)))}
                      className="w-12 bg-[#0D1117] border border-[#30363D] rounded text-center text-xs text-purple-400 font-mono font-bold py-0.5"
                    />
                    <button
                      type="button"
                      onClick={() => setCfgScale(Math.min(30, Number((cfgScale + 0.5).toFixed(1))))}
                      className="w-5 h-5 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] flex items-center justify-center text-white"
                    >
                      <Plus className="w-3 h-3" />
                    </button>
                  </div>
                </div>
                <input
                  type="range"
                  min="1.0"
                  max="15.0"
                  step="0.5"
                  value={cfgScale}
                  onChange={(e) => setCfgScale(Number(e.target.value))}
                  className="w-full accent-purple-500 cursor-pointer"
                />
              </div>
            </div>

            {/* Seed Configuration */}
            <div>
              <div className="flex items-center justify-between text-[11px] text-[#8B949E] mb-1 font-semibold uppercase">
                <span>Seed</span>
                <label className="flex items-center space-x-1 cursor-pointer text-xs lowercase">
                  <input
                    type="checkbox"
                    checked={isRandomSeed}
                    onChange={(e) => setIsRandomSeed(e.target.checked)}
                    className="accent-purple-500 rounded"
                  />
                  <span>randomize (-1)</span>
                </label>
              </div>
              <div className="flex space-x-1.5">
                <input
                  type="number"
                  disabled={isRandomSeed}
                  value={isRandomSeed ? -1 : seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="flex-1 bg-[#0D1117] border border-[#30363D] rounded px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-purple-500 disabled:opacity-50"
                />
                <button
                  type="button"
                  disabled={isRandomSeed}
                  onClick={() => setSeed(Math.floor(Math.random() * 2147483647))}
                  className="px-2.5 py-1.5 bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] text-[#8993A7] hover:text-white rounded text-xs disabled:opacity-40"
                  title="Roll new random seed"
                >
                  <Shuffle className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* 7. Neural Upscalers Section */}
            <div className="pt-2 border-t border-[#30363D] space-y-2">
              <div className="flex items-center space-x-1.5 text-xs font-bold text-cyan-300 uppercase tracking-wider">
                <Maximize2 className="w-3.5 h-3.5 text-cyan-400" />
                <span>High-Res Neural Upscaler</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[10px] text-[#8B949E] mb-0.5">Model</label>
                  <select
                    value={upscaleMethod}
                    onChange={(e) => setUpscaleMethod(e.target.value)}
                    className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                  >
                    {upscaleMethods.map((m) => (
                      <option key={m} value={m}>
                        {m === "None" ? "None (Disabled)" : m}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] text-[#8B949E] mb-0.5">Factor</label>
                  <select
                    value={upscaleFactor}
                    disabled={upscaleMethod === "None"}
                    onChange={(e) => setUpscaleFactor(parseFloat(e.target.value))}
                    className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 disabled:opacity-40"
                  >
                    {upscaleFactors.map((f) => (
                      <option key={f} value={f}>
                        {f}x
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* 8. Bottom Generation Action Button with Accurate Progress Bar */}
          <div className="sticky bottom-0 bg-[#0D1117] pt-2 pb-1 space-y-2">
            {/* Live Progress Bar during active generation */}
            {isGenerating && (
              <div className="space-y-1.5 p-2.5 bg-[#161B22] border border-[#30363D] rounded-lg shadow-md">
                <div className="flex items-center justify-between text-[11px] font-mono text-[#8B949E]">
                  <span className="truncate max-w-[280px] text-white">
                    {progressStep > 0
                      ? `Sampling step ${progressStep} / ${displayTotal}`
                      : progressMessage || "Preparing synthesis..."}
                  </span>
                  <span className="font-bold text-cyan-300">
                    {progressStep > 0 ? `${percent}%` : "In VRAM"}
                  </span>
                </div>
                <div className="w-full h-2 bg-[#0D1117] rounded-full overflow-hidden border border-[#30363D]">
                  {progressStep > 0 ? (
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 transition-all duration-200"
                      style={{ width: `${percent}%` }}
                    />
                  ) : (
                    <div className="h-full bg-blue-500 w-1/3 animate-pulse rounded-full" />
                  )}
                </div>
              </div>
            )}

            {isGenerating ? (
              <button
                type="button"
                onClick={() => cancel()}
                className="w-full py-3 bg-red-600 hover:bg-red-700 text-white text-xs font-bold uppercase rounded-lg shadow-lg flex items-center justify-center space-x-2 transition"
              >
                <Square className="w-4 h-4 fill-white" />
                <span>Cancel Generation ({progressStep > 0 ? `${percent}%` : "Loading..."})</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={() => generate()}
                className="w-full py-3 bg-gradient-to-r from-teal-600 via-blue-600 to-purple-600 hover:from-teal-500 hover:to-purple-500 text-white text-xs font-bold uppercase tracking-wider rounded-lg shadow-lg flex items-center justify-center space-x-2 transition"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>Generate Image</span>
              </button>
            )}
          </div>
        </div>

        {/* Right Dominant Preview Pane */}
        <div className="flex-1 bg-[#090C12] flex flex-col min-w-0 h-full overflow-hidden p-6">
          {isGenerating ? (
            /* Live Centerpiece Generation Overlay */
            <div className="flex-1 flex flex-col items-center justify-center">
              <div className="w-full max-w-md bg-[#161B22] border border-[#30363D] rounded-2xl shadow-2xl p-6 space-y-4 text-center">
                <div className="w-12 h-12 rounded-2xl bg-[#0D1117] border border-[#30363D] flex items-center justify-center text-teal-400 mx-auto shadow-inner">
                  <Loader2 className="w-6 h-6 animate-spin text-teal-400" />
                </div>

                <div className="space-y-1">
                  <h3 className="font-bold text-sm text-white tracking-wide flex items-center justify-center space-x-1.5">
                    <Compass className="w-4 h-4 text-emerald-400" />
                    <span>
                      {progressStep > 0
                        ? "Denoising Latents on GPU"
                        : "Preparing Pipeline & VRAM"}
                    </span>
                  </h3>
                  <p className="text-xs text-[#8B949E] font-mono leading-relaxed truncate px-2">
                    {progressMessage || `Generating with '${activeModel?.name || selectedModelId}' on GPU...`}
                  </p>
                </div>

                {/* Progress Bar */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex items-center justify-between text-xs font-mono text-[#8993A7]">
                    <span>Step {progressStep} / {displayTotal}</span>
                    <span className="font-bold text-white">
                      {progressStep > 0 ? `${percent}%` : "Loading..."}
                    </span>
                  </div>
                  <div className="w-full h-2.5 bg-[#0D1117] rounded-full overflow-hidden border border-[#30363D] p-0.5">
                    {progressStep > 0 ? (
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 transition-all duration-200"
                        style={{ width: `${percent}%` }}
                      />
                    ) : (
                      <div className="h-full bg-gradient-to-r from-teal-600 to-blue-500 w-1/2 animate-pulse rounded-full" />
                    )}
                  </div>
                </div>

                {/* Target Specs */}
                <div className="flex items-center justify-center space-x-2 pt-2 text-[10px] font-mono text-[#8B949E]">
                  <span className="px-2 py-0.5 rounded bg-[#0D1117] border border-[#30363D]">
                    {width} x {height}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-[#0D1117] border border-[#30363D]">
                    Steps: {displayTotal} | CFG: {cfgScale}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-[#0D1117] border border-[#30363D] truncate max-w-[150px]">
                    {activeModel?.name || selectedModelId || "SDXL"}
                  </span>
                </div>
              </div>
            </div>
          ) : result ? (
            /* Rendered Image Viewport */
            <div className="flex-1 flex flex-col min-h-0 items-center justify-center relative">
              {/* Output Image */}
              <div className="relative max-h-full max-w-full flex items-center justify-center rounded-xl overflow-hidden shadow-2xl border border-[#21262D] bg-[#000000]">
                <img
                  src={result.imageUrl}
                  alt={`Generated render seed ${result.seed}`}
                  className="max-h-[calc(100vh-180px)] max-w-full object-contain rounded-lg shadow-lg"
                />
              </div>

              {/* Action Toolbar on Bottom */}
              <div className="mt-3 flex flex-wrap items-center justify-between w-full max-w-2xl px-4 py-2 bg-[#161B22] border border-[#30363D] rounded-xl shadow-lg">
                <div className="flex items-center space-x-2 text-xs font-mono text-[#8B949E]">
                  <span className="bg-[#0D1117] px-2 py-1 rounded border border-[#30363D]">
                    Seed: {result.seed}
                  </span>
                  {result.generationTimeMs && (
                    <span className="bg-[#0D1117] px-2 py-1 rounded border border-[#30363D]">
                      {(result.generationTimeMs / 1000).toFixed(2)}s
                    </span>
                  )}
                  <span className="bg-[#0D1117] px-2 py-1 rounded border border-[#30363D]">
                    {width}x{height}
                  </span>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={copySeedText}
                    className="px-2.5 py-1.5 bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] text-[#8993A7] hover:text-white rounded-lg text-xs font-semibold flex items-center space-x-1 transition"
                  >
                    {copiedSeed ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>Seed</span>
                  </button>

                  <button
                    type="button"
                    disabled={isUpscalingStandalone}
                    onClick={() => handleOnDemandUpscale("4x-realcugan", 2)}
                    className="px-3 py-1.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-bold flex items-center space-x-1.5 shadow-md disabled:opacity-50 transition"
                  >
                    {isUpscalingStandalone ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Maximize2 className="w-3.5 h-3.5" />
                    )}
                    <span>{isUpscalingStandalone ? "Upscaling..." : "Upscale 2x"}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      if (result.imageUrl) {
                        const link = document.createElement("a");
                        link.href = result.imageUrl;
                        link.download = `imagestudio_${result.seed}.png`;
                        link.click();
                      }
                    }}
                    className="px-3.5 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-bold flex items-center space-x-1.5 shadow-md transition"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download</span>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            /* Idle Placeholder Viewport */
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-3">
              <div className="w-16 h-16 rounded-2xl bg-[#161B22] border border-[#30363D] flex items-center justify-center text-[#8B949E] shadow-inner">
                <ImageIcon className="w-8 h-8 opacity-40" />
              </div>
              <div className="space-y-1 max-w-sm">
                <h3 className="font-bold text-sm text-white">No Image Generated Yet</h3>
                <p className="text-xs text-[#8B949E] leading-relaxed">
                  Select your Pose, POV, and Clothing from the dropdowns to instantly populate your prompt, adjust settings, and click &ldquo;Generate Image&rdquo;.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* LoRA Selector Modal Popover */}
      <LoraSelectorModal
        isOpen={isLoraModalOpen}
        onClose={() => setIsLoraModalOpen(false)}
        availableLoras={availableLoras}
        activeLoraPaths={activePaths}
        onSelectLora={addLora}
      />
    </div>
  );
};
