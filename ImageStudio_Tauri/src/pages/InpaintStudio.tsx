import React, { useState, useRef } from "react";
import {
  Paintbrush,
  Eraser,
  RotateCcw,
  Sparkles,
  Upload,
  Download,
  Copy,
  Check,
  AlertCircle,
  Loader2,
  Sliders,
  Layers,
  Dices,
  Trash2,
  Brain,
  Plus,
  Ban,
  Shield,
  ArrowRightLeft,
} from "lucide-react";
import { useGeneration } from "../state/generationContext";
import {
  AVAILABLE_SAMPLERS,
  AVAILABLE_SCHEDULERS,
  generationService,
} from "../services/generationService";
import { LoraSelectorModal } from "../components/generation/LoraSelectorModal";
import { LoraCard } from "../components/generation/LoraCard";
import { GenerationRequest } from "../types";

export const InpaintStudio: React.FC = () => {
  const {
    models,
    availableLoras,
    selectedModelId,
    setSelectedModelId,
    activeLoras,
    addLora,
    removeLora,
    updateLoraWeight,
    toggleLora,
    sampler,
    setSampler,
    scheduler,
    setScheduler,
    steps,
    setSteps,
    cfgScale,
    setCfgScale,
    seed,
    setSeed,
  } = useGeneration();

  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 2147483647));
  };

  // Inpainting Canvas & Mask State
  const [baseImageSrc, setBaseImageSrc] = useState<string | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number }>({
    width: 1024,
    height: 1024,
  });

  const [tool, setTool] = useState<"brush" | "eraser">("brush");
  const [brushSize, setBrushSize] = useState<number>(36);
  const [maskMode, setMaskMode] = useState<"inpaint_painted" | "protect_painted">("inpaint_painted");
  const [maskBlur, setMaskBlur] = useState<number>(2);
  const [denoisingStrength, setDenoisingStrength] = useState<number>(0.85);
  const [isDetectingFace, setIsDetectingFace] = useState<boolean>(false);

  // Prompts
  const [prompt, setPrompt] = useState<string>(
    "1girl, wearing dark blue sheer embroidered saree, delicate gold lace blouse, intricate zari borders, raised threads, jewelry, chandelier lighting, high quality, masterpiece"
  );
  const [negativePrompt, setNegativePrompt] = useState<string>(
    "blurry, low quality, distorted, bad anatomy, deformed, artifacts, ugly, worst quality, extra limbs"
  );

  // Execution & Progress State
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [progress, setProgress] = useState<{ step: number; total: number; message: string }>({
    step: 0,
    total: 30,
    message: "",
  });
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"result" | "original" | "mask">("result");
  const [copiedPrompt, setCopiedPrompt] = useState(false);
  const [isLoraModalOpen, setIsLoraModalOpen] = useState(false);

  // Canvas Refs
  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const isDrawingRef = useRef<boolean>(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);

  // Load image onto background canvas
  const handleImageLoad = (imgSrc: string) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // Scale large images proportionally to fit within 1280px (multiple of 8 for SDXL)
      let nw = img.naturalWidth;
      let nh = img.naturalHeight;
      const maxDim = 1280;
      if (nw > maxDim || nh > maxDim) {
        const ratio = Math.min(maxDim / nw, maxDim / nh);
        nw = Math.round(nw * ratio);
        nh = Math.round(nh * ratio);
      }
      const w = Math.round(nw / 8) * 8;
      const h = Math.round(nh / 8) * 8;
      setImageDimensions({ width: w, height: h });
      setBaseImageSrc(imgSrc);
      setResultImage(null);

      setTimeout(() => {
        // Draw image onto background canvas
        const imgCanvas = imageCanvasRef.current;
        if (imgCanvas) {
          imgCanvas.width = w;
          imgCanvas.height = h;
          const ctx = imgCanvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, w, h);
            ctx.drawImage(img, 0, 0, w, h);
          }
        }

        // Initialize empty mask canvas
        const maskCanvas = maskCanvasRef.current;
        if (maskCanvas) {
          maskCanvas.width = w;
          maskCanvas.height = h;
          const mCtx = maskCanvas.getContext("2d");
          if (mCtx) {
            mCtx.clearRect(0, 0, w, h);
          }
        }
      }, 50);
    };
    img.src = imgSrc;
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        if (evt.target?.result) {
          handleImageLoad(evt.target.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  // Helper to convert mouse/touch event to canvas coordinates
  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } | null => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const drawStroke = (from: { x: number; y: number }, to: { x: number; y: number }) => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.lineWidth = brushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (tool === "brush") {
      ctx.globalCompositeOperation = "source-over";
      // Semi-transparent ruby red for mask overlay
      ctx.strokeStyle = "rgba(244, 63, 94, 0.65)";
      ctx.fillStyle = "rgba(244, 63, 94, 0.65)";
    } else {
      // Eraser removes painted mask
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "rgba(0, 0, 0, 1)";
    }

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoords(e);
    if (!coords) return;
    isDrawingRef.current = true;
    lastPosRef.current = coords;
    drawStroke(coords, coords);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current || !lastPosRef.current) return;
    const coords = getCanvasCoords(e);
    if (!coords) return;
    drawStroke(lastPosRef.current, coords);
    lastPosRef.current = coords;
  };

  const handleMouseUp = () => {
    isDrawingRef.current = false;
    lastPosRef.current = null;
  };

  const clearMask = () => {
    const canvas = maskCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  };

  const invertMask = () => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const alpha = data[i + 3];
      if (alpha > 20) {
        // Was painted -> make transparent
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 0;
      } else {
        // Was empty -> paint with ruby mask
        data[i] = 244;
        data[i + 1] = 63;
        data[i + 2] = 94;
        data[i + 3] = 165;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  };

  // AI-Assisted Auto Face Detection & Precise Masking
  const handleAutoDetectFace = async () => {
    if (!baseImageSrc) {
      setErrorMessage("Please upload or provide an image first.");
      return;
    }
    setIsDetectingFace(true);
    setErrorMessage(null);
    try {
      const res = await generationService.detectFace(baseImageSrc);
      if (!res.success || !res.maskBase64) {
        setErrorMessage(res.error || "No face detected in the image.");
        return;
      }
      const maskImg = new Image();
      maskImg.crossOrigin = "anonymous";
      maskImg.onload = () => {
        const maskCanvas = maskCanvasRef.current;
        if (!maskCanvas) return;
        const ctx = maskCanvas.getContext("2d");
        if (!ctx) return;

        ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

        const offscreen = document.createElement("canvas");
        offscreen.width = maskCanvas.width;
        offscreen.height = maskCanvas.height;
        const offCtx = offscreen.getContext("2d");
        if (offCtx) {
          offCtx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height);
          const maskData = offCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
          const overlayData = ctx.createImageData(maskCanvas.width, maskCanvas.height);

          for (let i = 0; i < maskData.data.length; i += 4) {
            const val = maskData.data[i];
            if (val > 50) {
              overlayData.data[i] = 244; // R
              overlayData.data[i + 1] = 63; // G
              overlayData.data[i + 2] = 94; // B
              overlayData.data[i + 3] = 175; // Semi-transparent Alpha
            }
          }
          ctx.putImageData(overlayData, 0, 0);
        }
      };
      maskImg.src = res.maskBase64;
    } catch (e: any) {
      setErrorMessage(e?.message || "Face detection failed.");
    } finally {
      setIsDetectingFace(false);
    }
  };

  // Build binary black & white mask base64 (255 = inpaint, 0 = keep unchanged)
  const exportBinaryMaskBase64 = (): string | null => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return null;

    const offscreen = document.createElement("canvas");
    offscreen.width = canvas.width;
    offscreen.height = canvas.height;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return null;

    const srcCtx = canvas.getContext("2d");
    if (!srcCtx) return null;

    const srcData = srcCtx.getImageData(0, 0, canvas.width, canvas.height);
    const outData = offCtx.createImageData(canvas.width, canvas.height);

    let paintedCount = 0;
    for (let i = 0; i < srcData.data.length; i += 4) {
      if (srcData.data[i + 3] > 20) {
        paintedCount++;
      }
    }

    if (paintedCount === 0) {
      if (maskMode === "protect_painted") {
        setErrorMessage("No face painted to lock! Please paint the face or click 'Auto-Detect Face'.");
      } else {
        setErrorMessage("No area painted to inpaint! Please paint the attire or region you want regenerated.");
      }
      return null;
    }

    for (let i = 0; i < srcData.data.length; i += 4) {
      const alpha = srcData.data[i + 3];
      const isPainted = alpha > 20;

      // If maskMode === 'protect_painted' (Face Lock):
      // The painted area (face) becomes BLACK (0) to protect it.
      // The unpainted area (body/clothes/background) becomes WHITE (255) to inpaint it!
      // If maskMode === 'inpaint_painted':
      // The painted area becomes WHITE (255) to inpaint it.
      const shouldInpaint =
        maskMode === "protect_painted" ? !isPainted : isPainted;

      const val = shouldInpaint ? 255 : 0;
      outData.data[i] = val; // R
      outData.data[i + 1] = val; // G
      outData.data[i + 2] = val; // B
      outData.data[i + 3] = 255; // Fully opaque
    }

    offCtx.putImageData(outData, 0, 0);
    return offscreen.toDataURL("image/png");
  };

  const executeInpaint = async () => {
    if (!baseImageSrc) {
      setErrorMessage("Please upload or provide a base image first.");
      return;
    }

    const binaryMask = exportBinaryMaskBase64();
    if (!binaryMask) {
      setErrorMessage("Failed to export inpainting mask.");
      return;
    }

    setIsGenerating(true);
    setErrorMessage(null);
    setProgress({ step: 0, total: steps, message: "Initializing SDXL Inpainting Pipeline..." });

    try {
      const lorasPayload: Record<string, number> = {};
      activeLoras
        .filter((l) => l.enabled)
        .forEach((l) => {
          lorasPayload[l.path] = l.weight;
        });

      const req: GenerationRequest = {
        modelId: selectedModelId || "hassakuXLIllustrious_v22",
        prompt: prompt.trim() || "masterpiece, high quality",
        negativePrompt: negativePrompt.trim(),
        width: imageDimensions.width,
        height: imageDimensions.height,
        steps,
        cfgScale,
        sampler,
        scheduler,
        seed: seed === -1 ? Math.floor(Math.random() * 2147483647) : seed,
        loras: lorasPayload,
        initImage: baseImageSrc,
        maskImage: binaryMask,
        maskBlur,
        denoisingStrength,
      };

      const result = await generationService.generate(req, (step, total, msg) => {
        setProgress({ step, total, message: msg });
      });

      setResultImage(result.imageUrl);
      setViewMode("result");
    } catch (err: any) {
      setErrorMessage(err?.message || "Inpainting generation failed.");
    } finally {
      setIsGenerating(false);
    }
  };

  const useResultAsInput = () => {
    if (resultImage) {
      handleImageLoad(resultImage);
      clearMask();
      setViewMode("result");
    }
  };

  const activeModel = models.find((m) => m.id === selectedModelId);

  return (
    <div className="flex-1 flex flex-col h-full bg-[#090C12] text-[#E8ECF4] overflow-hidden select-none">
      {/* Top Header Bar */}
      <div className="h-12 px-6 bg-[#0D1117] border-b border-[#21262D] flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-3">
          <Paintbrush className="w-5 h-5 text-rose-400" />
          <span className="font-bold text-sm tracking-wide bg-gradient-to-r from-rose-400 via-purple-400 to-indigo-300 bg-clip-text text-transparent">
            INPAINTING LAB
          </span>
          <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/50 flex items-center space-x-1">
            <Shield className="w-3 h-3 text-rose-400" />
            <span>Face-Lock & Attire Inpaint</span>
          </span>
        </div>

        {/* Checkpoint Status Indicator */}
        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="text-[#8B949E] text-[11px]">Checkpoint:</span>
          <span className="text-white font-semibold px-2 py-0.5 rounded bg-[#161B22] border border-[#30363D]">
            {activeModel?.name || selectedModelId || "HassakuXL"}
          </span>
        </div>
      </div>

      {/* Error banner */}
      {errorMessage && (
        <div className="px-6 py-2 bg-red-950/80 border-b border-red-800/80 text-red-200 text-xs flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-xs text-red-400 hover:text-white"
          >
            ✕
          </button>
        </div>
      )}

      {/* Main Studio Body */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Left Control Column (540px) */}
        <div className="w-[540px] border-r border-[#21262D] bg-[#0D1117] flex flex-col h-full overflow-y-auto custom-scrollbar p-4 space-y-3.5 shrink-0">
          {/* 1. Model Checkpoint Selector */}
          <div className="space-y-1.5 p-3 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-bold text-blue-300 uppercase tracking-wider">
                <Brain className="w-4 h-4 text-blue-400" />
                <span>Model Checkpoint</span>
              </div>
            </div>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-blue-500 font-medium cursor-pointer"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} [{m.architecture?.toUpperCase() || "MODEL"}]
                </option>
              ))}
            </select>
          </div>

          {/* 2. Inpainting Strategy & Mask Mode */}
          <div className="p-3.5 bg-gradient-to-br from-[#1c1322] via-[#161B22] to-[#141824] rounded-xl border border-rose-900/50 shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-1.5 text-rose-300 text-xs font-bold uppercase tracking-wider">
                <Shield className="w-3.5 h-3.5 text-rose-400" />
                <span>Inpainting Strategy</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => {
                  setMaskMode("protect_painted");
                  setDenoisingStrength(0.60);
                }}
                className={`p-2.5 rounded-lg border text-left transition ${
                  maskMode === "protect_painted"
                    ? "bg-rose-950/70 border-rose-500 text-white shadow-sm"
                    : "bg-[#0D1117] border-[#30363D] text-[#8B949E] hover:text-white"
                }`}
              >
                <div className="text-xs font-bold flex items-center space-x-1.5">
                  <span>🔒 Protect Face</span>
                </div>
                <div className="text-[10px] mt-1 opacity-80 leading-snug">
                  Paint face in red to lock it 100%. Generates new attire & scene around it.
                </div>
              </button>

              <button
                type="button"
                onClick={() => {
                  setMaskMode("inpaint_painted");
                  setDenoisingStrength(0.85);
                }}
                className={`p-2.5 rounded-lg border text-left transition ${
                  maskMode === "inpaint_painted"
                    ? "bg-purple-950/70 border-purple-500 text-white shadow-sm"
                    : "bg-[#0D1117] border-[#30363D] text-[#8B949E] hover:text-white"
                }`}
              >
                <div className="text-xs font-bold flex items-center space-x-1.5">
                  <span>🎨 Inpaint Painted Area</span>
                </div>
                <div className="text-[10px] mt-1 opacity-80 leading-snug">
                  Paint clothes/background in red to regenerate only the painted zone.
                </div>
              </button>
            </div>

            {/* AI Auto Face Detection Button */}
            {maskMode === "protect_painted" && (
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/40">
                <div className="text-[11px] text-rose-200">
                  <span className="font-semibold">AI Neural Detection:</span> Auto-lock exact face
                </div>
                <button
                  type="button"
                  onClick={handleAutoDetectFace}
                  disabled={isDetectingFace || !baseImageSrc}
                  className="px-2.5 py-1 text-xs font-semibold rounded bg-rose-600 hover:bg-rose-500 disabled:opacity-50 text-white flex items-center space-x-1.5 transition shrink-0"
                >
                  {isDetectingFace ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Detecting...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-3.5 h-3.5" />
                      <span>Auto-Detect Face</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Denoising Strength & Mask Feathering */}
            <div className="grid grid-cols-2 gap-3 pt-1 border-t border-[#30363D]/60">
              <div>
                <div className="flex justify-between text-[11px] font-semibold text-[#8B949E] uppercase mb-1">
                  <span>Denoising Strength</span>
                  <span className="font-mono text-rose-300">{denoisingStrength.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="1.0"
                  step="0.05"
                  value={denoisingStrength}
                  onChange={(e) => setDenoisingStrength(parseFloat(e.target.value))}
                  className="w-full accent-rose-500 cursor-pointer"
                />
                <div className="text-[9px] text-[#8B949E] mt-0.5">
                  {maskMode === "protect_painted"
                    ? denoisingStrength > 0.70
                      ? "⚠️ >0.70 causes anatomical distortions. 0.55-0.65 optimal."
                      : "0.55-0.65 preserves head/neck while regenerating attire"
                    : denoisingStrength >= 0.8
                    ? "Full redraw of painted zone"
                    : "Partial restyle of painted zone"}
                </div>
              </div>

              <div>
                <div className="flex justify-between text-[11px] font-semibold text-[#8B949E] uppercase mb-1">
                  <span>Edge Feather (Blur)</span>
                  <span className="font-mono text-purple-300">{maskBlur}px</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="20"
                  step="1"
                  value={maskBlur}
                  onChange={(e) => setMaskBlur(parseInt(e.target.value, 10))}
                  className="w-full accent-purple-500 cursor-pointer"
                />
                <div className="text-[9px] text-[#8B949E] mt-0.5">
                  Seamlessly blends edges into skin/hair
                </div>
              </div>
            </div>
          </div>

          {/* 3. Enlarged Positive Prompt Panel */}
          <div className="space-y-1.5 p-3.5 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-purple-300 uppercase tracking-wider flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                <span>Attire & Ambiance Prompt</span>
              </span>
              <div className="flex items-center space-x-1 text-[10px] font-mono text-[#8B949E]">
                <span>{prompt.length} chars</span>
              </div>
            </div>

            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={7}
              placeholder="Describe the new attire, fabric, embroidery, jewelry, lighting, room ambiance..."
              className="w-full min-h-[150px] bg-[#0D1117] border border-[#30363D] rounded-lg p-3 text-xs text-white font-mono leading-relaxed focus:outline-none focus:border-purple-500 resize-y custom-scrollbar"
            />

            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center space-x-1.5">
                <button
                  type="button"
                  onClick={() =>
                    setPrompt(
                      "1girl, wearing royal midnight blue translucent saree, gold zardozi embroidered borders, raised metallic threadings, lace corset blouse, gold chandelier lighting, chiaroscuro, subsurface scattering on porcelain skin, intricate jewelry, masterpiece, 8k"
                    )
                  }
                  className="px-2.5 py-1 rounded bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] text-[#35D6C5] text-[11px] font-semibold flex items-center space-x-1 transition"
                >
                  <Sparkles className="w-3 h-3" />
                  <span>Saree & Jewelry Preset</span>
                </button>
                <button
                  type="button"
                  onClick={() => setPrompt("")}
                  className="p-1 rounded bg-[#0D1117] hover:bg-red-950/40 border border-[#30363D] text-[#8993A7] hover:text-red-400 text-[11px] transition"
                  title="Clear prompt"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>

              <button
                type="button"
                onClick={() => {
                  navigator.clipboard.writeText(prompt);
                  setCopiedPrompt(true);
                  setTimeout(() => setCopiedPrompt(false), 1500);
                }}
                className="text-[11px] font-semibold text-gray-400 hover:text-white flex items-center space-x-1"
              >
                {copiedPrompt ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                <span>{copiedPrompt ? "Copied" : "Copy"}</span>
              </button>
            </div>
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
                onClick={() =>
                  setNegativePrompt(
                    "blurry, low quality, distorted, bad anatomy, deformed, artifacts, ugly, worst quality, extra limbs"
                  )
                }
                className="text-[10px] text-[#8B949E] hover:text-white"
              >
                Default Preset
              </button>
            </div>

            <textarea
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              rows={3}
              placeholder="Elements to suppress: blurry, worst quality, deformed limbs..."
              className="w-full min-h-[70px] bg-[#0D1117] border border-[#30363D] rounded-lg p-2.5 text-xs text-[#8B949E] font-mono leading-relaxed focus:outline-none focus:border-rose-500 resize-y custom-scrollbar"
            />
          </div>

          {/* 5. LoRA Adapters Rack with Categories */}
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
                  No LoRAs attached. Attach character, lighting, or texture LoRAs.
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

          {/* 6. Advanced Sampler & Generation Settings */}
          <div className="p-3 bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm space-y-3">
            <div className="flex items-center space-x-1.5 text-[#C9D1D9] text-xs font-bold uppercase tracking-wider">
              <Sliders className="w-3.5 h-3.5 text-rose-400" />
              <span>Inpainting Parameters</span>
            </div>

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

            {/* Steps & CFG */}
            <div className="grid grid-cols-2 gap-3 pt-1">
              <div>
                <div className="flex justify-between text-[11px] text-[#8B949E] font-semibold uppercase mb-1">
                  <span>Steps ({steps})</span>
                </div>
                <input
                  type="range"
                  min="15"
                  max="50"
                  step="1"
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value, 10))}
                  className="w-full accent-rose-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-[11px] text-[#8B949E] font-semibold uppercase mb-1">
                  <span>CFG Scale ({cfgScale.toFixed(1)})</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="14"
                  step="0.5"
                  value={cfgScale}
                  onChange={(e) => setCfgScale(parseFloat(e.target.value))}
                  className="w-full accent-rose-500 cursor-pointer"
                />
              </div>
            </div>

            {/* Seed */}
            <div>
              <div className="flex items-center justify-between text-[11px] text-[#8B949E] font-semibold uppercase mb-1">
                <span>Seed</span>
                <button
                  type="button"
                  onClick={randomizeSeed}
                  className="text-[10px] text-purple-400 hover:text-white flex items-center space-x-1"
                >
                  <Dices className="w-3 h-3" />
                  <span>Random</span>
                </button>
              </div>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
                className="w-full bg-[#0D1117] border border-[#30363D] rounded px-2.5 py-1.5 text-xs text-white font-mono"
              />
            </div>
          </div>

          {/* Primary Inpaint Button */}
          <div className="pt-2 sticky bottom-0 bg-[#0D1117] pb-1">
            <button
              type="button"
              disabled={isGenerating || !baseImageSrc}
              onClick={executeInpaint}
              className={`w-full py-3.5 rounded-xl font-bold text-sm tracking-wide shadow-lg flex items-center justify-center space-x-2 transition ${
                isGenerating || !baseImageSrc
                  ? "bg-rose-950/40 text-rose-400/50 border border-rose-900/30 cursor-not-allowed"
                  : "bg-gradient-to-r from-rose-600 via-purple-600 to-indigo-600 hover:from-rose-500 hover:to-indigo-500 text-white border border-rose-400/30 hover:shadow-rose-500/20"
              }`}
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-white" />
                  <span>Inpainting ({progress.step}/{progress.total})...</span>
                </>
              ) : (
                <>
                  <Paintbrush className="w-4 h-4 text-white" />
                  <span>Execute Inpaint Generation</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Canvas & Viewport Area */}
        <div className="flex-1 flex flex-col bg-[#090C12] overflow-hidden p-4 space-y-3">
          {/* Canvas Toolbar */}
          <div className="h-11 px-4 bg-[#161B22] rounded-xl border border-[#30363D] flex items-center justify-between shrink-0 shadow-sm">
            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={() => setTool("brush")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition ${
                  tool === "brush"
                    ? "bg-rose-600 text-white shadow-sm"
                    : "bg-[#0D1117] text-[#8B949E] hover:text-white border border-[#30363D]"
                }`}
              >
                <Paintbrush className="w-3.5 h-3.5" />
                <span>Brush</span>
              </button>

              <button
                type="button"
                onClick={() => setTool("eraser")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition ${
                  tool === "eraser"
                    ? "bg-rose-600 text-white shadow-sm"
                    : "bg-[#0D1117] text-[#8B949E] hover:text-white border border-[#30363D]"
                }`}
              >
                <Eraser className="w-3.5 h-3.5" />
                <span>Eraser</span>
              </button>

              {/* Brush Size Slider */}
              <div className="flex items-center space-x-2 pl-3 border-l border-[#30363D]">
                <span className="text-[11px] font-semibold text-[#8B949E]">Size:</span>
                <input
                  type="range"
                  min="8"
                  max="120"
                  step="2"
                  value={brushSize}
                  onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
                  className="w-24 accent-rose-500 cursor-pointer"
                />
                <span className="text-xs font-mono text-white w-6">{brushSize}px</span>
              </div>

              <button
                type="button"
                onClick={invertMask}
                title="Invert painted mask"
                className="px-2.5 py-1.5 rounded-lg bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] text-[#C9D1D9] text-xs font-semibold flex items-center space-x-1 transition"
              >
                <ArrowRightLeft className="w-3.5 h-3.5 text-purple-400" />
                <span>Invert Mask</span>
              </button>

              <button
                type="button"
                onClick={clearMask}
                title="Clear all painted mask"
                className="px-2.5 py-1.5 rounded-lg bg-[#0D1117] hover:bg-red-950/40 border border-[#30363D] hover:border-red-500/50 text-[#8B949E] hover:text-red-400 text-xs font-semibold flex items-center space-x-1 transition"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear Mask</span>
              </button>
            </div>

            {/* View Mode & Upload */}
            <div className="flex items-center space-x-2">
              {resultImage && (
                <>
                  <div className="flex rounded-lg bg-[#0D1117] p-0.5 border border-[#30363D]">
                    <button
                      type="button"
                      onClick={() => setViewMode("result")}
                      className={`px-2.5 py-1 rounded text-xs font-bold transition ${
                        viewMode === "result"
                          ? "bg-rose-600 text-white"
                          : "text-[#8B949E] hover:text-white"
                      }`}
                    >
                      Result
                    </button>
                    <button
                      type="button"
                      onClick={() => setViewMode("original")}
                      className={`px-2.5 py-1 rounded text-xs font-bold transition ${
                        viewMode === "original"
                          ? "bg-rose-600 text-white"
                          : "text-[#8B949E] hover:text-white"
                      }`}
                    >
                      Original
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={useResultAsInput}
                    className="px-2.5 py-1.5 rounded-lg bg-purple-950/60 hover:bg-purple-900 border border-purple-800/60 text-purple-200 text-xs font-bold flex items-center space-x-1 transition"
                    title="Send generated result back into canvas for another pass"
                  >
                    <RotateCcw className="w-3.5 h-3.5 text-purple-300" />
                    <span>Inpaint Again</span>
                  </button>

                  <a
                    href={resultImage}
                    download={`inpainted_${Date.now()}.png`}
                    className="px-2.5 py-1.5 rounded-lg bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] text-white text-xs font-bold flex items-center space-x-1 transition"
                  >
                    <Download className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Save</span>
                  </a>
                </>
              )}

              <label className="px-3 py-1.5 rounded-lg bg-[#0D1117] hover:bg-[#21262D] border border-[#30363D] hover:border-purple-500/50 text-[#C9D1D9] text-xs font-semibold flex items-center space-x-1.5 cursor-pointer transition">
                <Upload className="w-3.5 h-3.5 text-cyan-400" />
                <span>Upload Image</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>
          </div>

          {/* Canvas Viewport */}
          <div
            ref={containerRef}
            className="flex-1 rounded-2xl bg-[#0B0E14] border border-[#21262D] flex items-center justify-center overflow-hidden relative p-2 shadow-inner"
          >
            {!baseImageSrc ? (
              <label className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-[#30363D] hover:border-rose-500/60 rounded-2xl cursor-pointer transition space-y-3 max-w-md text-center bg-[#161B22]/30">
                <div className="w-14 h-14 rounded-full bg-rose-950/40 border border-rose-800/50 flex items-center justify-center text-rose-400">
                  <Upload className="w-7 h-7" />
                </div>
                <div>
                  <div className="font-bold text-sm text-white">
                    Upload Character Image to Inpaint
                  </div>
                  <div className="text-xs text-[#8B949E] mt-1">
                    Select an image from your device or drop it here. You can paint the face to protect it, or paint the clothes to redesign them.
                  </div>
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            ) : (
              <div
                className="relative max-h-full max-w-full flex items-center justify-center"
                style={{
                  aspectRatio: `${imageDimensions.width} / ${imageDimensions.height}`,
                }}
              >
                {/* Result Image View Mode */}
                {resultImage && viewMode === "result" ? (
                  <img
                    src={resultImage}
                    alt="Inpainted Result"
                    className="max-h-[calc(100vh-180px)] max-w-full object-contain rounded-lg shadow-2xl border border-[#30363D]"
                  />
                ) : (
                  <>
                    {/* Background Original Image Canvas */}
                    <canvas
                      ref={imageCanvasRef}
                      className="max-h-[calc(100vh-180px)] max-w-full object-contain rounded-lg shadow-2xl"
                    />

                    {/* Interactive Mask Overlay Canvas */}
                    <canvas
                      ref={maskCanvasRef}
                      onMouseDown={handleMouseDown}
                      onMouseMove={handleMouseMove}
                      onMouseUp={handleMouseUp}
                      onMouseLeave={handleMouseUp}
                      style={{ cursor: tool === "brush" ? "crosshair" : "cell" }}
                      className="absolute inset-0 w-full h-full object-contain rounded-lg touch-none"
                    />
                  </>
                )}

                {/* Progress Overlay */}
                {isGenerating && (
                  <div className="absolute inset-0 bg-black/75 backdrop-blur-xs rounded-lg flex flex-col items-center justify-center space-y-3 z-30 p-6">
                    <Loader2 className="w-10 h-10 animate-spin text-rose-400" />
                    <div className="text-sm font-bold text-white">
                      Inpainting in Progress...
                    </div>
                    <div className="text-xs font-mono text-[#8B949E]">
                      Step {progress.step} of {progress.total}
                    </div>
                    {/* Progress Bar */}
                    <div className="w-64 h-2 bg-[#161B22] rounded-full overflow-hidden border border-[#30363D]">
                      <div
                        className="h-full bg-gradient-to-r from-rose-500 via-purple-500 to-indigo-500 transition-all duration-150"
                        style={{
                          width: `${progress.total ? (progress.step / progress.total) * 100 : 0}%`,
                        }}
                      />
                    </div>
                    <div className="text-[11px] text-gray-400 font-mono text-center max-w-sm">
                      {progress.message || "Synthesizing new attire & lighting..."}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* LoRA Selector Modal */}
      <LoraSelectorModal
        isOpen={isLoraModalOpen}
        onClose={() => setIsLoraModalOpen(false)}
        availableLoras={availableLoras}
        activeLoraPaths={new Set(activeLoras.map((l) => l.path))}
        onSelectLora={addLora}
      />
    </div>
  );
};
