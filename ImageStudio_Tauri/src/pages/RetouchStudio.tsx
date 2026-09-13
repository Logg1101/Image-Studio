import React, { useState, useRef, useEffect } from "react";
import {
  Paintbrush,
  Eraser,
  Upload,
  Download,
  Scissors,
  Loader2,
  Trash2,
  Check,
  AlertCircle,
  Undo2,
  Sparkles,
  Wand2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
} from "lucide-react";
import { retouchService } from "../services/retouchService";
import { useGeneration } from "../state/generationContext";

export const RetouchStudio: React.FC = () => {
  const { selectedModelId } = useGeneration();
  const [baseImageSrc, setBaseImageSrc] = useState<string | null>(null);
  const [imageDims, setImageDims] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [brushSize, setBrushSize] = useState<number>(32);
  const [tool, setTool] = useState<"brush" | "eraser">("brush");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [fixPrompt, setFixPrompt] = useState<string>("perfect hands, detailed fingers, correct limbs");

  // Zoom & Pan state
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const isPanningRef = useRef<boolean>(false);
  const startPanRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const isDrawingRef = useRef<boolean>(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);

  // Synchronize base image to image canvas once mounted / updated
  useEffect(() => {
    if (!baseImageSrc || imageDims.width === 0) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const imgCanvas = imageCanvasRef.current;
      if (imgCanvas) {
        imgCanvas.width = imageDims.width;
        imgCanvas.height = imageDims.height;
        const ctx = imgCanvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, imageDims.width, imageDims.height);
          ctx.drawImage(img, 0, 0);
        }
      }
      const mCanvas = maskCanvasRef.current;
      if (mCanvas) {
        mCanvas.width = imageDims.width;
        mCanvas.height = imageDims.height;
      }
    };
    img.src = baseImageSrc;
  }, [baseImageSrc, imageDims]);

  const handleImageLoad = (dataUrl: string) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      setErrorMessage(null);
      setImageDims({ width: img.naturalWidth || img.width, height: img.naturalHeight || img.height });
      setBaseImageSrc(dataUrl);
      setZoom(1);
      setPan({ x: 0, y: 0 });
      clearMask();
    };
    img.src = dataUrl;
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        if (evt.target?.result) {
          const res = evt.target.result as string;
          setHistory([res]);
          handleImageLoad(res);
          setStatusMessage("Photo loaded at full resolution.");
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const getCanvasCoords = (e: React.PointerEvent<HTMLCanvasElement>): { x: number; y: number; scaleX: number } | null => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
      scaleX,
    };
  };

  const drawStroke = (from: { x: number; y: number }, to: { x: number; y: number }, scale: number) => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Scale stroke width to canvas resolution so brush visually matches screen brushSize accurately
    ctx.lineWidth = Math.max(2, brushSize * scale);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (tool === "brush") {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "rgba(239, 68, 68, 0.65)"; // Semi-transparent red overlay
    } else {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "rgba(0, 0, 0, 1)";
    }

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (!baseImageSrc) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    setZoom((prev) => Math.min(4, Math.max(0.25, Math.round(prev * factor * 100) / 100)));
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {}

    // Right-click (2), middle-click (1), or Alt+click -> Pan image
    if (e.button === 1 || e.button === 2 || e.altKey) {
      isPanningRef.current = true;
      startPanRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
      return;
    }

    if (e.button === 0) {
      const coords = getCanvasCoords(e);
      if (!coords) return;
      isDrawingRef.current = true;
      lastPosRef.current = { x: coords.x, y: coords.y };
      drawStroke(coords, coords, coords.scaleX);
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (isPanningRef.current) {
      setPan({
        x: e.clientX - startPanRef.current.x,
        y: e.clientY - startPanRef.current.y,
      });
      return;
    }

    if (!isDrawingRef.current || !lastPosRef.current) return;
    const coords = getCanvasCoords(e);
    if (!coords) return;
    drawStroke(lastPosRef.current, coords, coords.scaleX);
    lastPosRef.current = { x: coords.x, y: coords.y };
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (isPanningRef.current) {
      isPanningRef.current = false;
    }
    if (isDrawingRef.current) {
      isDrawingRef.current = false;
      lastPosRef.current = null;
    }
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {}
  };

  const clearMask = () => {
    const canvas = maskCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const exportBinaryMaskBase64 = (): string | null => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return null;

    const offscreen = document.createElement("canvas");
    offscreen.width = canvas.width;
    offscreen.height = canvas.height;
    const offCtx = offscreen.getContext("2d");
    const srcCtx = canvas.getContext("2d");
    if (!offCtx || !srcCtx) return null;

    const srcData = srcCtx.getImageData(0, 0, canvas.width, canvas.height);
    const outData = offCtx.createImageData(canvas.width, canvas.height);

    let painted = 0;
    for (let i = 0; i < srcData.data.length; i += 4) {
      const alpha = srcData.data[i + 3];
      const val = alpha > 20 ? 255 : 0;
      if (val > 0) painted++;
      outData.data[i] = val;
      outData.data[i + 1] = val;
      outData.data[i + 2] = val;
      outData.data[i + 3] = 255;
    }

    if (painted === 0) return null;
    offCtx.putImageData(outData, 0, 0);
    return offscreen.toDataURL("image/png");
  };

  const handleEraseThings = async () => {
    if (!baseImageSrc) {
      setErrorMessage("Please load an image first.");
      return;
    }
    const maskBase64 = exportBinaryMaskBase64();
    if (!maskBase64) {
      setErrorMessage("Please paint over the object or watermark you want to erase.");
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);
    setStatusMessage("Erasing object and reconstructing background via AI (LaMa)...");

    try {
      const data = await retouchService.eraseObject(baseImageSrc, maskBase64);
      handleImageLoad(data.image);
      setHistory((prev) => [...prev, data.image]);
      setStatusMessage("Object erased & background synthesized cleanly!");
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to erase object.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFixArtifacts = async () => {
    if (!baseImageSrc) {
      setErrorMessage("Please load an image first.");
      return;
    }
    const maskBase64 = exportBinaryMaskBase64();
    if (!maskBase64) {
      setErrorMessage("Please paint over the artifact, blur, or glitch you want to fix.");
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);
    setStatusMessage("Redrawing selection with anatomically correct details via AI inpaint...");

    try {
      const data = await retouchService.fixArtifacts(baseImageSrc, maskBase64, fixPrompt.trim(), selectedModelId);
      handleImageLoad(data.image);
      setHistory((prev) => [...prev, data.image]);
      setStatusMessage("Anatomy & artifacts repaired with AI details!");
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to fix artifacts.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRemoveBackground = async () => {
    if (!baseImageSrc) {
      setErrorMessage("Please load an image first.");
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);
    setStatusMessage("Removing background using AI segmentation (U-2-Net)...");

    try {
      const data = await retouchService.removeBackground(baseImageSrc);
      handleImageLoad(data.image);
      setHistory((prev) => [...prev, data.image]);
      setStatusMessage("Background removed cleanly with transparent alpha!");
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to remove background.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleUndo = () => {
    if (history.length > 1) {
      const newHistory = [...history];
      newHistory.pop();
      const prevImage = newHistory[newHistory.length - 1];
      setHistory(newHistory);
      handleImageLoad(prevImage);
      setStatusMessage("Reverted to previous step.");
    }
  };

  const handleSave = () => {
    if (!baseImageSrc) return;
    const a = document.createElement("a");
    a.href = baseImageSrc;
    const timestamp = Date.now();
    a.download = `retouched_${timestamp}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setStatusMessage("Image exported at full resolution (PNG)!");
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#090C12] text-[#E8ECF4] overflow-hidden select-none">
      {/* Header */}
      <div className="h-12 px-6 bg-[#0D1117] border-b border-[#21262D] flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-3">
          <Paintbrush className="w-5 h-5 text-emerald-400" />
          <span className="font-bold text-sm tracking-wide bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent">
            RETOUCH & CLEAN
          </span>
          <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800/50">
            Object Eraser • Cutout • Artifact Fix
          </span>
        </div>
      </div>

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Toolbar */}
        <div className="w-80 bg-[#0D1117] border-r border-[#21262D] p-4 flex flex-col space-y-4 overflow-y-auto">
          {/* File Upload */}
          <label className="flex items-center justify-center space-x-2 p-3 rounded-lg border border-dashed border-[#30363D] hover:border-emerald-500 hover:bg-[#161B22] cursor-pointer transition text-xs font-semibold text-[#8B949E] hover:text-white">
            <Upload className="w-4 h-4 text-emerald-400" />
            <span>Open Photo...</span>
            <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
          </label>

          {/* Brush Controls */}
          <div className="p-3 bg-[#161B22] border border-[#21262D] rounded-xl space-y-3">
            <div className="text-xs font-bold uppercase tracking-wider text-[#8B949E]">
              Brush Tools
            </div>
            <div className="flex space-x-2">
              <button
                type="button"
                onClick={() => setTool("brush")}
                className={`flex-1 py-1.5 px-3 rounded-lg flex items-center justify-center space-x-2 text-xs font-semibold border transition ${
                  tool === "brush"
                    ? "bg-emerald-600 border-emerald-400 text-white"
                    : "bg-[#0D1117] border-[#30363D] text-[#8B949E] hover:text-white"
                }`}
              >
                <Paintbrush className="w-3.5 h-3.5" />
                <span>Mask Brush</span>
              </button>
              <button
                type="button"
                onClick={() => setTool("eraser")}
                className={`flex-1 py-1.5 px-3 rounded-lg flex items-center justify-center space-x-2 text-xs font-semibold border transition ${
                  tool === "eraser"
                    ? "bg-rose-600 border-rose-400 text-white"
                    : "bg-[#0D1117] border-[#30363D] text-[#8B949E] hover:text-white"
                }`}
              >
                <Eraser className="w-3.5 h-3.5" />
                <span>Mask Eraser</span>
              </button>
            </div>

            <div>
              <div className="flex justify-between text-[11px] text-[#8B949E] font-medium mb-1">
                <span>Brush Size</span>
                <span className="font-mono text-white">{brushSize}px</span>
              </div>
              <input
                type="range"
                min="6"
                max="128"
                value={brushSize}
                onChange={(e) => setBrushSize(parseInt(e.target.value))}
                className="w-full accent-emerald-500 cursor-pointer"
              />
            </div>

            <button
              type="button"
              onClick={clearMask}
              className="w-full py-1.5 rounded-lg bg-[#0D1117] hover:bg-[#21262D] text-[#8B949E] hover:text-white border border-[#30363D] text-xs font-medium flex items-center justify-center space-x-1.5 transition"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear Mask</span>
            </button>
          </div>

          {/* Action Operations */}
          <div className="space-y-3 pt-2">
            {/* Button 1: Erase Things / Objects */}
            <div>
              <button
                type="button"
                onClick={handleEraseThings}
                disabled={isProcessing || !baseImageSrc}
                className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 text-white font-bold text-xs flex items-center justify-center space-x-2 shadow-lg transition cursor-pointer"
              >
                {isProcessing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
                <span>Erase Things / Objects</span>
              </button>
              <div className="text-[10px] text-[#8B949E] px-1 mt-1">
                Removes unwanted items, watermarks, or photobombers and reconstructs the background.
              </div>
            </div>

            {/* Button 2: Redraw Anatomy & Fix Artifacts */}
            <div className="p-3 bg-[#161B22] border border-purple-900/40 rounded-xl space-y-2">
              <div className="flex items-center justify-between text-xs font-bold text-purple-300">
                <span className="flex items-center space-x-1.5">
                  <Wand2 className="w-3.5 h-3.5 text-purple-400" />
                  <span>Redraw Anatomy / Fix</span>
                </span>
              </div>
              <input
                type="text"
                value={fixPrompt}
                onChange={(e) => setFixPrompt(e.target.value)}
                placeholder="e.g. perfect hands, detailed fingers, correct legs"
                className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg px-2.5 py-1.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
              <button
                type="button"
                onClick={handleFixArtifacts}
                disabled={isProcessing || !baseImageSrc}
                className="w-full py-2.5 px-4 rounded-lg bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 text-white font-bold text-xs flex items-center justify-center space-x-2 shadow-lg transition cursor-pointer"
              >
                {isProcessing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
                <span>Redraw & Fix Anatomy</span>
              </button>
              <div className="text-[10px] text-[#8B949E] px-0.5 leading-snug">
                Paints over bad hands, fingers, or deformed limbs and redraws proper anatomy via generative AI.
              </div>
            </div>

            {/* Button 3: Remove Background */}
            <div>
              <button
                type="button"
                onClick={handleRemoveBackground}
                disabled={isProcessing || !baseImageSrc}
                className="w-full py-2.5 px-4 rounded-xl bg-[#161B22] hover:bg-[#21262D] border border-cyan-500/40 text-cyan-300 hover:text-cyan-200 disabled:opacity-50 font-semibold text-xs flex items-center justify-center space-x-2 transition cursor-pointer"
              >
                <Scissors className="w-4 h-4 text-cyan-400" />
                <span>Delete Background (Cutout)</span>
              </button>
              <div className="text-[10px] text-[#8B949E] px-1 mt-1">
                One-click cutout: isolates the subject onto a clean transparent PNG.
              </div>
            </div>
          </div>

          {/* Undo & Save */}
          <div className="pt-4 border-t border-[#21262D] flex space-x-2">
            <button
              type="button"
              onClick={handleUndo}
              disabled={history.length <= 1 || isProcessing}
              className="flex-1 py-2 rounded-lg bg-[#161B22] hover:bg-[#21262D] disabled:opacity-40 text-xs font-semibold text-white border border-[#30363D] flex items-center justify-center space-x-1.5 transition"
            >
              <Undo2 className="w-3.5 h-3.5" />
              <span>Undo Step</span>
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={!baseImageSrc || isProcessing}
              className="flex-1 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-xs font-semibold text-white flex items-center justify-center space-x-1.5 transition cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Save</span>
            </button>
          </div>

          {statusMessage && (
            <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-800/40 text-emerald-300 text-xs flex items-center space-x-2">
              <Check className="w-4 h-4 shrink-0" />
              <span>{statusMessage}</span>
            </div>
          )}
          {errorMessage && (
            <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/40 text-rose-300 text-xs flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}
        </div>

        {/* Right Canvas Display */}
        <div
          className="flex-1 flex items-center justify-center p-6 bg-[#07090E] relative overflow-hidden select-none"
          onWheel={handleWheel}
        >
          {!baseImageSrc ? (
            <div className="text-center p-8 border border-dashed border-[#21262D] rounded-2xl max-w-sm">
              <Paintbrush className="w-10 h-10 text-[#30363D] mx-auto mb-3" />
              <div className="text-sm font-semibold text-[#8B949E] mb-1">
                No Photo Loaded
              </div>
              <div className="text-xs text-[#555E70] mb-4">
                Open a photo to draw on it, erase objects, isolate subjects, or fix artifacts.
              </div>
              <label className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold cursor-pointer transition inline-flex items-center space-x-2">
                <Upload className="w-3.5 h-3.5" />
                <span>Open Photo...</span>
                <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
              </label>
            </div>
          ) : (
            <div
              className="relative inline-flex items-center justify-center shadow-2xl rounded-lg overflow-hidden border border-[#21262D] select-none"
              style={{
                maxWidth: "100%",
                maxHeight: "calc(100vh - 120px)",
                aspectRatio: `${imageDims.width} / ${imageDims.height}`,
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: "center center",
                transition: isPanningRef.current ? "none" : "transform 0.08s ease-out",
                backgroundImage: `
                  linear-gradient(45deg, #1c2128 25%, transparent 25%),
                  linear-gradient(-45deg, #1c2128 25%, transparent 25%),
                  linear-gradient(45deg, transparent 75%, #1c2128 75%),
                  linear-gradient(-45deg, transparent 75%, #1c2128 75%)
                `,
                backgroundSize: "20px 20px",
                backgroundPosition: "0 0, 0 10px, 10px -10px, -10px 0px",
                backgroundColor: "#0d1117",
              }}
            >
              <canvas
                ref={imageCanvasRef}
                className="w-full h-full block"
              />
              <canvas
                ref={maskCanvasRef}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerUp}
                onContextMenu={(e) => e.preventDefault()}
                className="absolute inset-0 w-full h-full block touch-none cursor-crosshair"
              />
            </div>
          )}

          {/* Floating Zoom & Pan Controls */}
          {baseImageSrc && (
            <div className="absolute bottom-4 right-4 flex items-center space-x-1.5 bg-[#161B22]/90 backdrop-blur-md border border-[#30363D] px-2.5 py-1.5 rounded-xl shadow-2xl z-20">
              <button
                type="button"
                onClick={() => setZoom((z) => Math.max(0.25, Math.round((z - 0.25) * 100) / 100))}
                className="p-1.5 rounded-lg hover:bg-[#21262D] text-[#8B949E] hover:text-white transition cursor-pointer"
                title="Zoom Out (-)"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
                className="px-2 py-1 rounded-lg hover:bg-[#21262D] text-xs font-mono font-bold text-[#E8ECF4] transition cursor-pointer"
                title="Reset Zoom & Pan (100%)"
              >
                {Math.round(zoom * 100)}%
              </button>
              <button
                type="button"
                onClick={() => setZoom((z) => Math.min(4, Math.round((z + 0.25) * 100) / 100))}
                className="p-1.5 rounded-lg hover:bg-[#21262D] text-[#8B949E] hover:text-white transition cursor-pointer"
                title="Zoom In (+)"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <div className="w-[1px] h-4 bg-[#30363D] mx-0.5" />
              <button
                type="button"
                onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
                className="p-1.5 rounded-lg hover:bg-[#21262D] text-[#8B949E] hover:text-white transition cursor-pointer"
                title="Reset View"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RetouchStudio;
