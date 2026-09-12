import React from "react";
import { Zap, Cpu, HardDrive, Activity } from "lucide-react";
import { useGeneration } from "../../state/generationContext";

export const TopBar: React.FC = () => {
  const { systemStatus, selectedModelId, models } = useGeneration();
  const activeModel = models.find((m) => m.id === selectedModelId);
  const activeModelDisplay = activeModel?.name || "Hassaku XL";

  const vramPct =
    systemStatus.vramTotalGb > 0
      ? Math.min(Math.round((systemStatus.vramUsedGb / systemStatus.vramTotalGb) * 100), 100)
      : 0;

  const ramPct =
    systemStatus.ramTotalGb > 0
      ? Math.min(Math.round((systemStatus.ramUsedGb / systemStatus.ramTotalGb) * 100), 100)
      : 0;

  const getVramColor = (pct: number) => {
    if (pct >= 85) return "text-rose-400 bg-rose-500";
    if (pct >= 70) return "text-amber-400 bg-amber-500";
    return "text-cyan-400 bg-cyan-500";
  };

  const vramColor = getVramColor(vramPct);

  return (
    <header className="h-14 px-4 bg-[#0D1117] border-b border-[#21262D] flex items-center justify-between shrink-0 select-none shadow-sm">
      {/* Left Branding */}
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded-lg bg-[#161B22] border border-[#30363D] flex items-center justify-center text-[#8A2BE2] shadow-inner">
          <Zap className="w-5 h-5 fill-current" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-bold text-sm tracking-wide bg-gradient-to-r from-purple-400 to-indigo-300 bg-clip-text text-transparent">
              ImageStudio
            </span>
            <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded bg-purple-950/60 border border-purple-800/50 text-purple-300">
              {activeModelDisplay}
            </span>
          </div>
          <p className="text-[10px] text-[#8B949E] font-medium leading-none mt-0.5">
            AI Creative Workstation
          </p>
        </div>
      </div>

      {/* Right Telemetry Indicators */}
      <div className="flex items-center space-x-2">
        <div className="bg-[#161B22] border border-[#30363D] rounded-lg px-3 py-1.5 flex items-center space-x-4 text-xs font-mono">
          {/* GPU Chip Status & Core Load */}
          <div className="flex items-center space-x-1.5">
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-white font-semibold tracking-tight">
              {systemStatus.gpuName || "RTX GPU"}
            </span>
            {typeof systemStatus.gpuUtilPercent === "number" && (
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-950/60 border border-emerald-800/40 text-emerald-400 font-bold">
                {systemStatus.gpuUtilPercent}%
              </span>
            )}
          </div>

          {/* VRAM Meter with Visual Fill Bar */}
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1 text-[#8B949E]">
              <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
              <span>
                VRAM:{" "}
                <strong className={vramColor.split(" ")[0]}>
                  {systemStatus.vramUsedGb.toFixed(1)}
                </strong>{" "}
                / {systemStatus.vramTotalGb.toFixed(1)} GB
              </span>
            </div>
            {/* Visual Gauge */}
            <div className="w-16 h-1.5 bg-[#090C12] rounded-full overflow-hidden border border-[#30363D]">
              <div
                className={`h-full rounded-full transition-all duration-500 ${vramColor.split(" ")[1]}`}
                style={{ width: `${vramPct}%` }}
              />
            </div>
            <span className={`text-[10px] font-bold ${vramColor.split(" ")[0]}`}>
              {vramPct}%
            </span>
          </div>

          {/* System RAM Meter */}
          <div className="hidden sm:flex items-center space-x-2 text-[#8B949E]">
            <Cpu className="w-3.5 h-3.5 text-purple-400" />
            <span>
              RAM:{" "}
              <strong className="text-purple-300">
                {systemStatus.ramUsedGb.toFixed(1)}
              </strong>{" "}
              / {systemStatus.ramTotalGb.toFixed(1)} GB
            </span>
            <div className="w-12 h-1.5 bg-[#090C12] rounded-full overflow-hidden border border-[#30363D]">
              <div
                className="h-full bg-purple-500 rounded-full transition-all duration-500"
                style={{ width: `${ramPct}%` }}
              />
            </div>
          </div>

          {/* CPU Meter */}
          <div className="hidden md:flex items-center space-x-1.5 text-[#8B949E]">
            <Activity className="w-3.5 h-3.5 text-indigo-400" />
            <span>
              CPU: <strong className="text-indigo-300">{systemStatus.cpuPercent}%</strong>
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};

