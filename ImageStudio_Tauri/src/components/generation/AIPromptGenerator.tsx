import React, { useState, useEffect } from "react";
import {
  Sparkles,
  Settings2,
  RefreshCw,
  Copy,
  Check,
  Plus,
  ArrowRightLeft,
  Trash2,
  AlertCircle,
  Loader2,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  Server,
  Zap,
} from "lucide-react";
import { generationService } from "../../services/generationService";
import {
  PromptLLMSettings,
  PromptProfileInfo,
  PromptGenerateResult,
  TestConnectionResult,
} from "../../types";

interface AIPromptGeneratorProps {
  currentPrompt: string;
  onReplacePrompt: (newPrompt: string) => void;
  onAppendPrompt: (newPrompt: string) => void;
  onSetNegativePrompt?: (negativePrompt: string) => void;
}

const STYLE_OPTIONS = [
  { id: "General", label: "General / Neutral" },
  { id: "Anime / Illustration", label: "Anime / Illustration" },
  { id: "Cinematic Photograph", label: "Cinematic Photograph" },
  { id: "Fantasy Art", label: "Fantasy Art" },
  { id: "Cyberpunk", label: "Cyberpunk Aesthetic" },
  { id: "Watercolor", label: "Watercolor Painting" },
];

const INSPIRATION_PROMPTS = [
  "A realistic portrait of a young woman with black hair standing on a rainy Tokyo street at night, illuminated by neon signs.",
  "Blonde knight in silver armor holding a glowing magical sword in an enchanted moonlit forest.",
  "Cyberpunk hacker girl working on holographic displays in a high-tech subterranean laboratory.",
  "Ethereal majestic dragon soaring through aurora borealis night sky over snow-capped mountains.",
  "Anime girl in sailor school uniform resting under blooming cherry blossom trees on a breezy sunny afternoon.",
];

export const AIPromptGenerator: React.FC<AIPromptGeneratorProps> = ({
  currentPrompt,
  onReplacePrompt,
  onAppendPrompt,
  onSetNegativePrompt,
}) => {
  // Expansion State
  const [isExpanded, setIsExpanded] = useState<boolean>(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);

  // Natural Language Input & Controls
  const [description, setDescription] = useState<string>("");
  const [selectedProfile, setSelectedProfile] = useState<string>("joycaption");
  const [selectedStyle, setSelectedStyle] = useState<string>("General");

  // Profiles & Settings State
  const [profiles, setProfiles] = useState<PromptProfileInfo[]>([]);
  const [settings, setSettings] = useState<PromptLLMSettings>({
    enabled: true,
    provider: "openai_compatible",
    base_url: "http://127.0.0.1:11434/v1",
    api_key: "",
    model: "joycaption",
    temperature: 0.5,
    max_tokens: 512,
    timeout: 60,
    default_profile: "joycaption",
  });

  // Generation Execution State
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generatedResult, setGeneratedResult] = useState<PromptGenerateResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copiedTags, setCopiedTags] = useState<boolean>(false);

  // Connection Test State
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  const [connectionStatus, setConnectionStatus] = useState<TestConnectionResult | null>(null);
  const [availableServerModels, setAvailableServerModels] = useState<string[]>([]);

  // Load profiles and settings on mount
  useEffect(() => {
    const init = async () => {
      try {
        const [profList, sett] = await Promise.all([
          generationService.getPromptProfiles(),
          generationService.getPromptLLMSettings(),
        ]);
        setProfiles(profList);
        setSettings(sett);
        if (sett.default_profile) {
          setSelectedProfile(sett.default_profile);
        }
      } catch {
        // Ignored
      }
    };
    init();
  }, []);

  const handleTestConnection = async () => {
    setIsTestingConnection(true);
    setConnectionStatus(null);
    try {
      const res = await generationService.testPromptLLMConnection(settings);
      setConnectionStatus(res);
      if (res.success && res.models && res.models.length > 0) {
        setAvailableServerModels(res.models);
        if (!res.models.includes(settings.model)) {
          setSettings((prev) => ({ ...prev, model: res.models![0] }));
        }
      }
    } catch (e: any) {
      setConnectionStatus({
        success: false,
        base_url: settings.base_url,
        error: e.message || "Connection failed",
      });
    } finally {
      setIsTestingConnection(false);
    }
  };

  const handleSaveSettings = async () => {
    try {
      const saved = await generationService.savePromptLLMSettings(settings);
      setSettings(saved);
      setIsSettingsOpen(false);
    } catch (e) {
      console.error("Failed to save settings:", e);
    }
  };

  const handleGenerate = async () => {
    if (!description.trim()) {
      setErrorMessage("Please enter a description of the image you want to generate.");
      return;
    }

    setIsGenerating(true);
    setErrorMessage(null);

    try {
      const res = await generationService.generatePromptTags({
        text: description,
        profile: selectedProfile,
        style: selectedStyle,
        existing_prompt: currentPrompt,
        mode: "append",
        settings,
      });

      if (res.success) {
        setGeneratedResult(res);
      } else {
        setErrorMessage(res.error || "Failed to generate visual tags from description.");
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Error generating visual tags.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleReplace = () => {
    if (!generatedResult) return;
    onReplacePrompt(generatedResult.prompt);
  };

  const handleAppend = () => {
    if (!generatedResult) return;
    onAppendPrompt(generatedResult.prompt);
  };

  const handleApplyNegative = () => {
    if (!generatedResult?.negative_prompt || !onSetNegativePrompt) return;
    onSetNegativePrompt(generatedResult.negative_prompt);
  };

  const handleCopyTags = () => {
    if (!generatedResult) return;
    navigator.clipboard.writeText(generatedResult.prompt);
    setCopiedTags(true);
    setTimeout(() => setCopiedTags(false), 2000);
  };

  const handleClear = () => {
    setDescription("");
    setGeneratedResult(null);
    setErrorMessage(null);
  };

  return (
    <div className="bg-[#161B22] rounded-xl border border-[#30363D] shadow-sm overflow-hidden transition-all">
      {/* Top Header / Toggle Bar */}
      <div className="flex items-center justify-between px-3.5 py-2.5 bg-[#0F131B] border-b border-[#252C3A]">
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center space-x-2 text-xs font-bold text-[#35D6C5] hover:text-[#52e7d7] uppercase tracking-wider transition-colors"
        >
          <Sparkles className="w-4 h-4 text-[#35D6C5]" />
          <span>AI Prompt Generator</span>
          <span className="text-[10px] lowercase tracking-normal font-mono px-1.5 py-0.5 rounded bg-[#1C2333] text-[#8993A7] border border-[#252C3A]">
            Natural Language → JoyCaption Prompts
          </span>
          {isExpanded ? (
            <ChevronUp className="w-3.5 h-3.5 text-[#8993A7]" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-[#8993A7]" />
          )}
        </button>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={() => setIsSettingsOpen(!isSettingsOpen)}
            title="Configure LLM Provider (OpenAI Compatible, Ollama, LM Studio)"
            className={`p-1.5 rounded-lg border text-xs flex items-center space-x-1 transition-colors ${
              isSettingsOpen
                ? "bg-[#1C2333] text-[#7C6CFF] border-[#7C6CFF]"
                : "bg-[#0D1117] text-[#8993A7] hover:text-white border-[#30363D] hover:bg-[#21262D]"
            }`}
          >
            <Settings2 className="w-3.5 h-3.5" />
            <span className="text-[10px] font-medium hidden sm:inline">LLM Settings</span>
          </button>
        </div>
      </div>

      {/* Main Collapsible Body */}
      {isExpanded && (
        <div className="p-3.5 space-y-3">
          {/* Natural Language Textarea */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <label className="font-semibold text-slate-300">Natural Language Description</label>
              <button
                type="button"
                onClick={() => {
                  const randomPick =
                    INSPIRATION_PROMPTS[Math.floor(Math.random() * INSPIRATION_PROMPTS.length)];
                  setDescription(randomPick);
                }}
                className="text-[11px] text-[#35D6C5] hover:underline flex items-center space-x-1"
              >
                <Zap className="w-3 h-3" />
                <span>Example Idea</span>
              </button>
            </div>

            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              placeholder="Describe the image you want in plain English (e.g. A realistic portrait of a young woman with black hair standing on a rainy Tokyo street at night, illuminated by neon signs)..."
              className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg p-2.5 text-xs text-white placeholder-slate-500 font-sans leading-relaxed focus:outline-none focus:border-[#35D6C5] resize-y custom-scrollbar transition-colors"
            />
          </div>

          {/* Model Profile & Style Selectors */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-[11px] font-semibold text-slate-400">Model Profile</label>
              <select
                value={selectedProfile}
                onChange={(e) => setSelectedProfile(e.target.value)}
                className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-[#35D6C5] cursor-pointer"
              >
                {profiles.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label className="text-[11px] font-semibold text-slate-400">Visual Style</label>
              <select
                value={selectedStyle}
                onChange={(e) => setSelectedStyle(e.target.value)}
                className="w-full bg-[#0D1117] border border-[#30363D] rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-[#35D6C5] cursor-pointer"
              >
                {STYLE_OPTIONS.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Action Row */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={handleGenerate}
                disabled={isGenerating || !description.trim()}
                className={`
                  px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 shadow-sm transition-all
                  ${
                    isGenerating || !description.trim()
                      ? "bg-[#1C2333] text-[#555E70] border border-[#252C3A] cursor-not-allowed"
                      : "bg-gradient-to-r from-[#35D6C5] to-[#25B8A8] hover:from-[#43e6d5] hover:to-[#32c9b9] text-black font-bold cursor-pointer"
                  }
                `}
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Generating Tags...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>{generatedResult ? "Regenerate Tags" : "Generate Tags"}</span>
                  </>
                )}
              </button>

              {description && (
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-2 py-1.5 rounded-lg bg-[#0D1117] hover:bg-red-950/30 border border-[#30363D] hover:border-red-500/40 text-[#8993A7] hover:text-red-400 text-xs transition-colors"
                  title="Clear description and results"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            <div className="text-[11px] font-mono text-[#8993A7] truncate max-w-[200px]">
              Target: <span className="text-slate-300">{settings.model}</span>
            </div>
          </div>

          {/* Error Banner */}
          {errorMessage && (
            <div className="p-3 rounded-lg bg-red-950/40 border border-red-800/60 text-red-300 text-xs flex items-start space-x-2.5">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <p className="font-semibold">LLM Generation Failed</p>
                <p className="text-[11px] text-red-300/90 mt-0.5">{errorMessage}</p>
                <p className="text-[10px] text-red-400/80 mt-1 font-mono">
                  Make sure your local server (Ollama, LM Studio, llama.cpp, VRAM-Zero) is active at{" "}
                  {settings.base_url}
                </p>
              </div>
            </div>
          )}

          {/* Generated Result Showcase */}
          {generatedResult && (
            <div className="p-3 bg-[#0D1117] rounded-lg border border-[#30363D] space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <span className="font-bold text-emerald-400 flex items-center space-x-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>SDXL Visual Tags</span>
                  </span>
                  <span className="text-[10px] px-1.5 py-0.2 rounded bg-[#1C2333] text-[#8993A7] font-mono border border-[#252C3A]">
                    {generatedResult.parsed_tags.length} tags
                  </span>
                </div>

                <button
                  type="button"
                  onClick={handleCopyTags}
                  className="text-[11px] text-[#8993A7] hover:text-white flex items-center space-x-1 transition-colors"
                >
                  {copiedTags ? (
                    <Check className="w-3 h-3 text-emerald-400" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                  <span>{copiedTags ? "Copied" : "Copy"}</span>
                </button>
              </div>

              {/* Tag Content Preview */}
              <div className="p-2.5 bg-[#161B22] rounded border border-[#252C3A] text-xs text-slate-200 font-mono leading-relaxed max-h-[140px] overflow-y-auto custom-scrollbar select-text">
                {generatedResult.prompt}
              </div>

              {/* Injection Action Buttons */}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <button
                  type="button"
                  onClick={handleReplace}
                  className="px-3 py-1.5 rounded-lg bg-[#7C6CFF] hover:bg-[#8e80ff] text-white text-xs font-semibold flex items-center space-x-1.5 shadow-sm transition-colors cursor-pointer"
                  title="Replace existing positive prompt with these generated tags"
                >
                  <ArrowRightLeft className="w-3.5 h-3.5" />
                  <span>Replace Existing Prompt</span>
                </button>

                <button
                  type="button"
                  onClick={handleAppend}
                  className="px-3 py-1.5 rounded-lg bg-[#1F242C] hover:bg-[#2A313C] text-[#35D6C5] border border-[#35D6C5]/40 text-xs font-semibold flex items-center space-x-1.5 transition-colors cursor-pointer"
                  title="Append these tags to your existing positive prompt without duplicates"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Append to Prompt</span>
                </button>

                {generatedResult.negative_prompt && onSetNegativePrompt && (
                  <button
                    type="button"
                    onClick={handleApplyNegative}
                    className="px-2.5 py-1.5 rounded-lg bg-[#1F242C] hover:bg-[#2A313C] text-rose-300 border border-rose-800/40 text-[11px] font-medium transition-colors cursor-pointer"
                    title="Apply profile recommended negative prompt"
                  >
                    Apply Negative Preset
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* LLM Connection Settings Modal / Drawer */}
      {isSettingsOpen && (
        <div className="p-4 bg-[#0B0F17] border-t border-[#252C3A] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-xs font-bold text-slate-200 uppercase tracking-wide">
              <Server className="w-4 h-4 text-[#7C6CFF]" />
              <span>LLM Provider Configuration</span>
            </div>
            <button
              type="button"
              onClick={() => setIsSettingsOpen(false)}
              className="text-xs text-[#8993A7] hover:text-white"
            >
              Close
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            {/* Provider Type */}
            <div className="space-y-1">
              <label className="text-slate-400 font-medium">Provider Architecture</label>
              <select
                value={settings.provider}
                onChange={(e) => setSettings({ ...settings, provider: e.target.value })}
                className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-[#7C6CFF]"
              >
                <option value="openai_compatible">OpenAI Compatible (Ollama / LM Studio / llama.cpp / VRAM-Zero)</option>
              </select>
            </div>

            {/* Base URL */}
            <div className="space-y-1">
              <label className="text-slate-400 font-medium">Base URL</label>
              <input
                type="text"
                value={settings.base_url}
                onChange={(e) => setSettings({ ...settings, base_url: e.target.value })}
                placeholder="http://127.0.0.1:11434/v1"
                className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-2.5 py-1.5 text-white font-mono focus:outline-none focus:border-[#7C6CFF]"
              />
            </div>

            {/* Model Name */}
            <div className="space-y-1">
              <label className="text-slate-400 font-medium flex items-center justify-between">
                <span>Model Name</span>
                {availableServerModels.length > 0 && (
                  <span className="text-[10px] text-emerald-400 font-mono">
                    {availableServerModels.length} detected
                  </span>
                )}
              </label>
              {availableServerModels.length > 0 ? (
                <select
                  value={settings.model}
                  onChange={(e) => setSettings({ ...settings, model: e.target.value })}
                  className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-2.5 py-1.5 text-white font-mono focus:outline-none focus:border-[#7C6CFF]"
                >
                  {availableServerModels.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={settings.model}
                  onChange={(e) => setSettings({ ...settings, model: e.target.value })}
                  placeholder="e.g. joycaption or joycaption-alpha-two"
                  className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-2.5 py-1.5 text-white font-mono focus:outline-none focus:border-[#7C6CFF]"
                />
              )}
            </div>

            {/* API Key (Optional) */}
            <div className="space-y-1">
              <label className="text-slate-400 font-medium">API Key (Optional for local)</label>
              <input
                type="password"
                value={settings.api_key}
                onChange={(e) => setSettings({ ...settings, api_key: e.target.value })}
                placeholder="sk-... (not required for local Ollama)"
                className="w-full bg-[#161B22] border border-[#30363D] rounded-lg px-2.5 py-1.5 text-white font-mono focus:outline-none focus:border-[#7C6CFF]"
              />
            </div>

            {/* Temperature */}
            <div className="space-y-1">
              <div className="flex justify-between text-slate-400 font-medium">
                <span>Temperature</span>
                <span className="font-mono text-slate-200">{settings.temperature}</span>
              </div>
              <input
                type="range"
                min={0.0}
                max={1.0}
                step={0.05}
                value={settings.temperature}
                onChange={(e) =>
                  setSettings({ ...settings, temperature: parseFloat(e.target.value) })
                }
                className="w-full h-1.5 bg-[#1C2333] rounded-lg appearance-none cursor-pointer accent-[#7C6CFF]"
              />
            </div>

            {/* Max Tokens */}
            <div className="space-y-1">
              <div className="flex justify-between text-slate-400 font-medium">
                <span>Max Output Tokens</span>
                <span className="font-mono text-slate-200">{settings.max_tokens}</span>
              </div>
              <input
                type="range"
                min={128}
                max={1024}
                step={64}
                value={settings.max_tokens}
                onChange={(e) =>
                  setSettings({ ...settings, max_tokens: parseInt(e.target.value) })
                }
                className="w-full h-1.5 bg-[#1C2333] rounded-lg appearance-none cursor-pointer accent-[#7C6CFF]"
              />
            </div>
          </div>

          {/* Connection Test Result Banner */}
          {connectionStatus && (
            <div
              className={`p-2.5 rounded-lg border text-xs flex items-center space-x-2 ${
                connectionStatus.success
                  ? "bg-emerald-950/40 border-emerald-800/60 text-emerald-300"
                  : "bg-red-950/40 border-red-800/60 text-red-300"
              }`}
            >
              {connectionStatus.success ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              ) : (
                <XCircle className="w-4 h-4 text-red-400 shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <p className="font-semibold">
                  {connectionStatus.success ? "Connection Successful" : "Connection Failed"}
                </p>
                <p className="text-[11px] opacity-90 truncate">
                  {connectionStatus.message || connectionStatus.error}
                </p>
              </div>
            </div>
          )}

          {/* Footer Controls */}
          <div className="flex items-center justify-between pt-2 border-t border-[#252C3A]">
            <button
              type="button"
              onClick={handleTestConnection}
              disabled={isTestingConnection}
              className="px-3 py-1.5 rounded-lg bg-[#161B22] hover:bg-[#21262D] text-slate-200 border border-[#30363D] text-xs font-semibold flex items-center space-x-1.5 transition-colors"
            >
              {isTestingConnection ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <RefreshCw className="w-3.5 h-3.5 text-[#7C6CFF]" />
              )}
              <span>Test Connection</span>
            </button>

            <button
              type="button"
              onClick={handleSaveSettings}
              className="px-4 py-1.5 rounded-lg bg-[#7C6CFF] hover:bg-[#8e80ff] text-white text-xs font-bold transition-colors"
            >
              Save Configuration
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
