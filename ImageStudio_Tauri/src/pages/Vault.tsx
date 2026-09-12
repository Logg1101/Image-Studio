import React, { useState, useEffect, useCallback } from 'react';
import {
  Archive,
  Search,
  RefreshCw,
  Sparkles,
  Copy,
  Check,
  Trash2,
  Sliders,
  Image as ImageIcon,
  X,
} from 'lucide-react';
import { HistoryRecord } from '../types/history';
import { fetchHistory, deleteHistoryItem, syncHistory } from '../services/historyService';
import { useGeneration } from '../state/generationContext';

interface VaultProps {
  onNavigateToText2Img: () => void;
}

export const Vault: React.FC<VaultProps> = ({ onNavigateToText2Img }) => {
  const {
    setPrompt,
    setNegativePrompt,
    setSelectedModelId,
    setSteps,
    setCfgScale,
    setSeed,
    setIsRandomSeed,
    setDimensions,
  } = useGeneration();

  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [syncing, setSyncing] = useState<boolean>(false);
  const [syncNotice, setSyncNotice] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedRecord, setSelectedRecord] = useState<HistoryRecord | null>(null);
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const PAGE_SIZE = 60;

  const loadHistory = useCallback(async (query: string = '', forceSync: boolean = false) => {
    setLoading(true);
    if (forceSync) setSyncing(true);
    try {
      if (forceSync) {
        try {
          const syncRes = await syncHistory();
          if (syncRes.synced > 0) {
            setSyncNotice(`Synced ${syncRes.synced} new artwork from disk`);
            setTimeout(() => setSyncNotice(null), 4000);
          }
        } catch (e) {
          console.error('Failed to sync history:', e);
        }
      }
      const data = await fetchHistory(PAGE_SIZE, 0, query, forceSync);
      setRecords(data.records);
      setTotal(data.total);
      if (data.records.length > 0) {
        setSelectedRecord((prev) => {
          if (prev && data.records.some((r) => r.id === prev.id)) {
            return prev;
          }
          return data.records[0];
        });
      } else {
        setSelectedRecord(null);
      }
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
      setSyncing(false);
    }
  }, []);

  const handleLoadMore = async () => {
    if (loadingMore || records.length >= total) return;
    setLoadingMore(true);
    try {
      const data = await fetchHistory(PAGE_SIZE, records.length, searchQuery, false);
      setRecords((prev) => [...prev, ...data.records]);
      setTotal(data.total);
    } catch (err) {
      console.error('Failed to load more records:', err);
    } finally {
      setLoadingMore(false);
    }
  };

  useEffect(() => {
    loadHistory(searchQuery, true);
  }, []);

  // Poll for newly completed generations every 8 seconds while Vault is open
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const data = await fetchHistory(1, 0, searchQuery, false);
        if (data.total > total) {
          loadHistory(searchQuery, false);
        }
      } catch (e) {
        // ignore polling errors
      }
    }, 8000);
    return () => clearInterval(interval);
  }, [total, searchQuery, loadHistory]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    loadHistory(searchQuery, false);
  };

  const copyToClipboard = (text: string, fieldName: string) => {
    navigator.clipboard.writeText(text);
    setCopiedField(fieldName);
    setTimeout(() => setCopiedField(null), 2000);
  };

  const handleSendToText2Img = (record: HistoryRecord) => {
    if (record.prompt) setPrompt(record.prompt);
    if (record.negative_prompt) setNegativePrompt(record.negative_prompt);
    if (record.model_id) setSelectedModelId(record.model_id);
    if (record.steps) setSteps(record.steps);
    if (record.guidance_scale) setCfgScale(record.guidance_scale);
    if (record.seed !== undefined) {
      setSeed(record.seed);
      setIsRandomSeed(false);
    }
    if (record.width && record.height) {
      setDimensions(record.width, record.height);
    }
    onNavigateToText2Img();
  };

  const handleDelete = async (record: HistoryRecord) => {
    if (!window.confirm('Delete this generation record?')) return;
    setDeletingId(record.id);
    try {
      const ok = await deleteHistoryItem(record.id);
      if (ok) {
        setRecords((prev) => prev.filter((r) => r.id !== record.id));
        setTotal((t) => Math.max(0, t - 1));
        if (selectedRecord?.id === record.id) {
          setSelectedRecord(records.find((r) => r.id !== record.id) || null);
        }
      }
    } catch (err) {
      console.error('Failed to delete history item:', err);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0B0E14] text-[#E8ECF4] overflow-hidden">
      {/* Header */}
      <div className="h-14 border-b border-[#252C3A] px-4 sm:px-6 flex items-center justify-between bg-[#151A24]/60 backdrop-blur shrink-0 gap-3">
        <div className="flex items-center space-x-3 min-w-0">
          <div className="w-8 h-8 rounded-lg bg-[#7C6CFF]/15 border border-[#7C6CFF]/30 flex items-center justify-center text-[#7C6CFF] shrink-0">
            <Archive className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <h1 className="text-sm font-bold tracking-wide flex items-center space-x-2 truncate">
              <span className="truncate">VAULT & GENERATION HISTORY</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#7C6CFF]/20 text-[#7C6CFF] font-mono font-medium shrink-0">
                {records.length > 0 ? `${records.length} of ${total}` : `${total}`} Records
              </span>
              {syncNotice && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950/60 border border-emerald-700/50 text-emerald-400 font-mono animate-fade-in shrink-0">
                  {syncNotice}
                </span>
              )}
            </h1>
            <p className="text-[10px] text-[#8993A7] hidden sm:block truncate">
              Browse previous artwork, inspect prompt parameters, and send settings back to generation
            </p>
          </div>
        </div>

        {/* Search */}
        <div className="flex items-center space-x-2 sm:space-x-3 shrink-0">
          <form onSubmit={handleSearchSubmit} className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-[#8993A7]" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search prompt or model..."
              className="bg-[#0F131B] border border-[#252C3A] focus:border-[#7C6CFF] rounded-control pl-8 pr-3 py-1.5 text-xs text-[#E8ECF4] placeholder-[#8993A7] w-40 sm:w-56 md:w-64 focus:outline-none"
            />
          </form>

          <button
            type="button"
            onClick={() => loadHistory(searchQuery, true)}
            disabled={loading || syncing}
            className="p-1.5 rounded-control bg-[#151A24] border border-[#252C3A] hover:border-[#7C6CFF]/40 text-[#8993A7] hover:text-[#E8ECF4] transition-all"
            title="Refresh & Sync Vault"
          >
            <RefreshCw className={`w-4 h-4 ${(loading || syncing) ? 'animate-spin text-[#7C6CFF]' : ''}`} />
          </button>
        </div>
      </div>

      {/* Main Grid + Inspector */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Gallery */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 min-w-0">
          {loading && records.length === 0 ? (
            <div className="h-64 flex flex-col items-center justify-center space-y-3 text-[#8993A7]">
              <RefreshCw className="w-8 h-8 animate-spin text-[#7C6CFF]" />
              <p className="text-xs font-mono">Loading Vault Generations...</p>
            </div>
          ) : records.length === 0 ? (
            <div className="h-64 flex flex-col items-center justify-center space-y-2 text-[#8993A7]">
              <ImageIcon className="w-10 h-10 stroke-1" />
              <p className="text-sm font-semibold text-[#E8ECF4]">No Generations Found</p>
              <p className="text-xs text-[#8993A7]">Generate artwork in Text-to-Image or Composition Studio to populate the Vault.</p>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-3.5">
              {records.map((record) => {
                const isSelected = selectedRecord?.id === record.id;
                return (
                  <div
                    key={record.id}
                    onClick={() => setSelectedRecord(record)}
                    className={`group relative rounded-card overflow-hidden cursor-pointer border transition-all duration-200 bg-[#151A24] flex flex-col ${
                      isSelected
                        ? 'border-[#7C6CFF] ring-2 ring-[#7C6CFF]/30 shadow-lg'
                        : 'border-[#252C3A] hover:border-[#7C6CFF]/50 hover:shadow-md'
                    }`}
                  >
                    <div className="aspect-square w-full bg-[#0F131B] relative overflow-hidden flex items-center justify-center">
                      {record.is_available ? (
                        <img
                          src={`http://127.0.0.1:8188${record.image_url}`}
                          alt={record.prompt}
                          loading="lazy"
                          className="w-full h-full object-cover object-center group-hover:scale-105 transition-transform duration-300"
                          onError={(e) => { (e.target as HTMLElement).style.display = 'none'; }}
                        />
                      ) : (
                        <div className="text-center p-3 text-[#8993A7]">
                          <ImageIcon className="w-6 h-6 mx-auto mb-1 opacity-40" />
                          <span className="text-[9px]">Image Unlinked</span>
                        </div>
                      )}

                      <div className="absolute top-1.5 left-1.5 bg-black/75 backdrop-blur-sm px-1.5 py-0.5 rounded text-[9px] font-mono text-[#35D6C5] border border-white/10">
                        {record.width}×{record.height}
                      </div>

                      <div className="absolute top-1.5 right-1.5 bg-black/75 backdrop-blur-sm px-1.5 py-0.5 rounded text-[9px] font-mono uppercase text-[#7C6CFF] border border-white/10">
                        {record.architecture || 'SDXL'}
                      </div>
                    </div>

                    <div className="p-2.5 space-y-1 bg-[#151A24]">
                      <p className="text-[11px] font-medium text-[#E8ECF4] line-clamp-2 leading-tight">
                        {record.prompt}
                      </p>
                      <div className="flex items-center justify-between text-[9px] text-[#8993A7] pt-1 border-t border-[#252C3A]/60 font-mono">
                        <span>Seed: {record.seed}</span>
                        <span>{record.steps} steps</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {records.length < total && (
              <div className="pt-8 pb-4 flex justify-center">
                <button
                  type="button"
                  onClick={handleLoadMore}
                  disabled={loadingMore}
                  className="px-6 py-2.5 rounded-lg bg-[#151A24] border border-[#252C3A] hover:border-[#7C6CFF]/60 hover:bg-[#1c2230] text-xs font-semibold text-[#E8ECF4] flex items-center space-x-2 transition-all shadow-sm"
                >
                  {loadingMore ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#7C6CFF]" />
                      <span>Loading more records...</span>
                    </>
                  ) : (
                    <>
                      <span>Load More Artwork</span>
                      <span className="text-[10px] text-[#8993A7] font-mono">
                        (Showing {records.length} of {total})
                      </span>
                    </>
                  )}
                </button>
              </div>
            )}
          </>
          )}
        </div>

        {/* Inspector Sidebar */}
        {selectedRecord && (
          <div className="w-80 lg:w-96 border-l border-[#252C3A] bg-[#151A24] flex flex-col h-full overflow-hidden shrink-0 shadow-2xl z-10">
            <div className="p-3.5 border-b border-[#252C3A] flex items-center justify-between">
              <h2 className="text-xs font-bold uppercase tracking-wider text-[#E8ECF4] flex items-center space-x-1.5">
                <Sliders className="w-3.5 h-3.5 text-[#7C6CFF]" />
                <span>Generation Inspector</span>
              </h2>
              <div className="flex items-center space-x-1">
                <button
                  type="button"
                  onClick={() => handleDelete(selectedRecord)}
                  disabled={deletingId === selectedRecord.id}
                  className="p-1.5 rounded hover:bg-[#FF5C6C]/10 text-[#8993A7] hover:text-[#FF5C6C] transition-colors"
                  title="Delete Record"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedRecord(null)}
                  className="p-1.5 rounded hover:bg-[#252C3A] text-[#8993A7] hover:text-[#E8ECF4] transition-colors"
                  title="Close Inspector"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div className="rounded-lg overflow-hidden border border-[#252C3A] bg-[#0F131B] relative aspect-square flex items-center justify-center group">
                {selectedRecord.is_available ? (
                  <img
                    src={`http://127.0.0.1:8188${selectedRecord.image_url}`}
                    alt={selectedRecord.prompt}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center text-[#8993A7] text-xs">
                    Image file moved or deleted from disk
                  </div>
                )}
              </div>

              <button
                type="button"
                onClick={() => handleSendToText2Img(selectedRecord)}
                className="w-full py-2.5 px-4 rounded-card bg-gradient-to-r from-[#7C6CFF] to-[#6355E6] hover:from-[#6D5CE6] hover:to-[#5446D1] text-white font-bold text-xs flex items-center justify-center space-x-2 shadow-lg shadow-[#7C6CFF]/20 transition-all hover:scale-[1.01] active:scale-[0.99]"
              >
                <Sparkles className="w-4 h-4" />
                <span>Send to Text → Image</span>
              </button>

              <div className="space-y-1.5 bg-[#0F131B] border border-[#252C3A] rounded-card p-3">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-[#8993A7]">
                    Prompt
                  </span>
                  <button
                    type="button"
                    onClick={() => copyToClipboard(selectedRecord.prompt, 'prompt')}
                    className="flex items-center space-x-1 text-[10px] text-[#7C6CFF] hover:text-[#9B8DFF]"
                  >
                    {copiedField === 'prompt' ? (
                      <>
                        <Check className="w-3 h-3 text-[#35D6C5]" />
                        <span className="text-[#35D6C5]">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3" />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                </div>
                <p className="text-xs text-[#E8ECF4] leading-relaxed select-text font-sans">
                  {selectedRecord.prompt}
                </p>
              </div>

              {selectedRecord.negative_prompt && (
                <div className="space-y-1.5 bg-[#0F131B] border border-[#252C3A] rounded-card p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase font-bold tracking-wider text-[#8993A7]">
                      Negative Prompt
                    </span>
                    <button
                      type="button"
                      onClick={() => copyToClipboard(selectedRecord.negative_prompt || '', 'neg_prompt')}
                      className="flex items-center space-x-1 text-[10px] text-[#7C6CFF] hover:text-[#9B8DFF]"
                    >
                      {copiedField === 'neg_prompt' ? (
                        <>
                          <Check className="w-3 h-3 text-[#35D6C5]" />
                          <span className="text-[#35D6C5]">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3 h-3" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                  <p className="text-xs text-[#8993A7] leading-relaxed select-text font-mono text-[11px]">
                    {selectedRecord.negative_prompt}
                  </p>
                </div>
              )}

              <div className="space-y-2">
                <span className="text-[10px] uppercase font-bold tracking-wider text-[#8993A7]">
                  Generation Parameters
                </span>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <div className="bg-[#0F131B] border border-[#252C3A] rounded p-2">
                    <div className="text-[9px] text-[#8993A7] uppercase">Model</div>
                    <div className="text-[#E8ECF4] truncate font-medium" title={selectedRecord.model_id}>
                      {selectedRecord.model_id}
                    </div>
                  </div>

                  <div className="bg-[#0F131B] border border-[#252C3A] rounded p-2">
                    <div className="text-[9px] text-[#8993A7] uppercase">Architecture</div>
                    <div className="text-[#35D6C5] font-medium uppercase">
                      {selectedRecord.architecture || 'SDXL'}
                    </div>
                  </div>

                  <div className="bg-[#0F131B] border border-[#252C3A] rounded p-2">
                    <div className="text-[9px] text-[#8993A7] uppercase">Seed</div>
                    <div className="text-[#E8ECF4] font-medium flex items-center justify-between">
                      <span>{selectedRecord.seed}</span>
                      <button
                        type="button"
                        onClick={() => copyToClipboard(selectedRecord.seed.toString(), 'seed')}
                        className="text-[#8993A7] hover:text-[#7C6CFF]"
                      >
                        <Copy className="w-2.5 h-2.5" />
                      </button>
                    </div>
                  </div>

                  <div className="bg-[#0F131B] border border-[#252C3A] rounded p-2">
                    <div className="text-[9px] text-[#8993A7] uppercase">Resolution</div>
                    <div className="text-[#E8ECF4] font-medium">
                      {selectedRecord.width} × {selectedRecord.height}
                    </div>
                  </div>

                  <div className="bg-[#0F131B] border border-[#252C3A] rounded p-2">
                    <div className="text-[9px] text-[#8993A7] uppercase">Steps & CFG</div>
                    <div className="text-[#E8ECF4] font-medium">
                      {selectedRecord.steps} steps / {selectedRecord.guidance_scale} CFG
                    </div>
                  </div>

                  <div className="bg-[#0F131B] border border-[#252C3A] rounded p-2">
                    <div className="text-[9px] text-[#8993A7] uppercase">Generation Time</div>
                    <div className="text-[#E8ECF4] font-medium">
                      {selectedRecord.generation_time_ms
                        ? `${(selectedRecord.generation_time_ms / 1000).toFixed(1)}s`
                        : 'N/A'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
