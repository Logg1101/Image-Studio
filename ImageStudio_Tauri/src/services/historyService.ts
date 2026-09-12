import { HistoryResponse } from "../types/history";

const BRIDGE_URL = "http://127.0.0.1:8188";

export async function fetchHistory(
  limit: number = 60,
  offset: number = 0,
  search: string = "",
  sync: boolean = false
): Promise<HistoryResponse> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
    search: search.trim(),
    sync: sync ? "true" : "false",
  });
  const res = await fetch(`${BRIDGE_URL}/api/history?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch history: ${res.statusText}`);
  }
  return res.json();
}

export async function syncHistory(): Promise<{ success: boolean; synced: number; total: number }> {
  const res = await fetch(`${BRIDGE_URL}/api/history/sync`, {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(`Failed to sync history: ${res.statusText}`);
  }
  return res.json();
}

export async function deleteHistoryItem(recordId: number): Promise<boolean> {
  const res = await fetch(`${BRIDGE_URL}/api/history/${recordId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`Failed to delete history item: ${res.statusText}`);
  }
  const data = await res.json();
  return data.success;
}
