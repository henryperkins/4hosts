import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient } from "../services/queryClient";

const baseUrl = "/"; // Use Vite proxy with relative paths so cookies stay on 5173

async function api<T = unknown>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(new URL(path, window.location.origin).pathname.startsWith('/')
    ? path
    : new URL(path, baseUrl).toString(), {
    ...init,
    headers: { "Content-Type": "application/json", ...(init.headers || {}) },
    credentials: 'include'
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  // Parse JSON safely
  const text = await res.text();
  try {
    return (text ? JSON.parse(text) : {}) as T;
  } catch {
    return {} as T;
  }
}

export function useResearchStatus(researchId: string) {
  return useQuery({
    queryKey: ["research", "status", researchId],
    queryFn: () => api(`/research/status/${researchId}`),
    enabled: !!researchId,
    staleTime: 5000,
  });
}

export function useResearchResults(researchId: string) {
  return useQuery({
    queryKey: ["research", "results", researchId],
    queryFn: () => api(`/research/results/${researchId}`),
    enabled: !!researchId,
    staleTime: 30_000,
  });
}

export function useSubmitResearch() {
  return useMutation({
    mutationFn: (body: { query: string; options: import('../types').ResearchOptions }) =>
      api(`/research/query`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: (data) => {
      const d = data as { research_id?: string }
      if (d?.research_id) {
        queryClient.invalidateQueries({ queryKey: ["research", "status", d.research_id] });
      }
    },
  });
}

export function useCancelResearch() {
  return useMutation({
    mutationFn: (researchId: string) => api(`/research/cancel/${researchId}`, { method: "POST" }),
    onSuccess: (_data, researchId) => {
      queryClient.invalidateQueries({ queryKey: ["research", "status", researchId] });
      queryClient.invalidateQueries({ queryKey: ["research", "results", researchId] });
    },
  });
}
