import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient } from "../services/queryClient";

const baseUrl = (import.meta as any).env?.VITE_API_BASE_URL || "/";

async function api(path: string, init: RequestInit = {}) {
  const res = await fetch(new URL(path, baseUrl), {
    ...init,
    headers: { "Content-Type": "application/json", ...(init.headers || {}) },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
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
    mutationFn: (body: any) =>
      api(`/research/query`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: (data) => {
      if (data?.research_id) {
        queryClient.invalidateQueries({ queryKey: ["research", "status", data.research_id] });
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