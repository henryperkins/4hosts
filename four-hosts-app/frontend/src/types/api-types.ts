import { z } from 'zod'

// Properly typed API responses to replace 'any' types

export interface AuthTokenResponse {
  access_token: string
  refresh_token?: string
  token_type: string
  expires_in?: number
}

export interface AuthUser {
  id: string
  email: string
  name?: string
  role: 'FREE' | 'BASIC' | 'PRO' | 'ENTERPRISE' | 'ADMIN'
  created_at: string
  is_active: boolean
}

export interface LoginResponse {
  success: boolean
  user: AuthUser
  message?: string
}

export interface ErrorResponse {
  error: string
  detail?: string | Record<string, unknown>
  status?: number
  timestamp?: string
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy'
  services: {
    database: boolean
    cache: boolean
    search_apis: {
      google: boolean
      brave: boolean
      arxiv: boolean
      pubmed: boolean
    }
  }
  timestamp: string
}

export interface ResearchQuery {
  query: string
  paradigm?: string
  deep_research?: boolean
  max_results?: number
}

export interface ResearchResult {
  id: string
  query: string
  primary_paradigm: string
  secondary_paradigm?: string
  search_results: SearchResult[]
  answer?: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  created_at: string
  processing_time?: number
  error?: string
}

export interface SearchResult {
  title: string
  url: string
  snippet: string
  source: 'google' | 'brave' | 'arxiv' | 'pubmed'
  relevance_score?: number
  published_date?: string
  authors?: string[]
}

export interface WebSocketMessage {
  type:
    | 'status'
    | 'progress'
    | 'result'
    | 'error'
    // Connection/system events
    | 'connected'
    | 'disconnected'
    | 'ping'
    | 'pong'
    | 'system.notification'
    | 'rate_limit.warning'
    // Research events
    | 'research_progress'
    | 'research_phase_change'
    | 'source_found'
    | 'source_analyzed'
    | 'research_completed'
    | 'research_failed'
    | 'research_started'
    // Search/analysis events
    | 'search.started'
    | 'search.completed'
    | 'search.retry'
    | 'credibility.check'
    | 'deduplication.progress'
    | 'evidence_builder_skipped'
  data: {
    status?: string
    progress?: number
    message?: string
    result?: ResearchResult
    error?: string
    phase?: string
    old_phase?: string
    new_phase?: string
    source?: {
      title: string
      domain: string
      snippet: string
      url: string
      credibility_score?: number
    }
    total_sources?: number
    api?: string
    engine?: string
    server?: string
    tool?: string
    limit_type?: string
    remaining?: number
    reset_time?: string
    duplicates_removed?: number
    unique_sources?: number
    total_checked?: number
    source_id?: string
    analyzed_count?: number
    duration_seconds?: number
    query?: string
    index?: number
    total?: number
    results_count?: number
    domain?: string
    score?: number
    removed?: number
    // New fields for determinate progress tracking
    items_done?: number
    items_total?: number
    eta_seconds?: number
  }
  timestamp: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  limit: number
  has_next: boolean
  has_prev: boolean
}

export interface FeedbackSubmission {
  research_id: string
  rating: number
  comment?: string
  improvements?: string[]
}

export interface MetricsData {
  total_queries: number
  active_research: number
  paradigm_distribution: Record<string, number>
  average_processing_time: number
  cache_hit_rate: number
  system_health: 'healthy' | 'degraded' | 'unhealthy'
}

// Extended metrics snapshot (from /system/extended-stats)
export interface ExtendedStatsLatencyDistribution {
  count: number
  avg: number
  p50: number
  p95: number
  p99: number
}

export interface ExtendedStatsLatency {
  [stage: string]: ExtendedStatsLatencyDistribution
}

export interface ExtendedStatsFallbackRates {
  [stage: string]: number
}

export interface ExtendedStatsLLMUsageModel {
  calls: number
  tokens_in: number
  tokens_out: number
}

export interface ExtendedStatsLLMUsage {
  [model: string]: ExtendedStatsLLMUsageModel
}

export interface ExtendedStatsQuality {
  critic_avg_score: number
  hallucination_rate: number
  evidence_coverage_ratio: number
}

export interface ExtendedStatsCountersBucket {
  [labelCombo: string]: number
}

export interface ExtendedStatsCounters {
  [counterName: string]: ExtendedStatsCountersBucket
}

export interface ExtendedStatsSnapshot {
  latency: ExtendedStatsLatency
  fallback_rates: ExtendedStatsFallbackRates
  llm_usage: ExtendedStatsLLMUsage
  paradigm_distribution: Record<string, number>
  quality: ExtendedStatsQuality
  counters: ExtendedStatsCounters
  timestamp: string
}

export function isExtendedStatsSnapshot(data: unknown): data is ExtendedStatsSnapshot {
  return (
    typeof data === 'object' &&
    data !== null &&
    'latency' in data &&
    'fallback_rates' in data &&
    'llm_usage' in data &&
    'timestamp' in data
  )
}

// Type guards for runtime validation
export function isErrorResponse(data: unknown): data is ErrorResponse {
  return (
    typeof data === 'object' &&
    data !== null &&
    'error' in data &&
    typeof (data as ErrorResponse).error === 'string'
  )
}

export function isLoginResponse(data: unknown): data is LoginResponse {
  return (
    typeof data === 'object' &&
    data !== null &&
    'success' in data &&
    'user' in data &&
    typeof (data as LoginResponse).success === 'boolean'
  )
}

export function isWebSocketMessage(data: unknown): data is WebSocketMessage {
  if (
    typeof data !== 'object' ||
    data === null ||
    !('type' in data) ||
    !('data' in data)
  ) {
    return false
  }
  const t = (data as { type: string }).type
  const allowed: WebSocketMessage['type'][] = [
    'status', 'progress', 'result', 'error',
    'connected', 'disconnected', 'ping', 'pong', 'system.notification', 'rate_limit.warning',
    'research_progress', 'research_phase_change', 'source_found', 'source_analyzed', 'research_completed', 'research_failed', 'research_started',
    'search.started', 'search.completed', 'search.retry', 'credibility.check', 'deduplication.progress'
  ]
  return allowed.includes(t as WebSocketMessage['type'])
}

// ---------------------------------------------------------------------------
// Runtime validation for enriched research response payload
// ---------------------------------------------------------------------------

export const SourceItemSchema = z.object({
  title: z.string(),
  url: z.string().url().or(z.string().min(1)),
  domain: z.string().default(''),
  snippet: z.string().optional(),
  content: z.string().optional(),
  credibility_score: z.number().min(0).max(1).default(0),
  published_date: z.string().optional(),
  result_type: z.string().optional(),
  source_api: z.string().optional(),
  source_category: z.string().optional(),
  metadata: z.record(z.string(), z.any()).default({}),
})

export const AnswerSectionSchema = z.object({
  title: z.string().default(''),
  paradigm: z.string().default(''),
  content: z.string().default(''),
  confidence: z.number().default(0),
  sources_count: z.number().default(0),
  citations: z.array(z.string()).default([]),
  key_insights: z.array(z.string()).default([]),
})

export const ActionItemSchema = z.object({
  priority: z.string().default(''),
  action: z.string().default(''),
  timeframe: z.string().default(''),
  paradigm: z.string().default(''),
  owner: z.string().optional(),
  due_date: z.string().optional(),
})

export const CitationSchema = z.object({
  id: z.string().default(''),
  source: z.string().default(''),
  title: z.string().default(''),
  url: z.string().url().or(z.string().min(1)).default(''),
  credibility_score: z.number().optional(),
  paradigm_alignment: z.string().optional(),
})

export const ConflictSchema = z.object({
  conflict_type: z.string().optional(),
  description: z.string().default(''),
  primary_paradigm_view: z.string().optional(),
  secondary_paradigm_view: z.string().optional(),
  confidence: z.number().optional(),
})

export const ContextLayersSchema = z.object({
  write_focus: z.string().optional(),
  compression_ratio: z.number().optional(),
  token_budget: z.number().optional(),
  isolation_strategy: z.string().optional(),
  search_queries_count: z.number().optional(),
  layer_times: z.record(z.string(), z.number()).optional(),
  budget_plan: z.record(z.string(), z.number()).optional(),
  rewrite_primary: z.string().optional(),
  rewrite_alternatives: z.number().optional(),
  optimize_primary: z.string().optional(),
  optimize_variations_count: z.number().optional(),
  refined_queries_count: z.number().optional(),
  isolated_findings: z
    .object({
      focus_areas: z.array(z.string()).optional(),
      patterns: z.number().optional(),
    })
    .optional(),
})

export const ParadigmRoleSchema = z.object({
  paradigm: z.string(),
  confidence: z.number().optional(),
})

export const ParadigmAnalysisSchema = z.object({
  primary: ParadigmRoleSchema,
  secondary: ParadigmRoleSchema.optional(),
})

export const CredibilitySummarySchema = z.object({
  average_score: z.number().default(0),
  score_distribution: z
    .object({ high: z.number().default(0), medium: z.number().default(0), low: z.number().default(0) })
    .partial()
    .default({}),
  high_credibility_count: z.number().optional(),
  high_credibility_ratio: z.number().optional(),
})

export const ResearchMetadataSchema = z.object({
  processing_time_seconds: z.number().optional(),
  total_results: z.number().default(0),
  total_sources_analyzed: z.number().default(0),
  high_quality_sources: z.number().default(0),
  credibility_summary: CredibilitySummarySchema.default({
    average_score: 0,
    score_distribution: { high: 0, medium: 0, low: 0 },
  }),
  category_distribution: z.record(z.string(), z.number()).default({}),
  bias_distribution: z.record(z.string(), z.number()).default({}),
  paradigm_fit: z.record(z.string(), z.any()).optional(),
  research_depth: z.string().optional(),
  agent_trace: z.array(z.record(z.string(), z.any())).optional(),
  queries_executed: z.number().default(0),
  sources_used: z.array(z.string()).default([]),
  deduplication_stats: z.record(z.string(), z.any()).default({}),
  search_metrics: z.record(z.string(), z.any()).default({}),
  paradigm: z.string().optional(),
  context_layers: ContextLayersSchema.optional(),
  evidence_quotes: z.array(z.record(z.string(), z.any())).optional(),
  classification_details: z.record(z.string(), z.any()).optional(),
})

export const GeneratedAnswerSchema = z.object({
  summary: z.string().default(''),
  sections: z.array(AnswerSectionSchema).default([]),
  action_items: z.array(ActionItemSchema).default([]),
  citations: z.array(CitationSchema).default([]),
  confidence_score: z.number().optional(),
  metadata: z.record(z.string(), z.any()).default({}),
})

export const ResearchResponseSchema = z.object({
  research_id: z.string(),
  query: z.string(),
  status: z.string().default('ok'),
  paradigm_analysis: ParadigmAnalysisSchema,
  sources: z.array(SourceItemSchema).default([]),
  results: z.array(SourceItemSchema).default([]),
  answer: GeneratedAnswerSchema.optional(),
  integrated_synthesis: z
    .object({
      primary_answer: GeneratedAnswerSchema,
      integrated_summary: z.string().optional(),
      secondary_perspective: AnswerSectionSchema.nullable().optional(),
      synergies: z.array(z.string()).default([]),
      conflicts_identified: z.array(ConflictSchema).default([]),
    })
    .optional(),
  metadata: ResearchMetadataSchema,
  cost_info: z.record(z.string(), z.any()).default({}),
  export_formats: z.record(z.string(), z.string()).default({}),
})

export type SourceItem = z.infer<typeof SourceItemSchema>
export type ResearchResponse = z.infer<typeof ResearchResponseSchema>
