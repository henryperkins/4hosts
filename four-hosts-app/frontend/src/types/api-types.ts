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
  data: {
    status?: 'pending' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
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
