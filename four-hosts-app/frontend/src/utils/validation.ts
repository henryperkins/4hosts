import { z } from 'zod'
import type { 
  LoginResponse,
  ErrorResponse,
  ResearchResult,
  WebSocketMessage,
  MetricsData
} from '../types/api-types'

// Zod schemas for runtime validation
export const AuthUserSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  name: z.string().optional(),
  role: z.enum(['FREE', 'BASIC', 'PRO', 'ENTERPRISE', 'ADMIN']).or(
    z.enum(['free', 'basic', 'pro', 'enterprise', 'admin']).transform(val => val.toUpperCase() as 'FREE' | 'BASIC' | 'PRO' | 'ENTERPRISE' | 'ADMIN')
  ),
  created_at: z.string(),
  is_active: z.boolean()
})

export const LoginResponseSchema = z.object({
  success: z.boolean(),
  user: AuthUserSchema,
  message: z.string().optional()
})

export const ErrorResponseSchema = z.object({
  error: z.string(),
  detail: z.union([z.string(), z.record(z.string(), z.unknown())]).optional(),
  status: z.number().optional(),
  timestamp: z.string().optional()
})

export const SystemHealthSchema = z.object({
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  services: z.object({
    database: z.boolean(),
    cache: z.boolean(),
    search_apis: z.object({
      google: z.boolean(),
      brave: z.boolean(),
      arxiv: z.boolean(),
      pubmed: z.boolean()
    })
  }),
  timestamp: z.string()
})

export const ResearchResultSchema = z.object({
  id: z.string(),
  query: z.string(),
  primary_paradigm: z.string(),
  secondary_paradigm: z.string().optional(),
  search_results: z.array(z.object({
    title: z.string(),
    url: z.string().url(),
    snippet: z.string(),
    source: z.enum(['google', 'brave', 'arxiv', 'pubmed']),
    relevance_score: z.number().optional(),
    published_date: z.string().optional(),
    authors: z.array(z.string()).optional()
  })),
  answer: z.string().optional(),
  status: z.enum(['pending', 'processing', 'completed', 'failed']),
  created_at: z.string(),
  processing_time: z.number().optional(),
  error: z.string().optional()
})

export const WebSocketMessageSchema = z.object({
  type: z.enum([
    // Generic
    'status', 'progress', 'result', 'error',
    // Connection/system events
    'connected', 'disconnected', 'ping', 'pong', 'system.notification', 'rate_limit.warning',
    // Research-specific events
    'research_progress', 'research_phase_change', 'source_found', 'source_analyzed',
    'research_completed', 'research_failed', 'research_started',
    // Search and analysis events
    'search.started', 'search.completed', 'credibility.check', 'deduplication.progress'
  ]),
  data: z.object({
    status: z.string().optional(),
    progress: z.number().optional(),
    message: z.string().optional(),
    result: ResearchResultSchema.optional(),
    error: z.string().optional(),
    phase: z.string().optional(),
    old_phase: z.string().optional(),
    new_phase: z.string().optional(),
    source: z.object({
      title: z.string(),
      domain: z.string(),
      snippet: z.string(),
      url: z.string(),
      credibility_score: z.number().optional()
    }).optional(),
    total_sources: z.number().optional(),
    api: z.string().optional(),
    engine: z.string().optional(),
    server: z.string().optional(),
    tool: z.string().optional(),
    limit_type: z.string().optional(),
    remaining: z.number().optional(),
    reset_time: z.string().optional(),
    duplicates_removed: z.number().optional(),
    unique_sources: z.number().optional(),
    total_checked: z.number().optional(),
    source_id: z.string().optional(),
    analyzed_count: z.number().optional(),
    duration_seconds: z.number().optional(),
    query: z.string().optional(),
    index: z.number().optional(),
    total: z.number().optional(),
    results_count: z.number().optional(),
    domain: z.string().optional(),
    score: z.number().optional(),
    removed: z.number().optional()
  }),
  timestamp: z.string()
})

export const MetricsDataSchema = z.object({
  total_queries: z.number(),
  active_research: z.number(),
  paradigm_distribution: z.record(z.string(), z.number()),
  average_processing_time: z.number(),
  cache_hit_rate: z.number(),
  system_health: z.enum(['healthy', 'degraded', 'unhealthy'])
})

// Lightweight schema for context metrics used by ContextMetricsPanel
export const ContextMetricsSchema = z.object({
  context_pipeline: z.object({
    total_processed: z.number().default(0),
    average_processing_time: z.number().default(0),
    layer_metrics: z.record(z.string(), z.number()).default({}),
  })
})

// Validation functions with proper error handling
export function validateLoginResponse(data: unknown): LoginResponse {
  try {
    return LoginResponseSchema.parse(data)
  } catch (error) {
    console.error('Invalid login response:', error)
    throw new Error('Invalid login response format')
  }
}

export function validateErrorResponse(data: unknown): ErrorResponse | null {
  try {
    return ErrorResponseSchema.parse(data)
  } catch {
    return null
  }
}

export function validateWebSocketMessage(data: unknown): WebSocketMessage | null {
  try {
    return WebSocketMessageSchema.parse(data)
  } catch (error) {
    console.error('Invalid WebSocket message:', error)
    return null
  }
}

export function validateMetricsData(data: unknown): MetricsData {
  try {
    return MetricsDataSchema.parse(data)
  } catch (error) {
    console.error('Invalid metrics data:', error)
    throw new Error('Invalid metrics data format')
  }
}

export function validateContextMetrics(data: unknown): z.infer<typeof ContextMetricsSchema> {
  try {
    return ContextMetricsSchema.parse(data)
  } catch (error) {
    console.error('Invalid context metrics:', error)
    // Provide a safe default rather than throwing to avoid breaking the panel
    return { context_pipeline: { total_processed: 0, average_processing_time: 0, layer_metrics: {} } }
  }
}

export function validateResearchResult(data: unknown): ResearchResult {
  try {
    return ResearchResultSchema.parse(data)
  } catch (error) {
    console.error('Invalid research result:', error)
    throw new Error('Invalid research result format')
  }
}
