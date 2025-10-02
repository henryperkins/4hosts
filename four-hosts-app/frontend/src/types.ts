export type Paradigm = 'dolores' | 'teddy' | 'bernard' | 'maeve'

export interface ResearchOptions {
  depth: 'quick' | 'standard' | 'deep' | 'deep_research'
  include_secondary?: boolean
  max_sources?: number
  enable_real_search?: boolean
  language?: string
  region?: string
  enable_ai_classification?: boolean
  paradigm_override?: Paradigm | null
  // Deep research specific options
  search_context_size?: 'small' | 'medium' | 'large'
  user_location?: {
    country?: string
    city?: string
  }
}

export interface ParadigmClassification {
  primary: Paradigm
  secondary: Paradigm | null
  distribution: Record<string, number>
  confidence: number
  explanation: Record<string, string>
  // Optional structured signals from backend classification (for UI hints)
  signals?: Partial<Record<Paradigm, {
    keywords?: string[]
    intent_signals?: string[]
  }>>
}

export interface ResearchResult {
  research_id: string
  query: string
  status: string
  message?: string
  paradigm_analysis: {
    primary: {
      paradigm: Paradigm
      confidence: number
      approach?: string
      focus?: string
    },
    secondary?: {
        paradigm: Paradigm
        confidence?: number
        approach?: string
        focus?: string
    },
    context_engineering?: {
      compression_ratio: number
      token_budget: number
      isolation_strategy: string
      search_queries_count: number
    }
  },
  answer?: GeneratedAnswer
  integrated_synthesis?: IntegratedSynthesis
  mesh_synthesis?: MeshSynthesis | null
  sources: SourceResult[]
  results?: SourceResult[]
  export_formats?: Record<string, string>
  metadata: {
    total_results?: number
    total_sources_analyzed: number
    high_quality_sources: number
    search_queries_executed?: number
    queries_executed?: number
    sources_used?: string[]
    processing_time_seconds: number
    answer_generation_time?: number
    synthesis_quality?: number
    paradigms_used?: Paradigm[]
    deep_research_enabled?: boolean
    research_depth?: string
    evidence_builder_skipped?: boolean
    error_message?: string
    credibility_summary?: {
      average_score?: number
      high_credibility_count?: number
      high_credibility_ratio?: number
      score_distribution?: Record<string, number>
    }
    deduplication_stats?: Record<string, unknown>
    search_metrics?: Record<string, unknown>
    analysis_metrics?: {
      duration_ms?: number
      sources_total?: number
      sources_completed?: number
      progress_updates?: number
      updates_per_second?: number
      avg_update_gap_ms?: number
      p95_update_gap_ms?: number
      first_update_gap_ms?: number
      last_update_gap_ms?: number
      cancelled?: boolean
    }
    context_layers?: {
      write_focus?: string
      compression_ratio?: number
      token_budget?: number
      isolation_strategy?: string
      search_queries_count?: number
      layer_times?: Record<string, number>
      budget_plan?: Record<string, number>
      rewrite_primary?: string
      rewrite_alternatives?: number
      optimize_primary?: string
      optimize_variations_count?: number
      refined_queries_count?: number
      isolated_findings?: {
        focus_areas?: string[]
        patterns?: number
      }
    }
    agent_trace?: AgentTraceEvent[]
    actionable_content_ratio?: number
    bias_check?: {
      balanced: boolean
      domain_diversity: number
      dominant_domain?: string | null
      dominant_share?: number
      unique_types?: number
    }
    category_distribution?: Record<string, number>
    bias_distribution?: Record<string, number>
    paradigm_fit?: {
      primary: Paradigm
      confidence: number
      margin: number
    }
    evidence_quotes?: Array<Record<string, unknown>>
    paradigm?: Paradigm | string
    warnings?: unknown[]
    degraded?: boolean
  },
  cost_info?: {
    search_api_costs?: number
    llm_costs?: number
    total?: number
  }
}

export interface GeneratedAnswer {
  summary: string
  sections: AnswerSection[]
  action_items: ActionItem[]
  citations: Citation[]
  // New optional backend-provided fields (non-breaking)
  confidence_score?: number
  synthesis_quality?: number
  generation_time?: number
  metadata: Record<string, unknown>
}

export interface AnswerSection {
  title: string
  paradigm: Paradigm
  content: string
  confidence: number
  sources_count: number
  citations: string[]
  key_insights: string[]
}

export interface ActionItem {
  priority: string
  action: string
  timeframe: string
  paradigm: Paradigm
  // Optional content enhancements for ownership and deadlines
  owner?: string
  due_date?: string // ISO date string
  description?: string
}

export interface Citation {
  id: string
  source: string
  title: string
  url: string
  credibility_score: number
  paradigm_alignment: Paradigm
}

export interface Conflict {
    conflict_type: string
    description: string
    primary_paradigm_view: string
    secondary_paradigm_view: string
    confidence: number
}

export interface IntegratedSynthesis {
    primary_answer: GeneratedAnswer
    secondary_perspective: AnswerSection | null
    conflicts_identified: Conflict[]
    synergies: string[]
    integrated_summary: string
    confidence_score: number
}

export interface MeshSynthesisStance {
  paradigm?: string
  perspective?: string
  key_points?: string[]
}

export interface MeshSynthesis {
  integrated?: string
  synthesis?: string
  stances?: MeshSynthesisStance[]
  synergies?: string[]
  tensions?: string[]
}

// Authentication types
export interface User {
  id: string
  username: string
  email: string
  created_at?: string
  role?: 'free' | 'basic' | 'pro' | 'enterprise' | 'admin'
  preferences?: UserPreferences
}

export interface UserPreferences {
  default_paradigm?: Paradigm
  default_depth?: 'quick' | 'standard' | 'deep' | 'deep_research'
  enable_real_search?: boolean
  enable_ai_classification?: boolean
  theme?: 'light' | 'dark'
}

export interface AuthState {
  isAuthenticated: boolean
  user: User | null
  loading: boolean
}

// WebSocket types
export interface WSMessage {
  type: 'status_update' | 'progress' | 'result' | 'error'
  research_id: string
  data: unknown
}

// Research history
export interface ResearchHistoryItem {
  research_id: string
  query: string
  paradigm: Paradigm
  status: string
  created_at: string
  processing_time?: number
  options?: {
    depth?: string
    max_sources?: number
  }
  summary?: {
    answer_preview: string
    source_count: number
    total_cost: number
  }
}

export interface SourceResult {
  title: string
  url: string
  snippet: string
  domain: string
  credibility_score: number
  published_date?: string
  source_type?: string
  result_type?: string
  source_category?: string
  credibility_explanation?: string
  content?: string
  source_api?: string
  metadata?: Record<string, unknown>
  raw_data?: {
    below_relevance_threshold?: boolean
    [key: string]: unknown
  }
}

export interface AgentTraceEvent {
  step?: string
  iteration?: number
  coverage?: number
  proposed_queries?: string[]
  [key: string]: unknown
}
