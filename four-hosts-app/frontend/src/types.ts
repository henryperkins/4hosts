export type Paradigm = 'dolores' | 'teddy' | 'bernard' | 'maeve'

export interface ResearchOptions {
  depth: 'quick' | 'standard' | 'deep'
  include_secondary?: boolean
  max_sources?: number
  enable_real_search?: boolean
  language?: string
  region?: string
  enable_ai_classification?: boolean
}

export interface ParadigmClassification {
  primary: Paradigm
  secondary: Paradigm | null
  distribution: Record<string, number>
  confidence: number
  explanation: Record<string, string>
}

export interface ResearchResult {
  research_id: string
  query: string
  status: string
  paradigm_analysis: {
    primary: {
      paradigm: Paradigm
      confidence: number
      approach: string
      focus: string
    },
    secondary?: {
        paradigm: Paradigm
        confidence: number
        approach: string
        focus: string
    },
    context_engineering?: {
      compression_ratio: number
      token_budget: number
      isolation_strategy: string
      search_queries_count: number
    }
  },
  answer: GeneratedAnswer
  integrated_synthesis?: IntegratedSynthesis
  sources: SourceResult[]
  metadata: {
    total_sources_analyzed: number
    high_quality_sources: number
    search_queries_executed: number
    processing_time_seconds: number
    answer_generation_time?: number
    synthesis_quality?: number
    paradigms_used: Paradigm[]
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

// Authentication types
export interface User {
  id: string
  username: string
  email: string
  created_at?: string
  role?: string
  preferences?: UserPreferences
}

export interface UserPreferences {
  default_paradigm?: Paradigm
  default_depth?: 'quick' | 'standard' | 'deep'
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
  data: any
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
}
