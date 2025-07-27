export type Paradigm = 'dolores' | 'teddy' | 'bernard' | 'maeve'

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
    }
  }
  answer: {
    summary: string
    sections: Array<{
      title: string
      paradigm: Paradigm
      content: string
      confidence: number
      sources_count: number
    }>
    action_items: Array<{
      priority: string
      action: string
      timeframe: string
      paradigm: Paradigm
    }>
    citations: Array<{
      id: string
      source: string
      title: string
      url: string
      credibility_score: number
      paradigm_alignment: Paradigm
    }>
  }
  metadata: {
    total_sources_analyzed: number
    high_quality_sources: number
    search_queries_executed: number
    processing_time_seconds: number
    paradigms_used: Paradigm[]
  }
}

// Authentication types
export interface User {
  id: string
  username: string
  email: string
  created_at: string
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
}