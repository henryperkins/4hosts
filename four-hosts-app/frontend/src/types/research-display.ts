import type { ResearchResult, AnswerSection } from '../types'

export interface ResultsDisplayEnhancedProps {
  results: ResearchResult
}

export type ContextIsolationDetails = {
  focus_areas?: string[]
  patterns?: number
}

export type ContextLayersInfo = {
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
  isolated_findings?: ContextIsolationDetails
}

export type ActionItem = {
  action?: string
  timeframe?: string
  priority?: 'high' | 'medium' | 'low' | string
  paradigm?: string
  owner?: string
  due_date?: string
  description?: string
}

export type AnswerCitation = {
  id?: string
  title?: string
  source?: string
  url?: string
  credibility_score?: number
  paradigm_alignment?: string
}

export type EvidenceQuoteRaw =
  | string
  | {
      quote: string
      url?: string
      domain?: string
      title?: string
      credibility_score?: number
      published_date?: string
      id?: string
    }

export type EvidenceQuote = {
  quote: string
  url: string
  domain?: string
  title?: string
  credibility_score?: number
  published_date?: string
  id: string
}

export type AnswerMetadataType = Partial<ResearchResult['metadata']> & {
  evidence_quotes?: EvidenceQuoteRaw[]
  statistical_insights?: number
}

export type PrimaryAnswer = {
  citations?: AnswerCitation[]
  sections?: AnswerSection[]
  action_items?: ActionItem[]
  summary?: string
  metadata?: AnswerMetadataType
}

export type SecondaryPerspective = {
  title?: string
  content?: string
  paradigm?: string
  sources_count?: number
  confidence?: number
}

export type ConflictItem = { description?: string }

export type IntegratedSynthesis = {
  primary_answer?: PrimaryAnswer
  secondary_perspective?: SecondaryPerspective
  integrated_summary?: string
  synergies?: string[]
  conflicts_identified?: ConflictItem[]
  confidence_score?: number
}

export type ResearchSource = {
  url?: string
  domain?: string
  title?: string
  snippet?: string
  key_quote?: string
  credibility_explanation?: string
  credibility_score?: number
  source_category?: string
  published_date?: string
  raw_data?: { below_relevance_threshold?: boolean }
}

export type AgentTraceEntry = {
  step?: string
  iteration?: number
  coverage?: number
  proposed_queries?: string[]
  warnings?: string[]
}
