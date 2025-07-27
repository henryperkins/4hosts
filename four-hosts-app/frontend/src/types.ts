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