import { useMemo } from 'react'
import type { ResearchResult } from '../types'
import { getCredibilityBand } from '../utils/credibility'
import type {
  AnswerCitation,
  AnswerMetadataType,
  ContextLayersInfo,
  EvidenceQuote,
  EvidenceQuoteRaw,
  IntegratedSynthesis,
  PrimaryAnswer,
  ResearchSource,
} from '../types/research-display'

export type ConfidenceInfo = {
  percentage: number
  band: 'High' | 'Medium' | 'Low'
  because: string
}

export interface ResearchDisplayData {
  results: ResearchResult
  baseAnswer: PrimaryAnswer | null
  metadata: AnswerMetadataType
  contextLayers: ContextLayersInfo | null
  summary: string
  citations: AnswerCitation[]
  actionItems: PrimaryAnswer['action_items']
  evidenceQuotes: EvidenceQuote[]
  sources: ResearchSource[]
  domainInfo: Record<string, { category?: string; explanation?: string; score?: number }>
  evidenceSnapshot: {
    total: number
    strong: number
    moderate: number
    weak: number
    window?: string
  }
  categoryDistribution: Record<string, number>
  biasDistribution: Record<string, number>
  credibilityDistribution: Record<'high' | 'medium' | 'low', number>
  actionableRatio: number
  biasCheck: ResearchResult['metadata']['bias_check']
  analysisMetrics: AnalysisMetrics | null
  confidenceInfo: ConfidenceInfo
  integratedSynthesis?: IntegratedSynthesis
  meshSynthesis?: ResearchResult['mesh_synthesis']
  fetchedAt: string
  warnings: unknown[] | undefined
  degraded: boolean
  status: string
}

export interface AnalysisMetrics {
  durationMs?: number
  sourcesTotal?: number
  sourcesCompleted?: number
  progressUpdates?: number
  updatesPerSecond?: number
  avgUpdateGapMs?: number
  p95UpdateGapMs?: number
  firstUpdateGapMs?: number
  lastUpdateGapMs?: number
  cancelled?: boolean
}

const normaliseEvidenceQuote = (item: EvidenceQuoteRaw): EvidenceQuote | null => {
  if (typeof item === 'string') {
    return {
      quote: item,
      url: '',
      id: `quote-${item.slice(0, 16)}`,
    }
  }
  if (!item || typeof item.quote !== 'string') return null
  return {
    quote: item.quote,
    url: item.url ?? '',
    domain: item.domain,
    title: item.title,
    credibility_score: item.credibility_score,
    published_date: item.published_date,
    id: item.id ?? `quote-${item.quote.slice(0, 16)}`,
  }
}

export const useResearchData = (results: ResearchResult): ResearchDisplayData => {
  const integrated = results.integrated_synthesis as unknown as IntegratedSynthesis | undefined

  const baseAnswer = useMemo<PrimaryAnswer | null>(() => {
    if (integrated?.primary_answer) {
      return integrated.primary_answer
    }
    if (results.answer) {
      const ans = results.answer
      return {
        citations: ans.citations as unknown as AnswerCitation[],
        sections: ans.sections,
        action_items: ans.action_items,
        summary: ans.summary,
        metadata: ans.metadata as AnswerMetadataType,
      }
    }
    return null
  }, [integrated?.primary_answer, results.answer])

  const metadata = useMemo<AnswerMetadataType>(() => {
    const raw = results.metadata
    if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
      return raw as AnswerMetadataType
    }
    return {} as AnswerMetadataType
  }, [results.metadata])

  const contextLayers = useMemo<ContextLayersInfo | null>(() => {
    const legacy = results.paradigm_analysis?.context_engineering as ContextLayersInfo | undefined
    const enriched = metadata?.context_layers as ContextLayersInfo | undefined
    if (!legacy && !enriched) return null
    return { ...(legacy || {}), ...(enriched || {}) }
  }, [metadata?.context_layers, results.paradigm_analysis?.context_engineering])

  const sources = useMemo<ResearchSource[]>(() => {
    if (Array.isArray(results.sources)) {
      return results.sources as ResearchSource[]
    }
    return []
  }, [results.sources])

  const domainInfo = useMemo(() => {
    const map: Record<string, { category?: string; explanation?: string; score?: number }> = {}
    for (const source of sources) {
      const rawDomain = (source.domain || '').toLowerCase()
      const url = source.url ?? ''
      let domain = rawDomain
      if (!domain && url) {
        try {
          const parsed = new URL(url)
          domain = parsed.hostname.replace(/^www\./, '').toLowerCase()
        } catch {
          domain = url.replace(/^www\./, '').toLowerCase()
        }
      }
      if (!domain || map[domain]) continue
      map[domain] = {
        category: source.source_category,
        explanation: source.credibility_explanation,
        score: typeof source.credibility_score === 'number' ? source.credibility_score : undefined,
      }
    }
    return map
  }, [sources])

  const evidenceSnapshot = useMemo(() => {
    const total = sources.length
    let strong = 0
    let moderate = 0
    let weak = 0
    let minDate: number | null = null
    let maxDate: number | null = null
    for (const source of sources) {
      const score = typeof source.credibility_score === 'number' ? source.credibility_score : undefined
      if (typeof score === 'number') {
        const band = getCredibilityBand(score)
        if (band === 'high') strong += 1
        else if (band === 'medium') moderate += 1
        else weak += 1
      }
      if (source.published_date) {
        const epoch = Date.parse(source.published_date)
        if (!Number.isNaN(epoch)) {
          if (minDate === null || epoch < minDate) minDate = epoch
          if (maxDate === null || epoch > maxDate) maxDate = epoch
        }
      }
    }
    let window: string | undefined
    if (minDate && maxDate) {
      window = `${new Date(minDate).toLocaleDateString()}-${new Date(maxDate).toLocaleDateString()}`
    }
    return { total, strong, moderate, weak, window }
  }, [sources])

  const evidenceQuotes = useMemo<EvidenceQuote[]>(() => {
    const fromAnswer = baseAnswer?.metadata?.evidence_quotes
    const fromMetadata = metadata?.evidence_quotes
    const source = Array.isArray(fromAnswer) ? fromAnswer : Array.isArray(fromMetadata) ? fromMetadata : []
    return source
      .map(normaliseEvidenceQuote)
      .filter((quote): quote is EvidenceQuote => Boolean(quote))
  }, [baseAnswer?.metadata?.evidence_quotes, metadata?.evidence_quotes])

  const categoryDistribution = useMemo(
    () => (metadata?.category_distribution || {}) as Record<string, number>,
    [metadata?.category_distribution]
  )
  const biasDistribution = useMemo(
    () => (metadata?.bias_distribution || {}) as Record<string, number>,
    [metadata?.bias_distribution]
  )
  const credibilityDistribution = useMemo(
    () =>
      (metadata?.credibility_summary?.score_distribution || {
        high: 0,
        medium: 0,
        low: 0,
      }) as Record<'high' | 'medium' | 'low', number>,
    [metadata?.credibility_summary?.score_distribution]
  )

  const actionableRatio = Number(metadata?.actionable_content_ratio ?? 0)

  const analysisMetrics = useMemo<AnalysisMetrics | null>(() => {
    const raw = metadata?.analysis_metrics as Record<string, unknown> | undefined
    if (!raw || typeof raw !== 'object') return null

    const asFinite = (value: unknown): number | undefined => {
      if (typeof value === 'number' && Number.isFinite(value)) return value
      if (typeof value === 'string' && value.trim()) {
        const parsed = Number(value)
        if (Number.isFinite(parsed)) return parsed
      }
      return undefined
    }

    const metrics: AnalysisMetrics = {
      durationMs: asFinite(raw.duration_ms),
      sourcesTotal: asFinite(raw.sources_total),
      sourcesCompleted: asFinite(raw.sources_completed),
      progressUpdates: asFinite(raw.progress_updates),
      updatesPerSecond: asFinite(raw.updates_per_second),
      avgUpdateGapMs: asFinite(raw.avg_update_gap_ms),
      p95UpdateGapMs: asFinite(raw.p95_update_gap_ms),
      firstUpdateGapMs: asFinite(raw.first_update_gap_ms),
      lastUpdateGapMs: asFinite(raw.last_update_gap_ms),
      cancelled: Boolean(raw.cancelled),
    }

    const hasSignal = (
      (typeof metrics.durationMs === 'number' && metrics.durationMs > 0) ||
      (typeof metrics.progressUpdates === 'number' && metrics.progressUpdates > 0) ||
      (typeof metrics.sourcesTotal === 'number' && metrics.sourcesTotal > 0) ||
      (typeof metrics.sourcesCompleted === 'number' && metrics.sourcesCompleted > 0) ||
      (typeof metrics.updatesPerSecond === 'number' && metrics.updatesPerSecond > 0)
    )

    if (!hasSignal && !metrics.cancelled) {
      return null
    }

    return metrics
  }, [metadata?.analysis_metrics])

  const summary = baseAnswer?.summary || results.answer?.summary || ''
  const citations = (baseAnswer?.citations || []) as AnswerCitation[]
  const actionItems = baseAnswer?.action_items || []

  const confidenceInfo = useMemo<ConfidenceInfo>(() => {
    const rawConfidence =
      integrated?.confidence_score ?? results.paradigm_analysis?.primary?.confidence ?? 0
    const percentage = Math.max(0, Math.min(100, rawConfidence * 100))
    let band: ConfidenceInfo['band'] = 'Low'
    if (percentage >= 80) band = 'High'
    else if (percentage >= 60) band = 'Medium'

    const highRatio = metadata?.credibility_summary?.high_credibility_ratio
    const highCount = metadata?.credibility_summary?.high_credibility_count
    const becauseParts: string[] = []
    if (typeof highRatio === 'number') {
      becauseParts.push(`${Math.round(highRatio * 100)}% high-credibility sources`)
    } else if (typeof highCount === 'number' && sources.length) {
      becauseParts.push(`${Math.round((highCount / sources.length) * 100)}% high-credibility sources`)
    }
    const uniqueCategories = Object.keys(categoryDistribution || {}).length
    if (uniqueCategories >= 3) {
      becauseParts.push(`${uniqueCategories} source types`)
    }
    if (actionableRatio >= 0.85) {
      becauseParts.push('high actionable content')
    }

    return {
      percentage,
      band,
      because: becauseParts.join('; '),
    }
  }, [integrated?.confidence_score, results.paradigm_analysis?.primary?.confidence, metadata?.credibility_summary, sources.length, categoryDistribution, actionableRatio])

  const warnings = metadata?.warnings as unknown[] | undefined
  const degraded = Boolean(metadata?.degraded)

  return {
    results,
    baseAnswer,
    metadata,
    contextLayers,
    summary,
    citations,
    actionItems,
    evidenceQuotes,
    sources,
    domainInfo,
    evidenceSnapshot,
    categoryDistribution,
    biasDistribution,
    credibilityDistribution,
    actionableRatio,
    analysisMetrics,
    biasCheck: metadata?.bias_check,
    confidenceInfo,
    integratedSynthesis: integrated,
    meshSynthesis: results.mesh_synthesis,
    fetchedAt: new Date().toISOString(),
    warnings,
    degraded,
    status: results.status,
  }
}
