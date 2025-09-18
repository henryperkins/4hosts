import React, { useState, useRef, useEffect, useMemo } from 'react'
import { FiDownload, FiExternalLink, FiShield, FiAlertTriangle, FiChevronDown, FiChevronUp, FiZap, FiGitMerge, FiLoader, FiCheckCircle, FiAlertCircle, FiClock, FiFilter } from 'react-icons/fi'
import toast from 'react-hot-toast'
import api from '../services/api'
import type { ResearchResult, AnswerSection } from '../types'
import { getParadigmClass, getParadigmDescription } from '../constants/paradigm'
import { ContextMetricsPanel } from './ContextMetricsPanel'
import { EvidencePanel } from './EvidencePanel'
import { AnswerFeedback } from './feedback/AnswerFeedback'
import { Button } from './ui/Button'
import { getCredibilityBand, getCredibilityLabel, getCredibilityColor } from '../utils/credibility'

interface ResultsDisplayEnhancedProps {
  results: ResearchResult
}

type ContextIsolationDetails = {
  focus_areas?: string[]
  patterns?: number
}

type ContextLayersInfo = {
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

type ActionItem = {
  action?: string
  timeframe?: string
  priority?: 'high' | 'medium' | 'low' | string
  paradigm?: string
  owner?: string
  due_date?: string
}

type AnswerCitation = {
  id?: string
  title?: string
  source?: string
  url?: string
  credibility_score?: number
  paradigm_alignment?: string
}

type EvidenceQuoteRaw =
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

type EvidenceQuote = {
  quote: string
  url: string
  domain?: string
  title?: string
  credibility_score?: number
  published_date?: string
  id: string
}

type AnswerMetadataType = { evidence_quotes?: EvidenceQuoteRaw[]; statistical_insights?: number } & Record<string, unknown>

type PrimaryAnswer = {
  citations?: AnswerCitation[]
  sections?: AnswerSection[]
  action_items?: ActionItem[]
  summary?: string
  metadata?: AnswerMetadataType
}

type SecondaryPerspective = {
  title?: string
  content?: string
  paradigm?: string
  sources_count?: number
  confidence?: number
}

type ConflictItem = { description?: string }

type IntegratedSynthesis = {
  primary_answer?: PrimaryAnswer
  secondary_perspective?: SecondaryPerspective
  integrated_summary?: string
  synergies?: string[]
  conflicts_identified?: ConflictItem[]
  confidence_score?: number
}

type ResearchSource = {
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

type AgentTraceEntry = {
  step?: string
  iteration?: number
  coverage?: number
  proposed_queries?: string[]
  warnings?: string[]
}

export const ResultsDisplayEnhanced: React.FC<ResultsDisplayEnhancedProps> = ({ results }) => {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set([0]))
  const [isExporting, setIsExporting] = useState(false)
  const [exportFormat, setExportFormat] = useState<string | null>(null)
  const [showAllCitations, setShowAllCitations] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const [traceOpen, setTraceOpen] = useState(false)
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set(['all']))
  const [selectedCredBands, setSelectedCredBands] = useState<Set<'high' | 'medium' | 'low'>>(() => new Set(['high', 'medium', 'low']))
  const [sourcesPageSize, setSourcesPageSize] = useState<number>(20)
  const [sourcesPage, setSourcesPage] = useState<number>(1)
  const fetchedAtRef = useRef<string>(new Date().toISOString())

  const integrated_synthesis = results.integrated_synthesis as unknown as IntegratedSynthesis | undefined
  const baseAnswer = (integrated_synthesis?.primary_answer ??
    (results as unknown as { answer?: PrimaryAnswer }).answer) as PrimaryAnswer | undefined

  const metadata = results.metadata ?? {}

  const contextLayers = useMemo<ContextLayersInfo | null>(() => {
    const legacy = results.paradigm_analysis?.context_engineering as ContextLayersInfo | undefined
    const enriched = metadata?.context_layers as ContextLayersInfo | undefined
    if (!legacy && !enriched) return null
    return { ...legacy, ...enriched }
  }, [metadata?.context_layers, results.paradigm_analysis?.context_engineering])

  const actionableRatio = Number(metadata?.actionable_content_ratio ?? 0)
  const bias = metadata?.bias_check
  const categoryDistribution = (metadata?.category_distribution || {}) as Record<string, number>
  const biasDistribution = (metadata?.bias_distribution || {}) as Record<string, number>
  const credibilityDistribution = (metadata?.credibility_summary?.score_distribution || {}) as Record<'high' | 'medium' | 'low', number>

  const toBase64Safe = (s: string): string | null => {
    try {
      if (typeof window !== 'undefined' && typeof window.btoa === 'function') {
        return window.btoa(
          encodeURIComponent(s).replace(/%([0-9A-F]{2})/g, (_, p1) =>
            String.fromCharCode(parseInt(p1, 16))
          )
        )
      }
    } catch {
      // ignore
    }
    return null
  }

  const getDomain = (input?: string): string => {
    if (!input) return ''
    try {
      const url = new URL(input)
      return url.hostname.replace(/^www\./, '').toLowerCase()
    } catch {
      return input.replace(/^www\./, '').toLowerCase()
    }
  }

  const sourcesList = (Array.isArray(results.sources) ? results.sources : []) as ResearchSource[]

  const domainInfo = useMemo(() => {
    const map: Record<string, { category?: string; explanation?: string; score?: number }> = {}
    for (const s of sourcesList) {
      const d = getDomain(s.url) || (s.domain || '').toLowerCase()
      if (d && !map[d]) {
        map[d] = {
          category: s.source_category,
          explanation: s.credibility_explanation,
          score: typeof s.credibility_score === 'number' ? s.credibility_score : undefined,
        }
      }
    }
    return map
  }, [sourcesList])

  const evidenceSnapshot = useMemo(() => {
    const total = sourcesList.length
    let strong = 0, moderate = 0, weak = 0
    let minDate: number | null = null
    let maxDate: number | null = null
    for (const s of sourcesList) {
      const sc = typeof s.credibility_score === 'number' ? s.credibility_score : undefined
      if (typeof sc === 'number') {
        const band = getCredibilityBand(sc)
        if (band === 'high') strong++
        else if (band === 'medium') moderate++
        else weak++
      }
      if (s.published_date) {
        const t = Date.parse(s.published_date)
        if (!Number.isNaN(t)) {
          if (minDate === null || t < minDate) minDate = t
          if (maxDate === null || t > maxDate) maxDate = t
        }
      }
    }
    return {
      total,
      strong,
      moderate,
      weak,
      window: minDate && maxDate ? `${new Date(minDate).toLocaleDateString()}–${new Date(maxDate).toLocaleDateString()}` : undefined,
    }
  }, [sourcesList])

  const qualityLabel = getCredibilityLabel

  const bottomLine = (text: string, maxWords = 20): string => {
    if (!text) return 'No summary available.'
    const firstSentence = text.split(/(?<=[.!?])\s+/)[0] || text
    const words = firstSentence.trim().split(/\s+/)
    if (words.length <= maxWords) return firstSentence.trim()
    return words.slice(0, maxWords).join(' ') + '…'
  }

  const confidenceInfo = useMemo(() => {
    const rawConf = (integrated_synthesis?.confidence_score ?? results.paradigm_analysis?.primary?.confidence ?? 0)
    const conf = Math.max(0, Math.min(100, rawConf * 100))
    const summary = metadata?.credibility_summary
    const highShare =
      summary?.high_credibility_ratio ??
      (summary?.high_credibility_count && sourcesList.length
        ? summary.high_credibility_count / Math.max(1, sourcesList.length)
        : undefined)
    const uniqueCats = Object.keys(categoryDistribution || {}).length
    let band: 'High' | 'Medium' | 'Low' = 'Low'
    if (conf >= 80) band = 'High'
    else if (conf >= 60) band = 'Medium'
    const because: string[] = []
    if (typeof highShare === 'number') because.push(`${Math.round(highShare * 100)}% high‑credibility sources`)
    if (uniqueCats >= 3) because.push(`${uniqueCats} source types`)
    if ((metadata?.actionable_content_ratio || 0) >= 0.85) because.push('high actionable content')
    return { conf, band, because: because.join('; ') }
  }, [integrated_synthesis?.confidence_score, results.paradigm_analysis?.primary?.confidence, metadata?.credibility_summary, metadata?.actionable_content_ratio, categoryDistribution, sourcesList.length])

  const parseExplanation = (expl?: string): { bias?: string; fact?: string; cat?: string } => {
    if (!expl) return {}
    const out: { bias?: string; fact?: string; cat?: string } = {}
    const pairs = expl.split(',')
    for (const p of pairs) {
      const [k, v] = p.split('=').map(s => (s || '').trim())
      if (!k || !v) continue
      if (k.startsWith('bias')) out.bias = v
      if (k.startsWith('fact') || k.startsWith('factual')) out.fact = v
      if (k.startsWith('cat')) out.cat = v
    }
    return out
  }

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false)
      }
    }
    if (dropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [dropdownOpen])

  useEffect(() => {
    setSourcesPage(1)
  }, [selectedCategories, selectedCredBands, results.research_id])

  // Prepare evidence quotes before any early return (avoid conditional hooks later)
  const evidenceQuotes: EvidenceQuote[] = useMemo(() => {
    const answerMetadata = (baseAnswer?.metadata as AnswerMetadataType | undefined)
    const answerEvidence = Array.isArray(answerMetadata?.evidence_quotes) ? answerMetadata?.evidence_quotes : undefined
    const metadataEvidence = Array.isArray(metadata?.evidence_quotes) ? (metadata.evidence_quotes as EvidenceQuoteRaw[]) : undefined
    const raw = answerEvidence ?? metadataEvidence ?? []
    const out: EvidenceQuote[] = []
    raw.forEach((q, index) => {
      if (typeof q === 'string') {
        out.push({ quote: q, url: '', id: `quote-${index}` })
      } else if (q && typeof q === 'object' && 'quote' in q) {
        const o = q as Exclude<EvidenceQuoteRaw, string>
        const basis = (o.quote || '').slice(0, 20) + (o.url || '')
        const base = toBase64Safe(basis) ?? (globalThis.crypto && 'randomUUID' in globalThis.crypto ? globalThis.crypto.randomUUID() : Math.random().toString(36).slice(2))
        const stableId = o.id || `quote-${String(base).replace(/[^a-zA-Z0-9]/g, '').slice(0, 10)}-${index}`
        out.push({ ...o, url: o.url ?? '', id: stableId })
      }
    })
    return out
  }, [baseAnswer?.metadata, metadata?.evidence_quotes])

  // Coerce secondary perspective to a section (hook must be unconditional)
  const secondaryAsSection: AnswerSection | null = useMemo(() => {
    const sp = integrated_synthesis?.secondary_perspective
    if (!sp) return null
    const title = sp.title || 'Secondary perspective'
    const content = sp.content || ''
    const paradigm = sp.paradigm || 'dolores'
    const sources_count = typeof sp.sources_count === 'number' ? sp.sources_count : 0
    const confidence = typeof sp.confidence === 'number' ? sp.confidence : 0.6
    return { title, content, paradigm, sources_count, confidence } as AnswerSection
  }, [integrated_synthesis?.secondary_perspective])

  const allSections: AnswerSection[] = useMemo(() => {
    const baseSections = Array.isArray(baseAnswer?.sections) ? baseAnswer!.sections! : []
    return secondaryAsSection ? [...baseSections, secondaryAsSection] : baseSections
  }, [baseAnswer?.sections, secondaryAsSection])

  // Sources filtering/sorting/pagination
  const {
    viewSources,
    totalSourcesFiltered,
    pageCount,
    currentPage,
    pageStart,
    pageEnd,
  } = useMemo(() => {
    const byCat = selectedCategories.has('all')
      ? sourcesList
      : sourcesList.filter(s => selectedCategories.has((s.source_category || 'general')))
    const filtered = byCat.filter(s => selectedCredBands.has(getCredibilityBand(Number(s.credibility_score || 0))))
    const sorted = [...filtered].sort((a, b) => {
      const as = Number(a.credibility_score || 0)
      const bs = Number(b.credibility_score || 0)
      if (bs !== as) return bs - as
      const ad = a.published_date ? Date.parse(a.published_date) : 0
      const bd = b.published_date ? Date.parse(b.published_date) : 0
      if (bd !== ad) return bd - ad
      const at = (a.title || '').localeCompare(b.title || '')
      if (at !== 0) return at
      const adom = (a.domain || '').localeCompare(b.domain || '')
      if (adom !== 0) return adom
      return (a.url || '').localeCompare(b.url || '')
    })
    const total = sorted.length
    const pageSize = Math.max(1, Math.min(Number(sourcesPageSize || 20), total || 20))
    const pages = Math.max(1, Math.ceil(total / pageSize))
    const current = Math.min(sourcesPage, pages)
    const start = total === 0 ? 0 : (current - 1) * pageSize
    const end = total === 0 ? 0 : Math.min(start + pageSize, total)
    const view = sorted.slice(start, end)
    return {
      viewSources: view,
      totalSourcesFiltered: total,
      pageCount: pages,
      currentPage: current,
      pageStart: start,
      pageEnd: end,
    }
  }, [sourcesList, selectedCategories, selectedCredBands, sourcesPage, sourcesPageSize])

  // Early return now safe (no hooks after this point)
  if (!baseAnswer) {
    const noSearchResults =
      (metadata?.total_sources_analyzed === 0) ||
      (metadata as any)?.evidence_builder_skipped === true // If you have this in the type, remove "as any".
    const belowThresholdResults = sourcesList.some(s => s.raw_data?.below_relevance_threshold === true)
    const isProcessing = results.status === 'processing' || results.status === 'pending'

    return (
      <div className="mt-8 animate-fade-in">
        <div className="bg-surface rounded-lg shadow-lg p-8 text-center transition-colors duration-200 border border-border">
          <FiAlertCircle className="h-16 w-16 text-error mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-text mb-2">
            {isProcessing ? 'Research Still Processing' : 'Research Incomplete'}
          </h3>
          <p className="text-text-muted mb-4">
            {noSearchResults
              ? 'No relevant search results were found for your query. Try rephrasing or broadening your search terms.'
              : isProcessing
                ? 'The research is still being processed. Please wait a moment...'
                : 'This research could not be completed due to an error during processing.'}
          </p>
          {belowThresholdResults && (
            <div className="bg-warning/10 border border-warning/30 rounded-lg p-4 mb-4">
              <p className="text-sm text-warning">
                Note: Some results were below our quality threshold but were included to provide context.
              </p>
            </div>
          )}
          <div className="bg-error/10 border border-error/30 rounded-lg p-4">
            <p className="text-sm text-error">
              Status: {results.status || 'Unknown'}
            </p>
            {metadata && (
              <>
                <p className="text-sm text-error mt-1">
                  Research ID: {results.research_id}
                </p>
                {metadata.total_sources_analyzed !== undefined && (
                  <p className="text-sm text-error mt-1">
                    Sources analyzed: {metadata.total_sources_analyzed}
                  </p>
                )}
                {(metadata as any).error_message && (
                  <p className="text-sm text-error mt-1">
                    Error: {(metadata as any).error_message}
                  </p>
                )}
              </>
            )}
          </div>
          {!isProcessing && (
            <div className="mt-6">
              <Button
                variant="primary"
                onClick={() => window.location.reload()}
                className="mx-auto"
              >
                Try Another Search
              </Button>
            </div>
          )}
        </div>
      </div>
    )
  }

  const toggleSection = (index: number) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  const handleExport = async (format: string) => {
    setIsExporting(true)
    setExportFormat(format)
    setDropdownOpen(false)

    try {
      const exportFormats = (results as unknown as { export_formats?: Record<string, string> }).export_formats
      const presigned = exportFormats?.[format]
      if (typeof presigned === 'string' && presigned) {
        window.open(presigned, '_blank', 'noopener')
      } else {
        const blob = await api.exportResearch(results.research_id, format as 'pdf' | 'json' | 'csv' | 'markdown' | 'excel')
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `research-${results.research_id}.${format}`
        link.click()
        URL.revokeObjectURL(url)
      }

      toast.success(
        <div className="flex items-center gap-2">
          <FiCheckCircle className="h-4 w-4" />
          <span>Exported as {format.toUpperCase()}</span>
        </div>
      )
    } catch {
      toast.error(
        <div className="flex items-center gap-2">
          <FiAlertCircle className="h-4 w-4" />
          <span>Export failed</span>
        </div>
      )
    } finally {
      setIsExporting(false)
      setExportFormat(null)
    }
  }

  const getCredibilityIcon = (score: number) => {
    const band = getCredibilityBand(score)
    if (band === 'high') return <FiShield className="h-4 w-4" aria-label="High credibility" />
    if (band === 'medium') return <FiAlertTriangle className="h-4 w-4" aria-label="Medium credibility" />
    return <FiAlertCircle className="h-4 w-4" aria-label="Low credibility" />
  }

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'high':
        return <FiAlertCircle className="h-4 w-4 text-red-500 dark:text-red-400" aria-label="High priority" />
      case 'medium':
        return <FiClock className="h-4 w-4 text-yellow-500 dark:text-yellow-400" aria-label="Medium priority" />
      default:
        return <FiCheckCircle className="h-4 w-4 text-green-500 dark:text-green-400" aria-label="Low priority" />
    }
  }

  const citations = Array.isArray(baseAnswer.citations) ? baseAnswer.citations : []
  const actionItems = Array.isArray(baseAnswer.action_items) ? baseAnswer.action_items : []
  const summary = baseAnswer.summary || 'No summary available'

  const displayedCitations = showAllCitations ? citations : citations.slice(0, 5)

  return (
    <div className="mt-8 space-y-6 animate-fade-in">
      <div className="bg-surface rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up border border-border">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-text">Research results</h2>
            <p className="text-sm text-text-muted mt-1">Query: "{results.query}"</p>
            <p className="text-xs text-text-subtle">As of {new Date(fetchedAtRef.current).toLocaleString()}</p>
          </div>

          <div className="flex items-center gap-2">
            {(() => {
              const pri = results.paradigm_analysis?.primary?.paradigm || (metadata as any)?.paradigm || 'bernard'
              return (
                <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getParadigmClass(pri || 'bernard')}`}>
                  {getParadigmDescription(pri || 'bernard')}
                </span>
              )
            })()}
            {(() => {
              const sec = results.paradigm_analysis?.secondary?.paradigm
              return sec ? (
                <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getParadigmClass(sec || 'bernard')}`}>
                  + {getParadigmDescription(sec || 'bernard')}
                </span>
              ) : null
            })()}

            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                disabled={isExporting}
                className="p-2 text-text-muted hover:text-text hover:bg-surface-subtle rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Export results"
                aria-expanded={dropdownOpen}
                aria-haspopup="true"
              >
                {isExporting ? (
                  <FiLoader className="h-5 w-5 animate-spin" />
                ) : (
                  <FiDownload className="h-5 w-5" />
                )}
              </button>

              <div
                className={`absolute right-0 mt-2 w-48 bg-surface rounded-lg shadow-xl border border-border transition-all duration-200 z-10 ${dropdownOpen ? 'opacity-100 visible translate-y-0' : 'opacity-0 invisible -translate-y-2'
                  }`}
                role="menu"
              >
                {
                  (() => {
                    const allowed = ['json', 'csv', 'pdf', 'markdown', 'excel'] as const
                    const exportFormats = (results as unknown as { export_formats?: Record<string, string> }).export_formats || {}
                    const map: Record<string, string> = { ...exportFormats }
                    if (Object.keys(map).length === 0) {
                      allowed.forEach((f) => {
                        map[f] = `/v1/research/${results.research_id}/export/${f}`
                      })
                    }
                    return allowed.filter((f) => f in map).map((fmt) => (
                      <Button
                        key={fmt}
                        variant="ghost"
                        size="sm"
                        className="w-full justify-start text-left capitalize"
                        onClick={() => handleExport(fmt)}
                        disabled={isExporting}
                        role="menuitem"
                      >
                        {exportFormat === fmt && isExporting && (
                          <FiLoader className="h-4 w-4 animate-spin" />
                        )}
                        Export as {fmt.toUpperCase()}
                      </Button>
                    ))
                  })()
                }
              </div>
            </div>
          </div>
        </div>

        <div className="mt-3">
          <p className="text-text text-base"><span className="font-semibold">Bottom line:</span> {bottomLine(summary)}</p>
          <div className="mt-2 text-sm text-text">
            <span className="font-medium">Evidence:</span> {evidenceSnapshot.total.toLocaleString()} sources ({evidenceSnapshot.strong} strong, {evidenceSnapshot.moderate} moderate, {evidenceSnapshot.weak} weak){evidenceSnapshot.window ? ` · timeframe ${evidenceSnapshot.window}` : ''}.
          </div>
          <div className="mt-1 text-sm text-text">
            <span className="font-medium">Confidence:</span> {confidenceInfo.band} ({Math.round(confidenceInfo.conf)}%){confidenceInfo.because ? ` — because ${confidenceInfo.because}.` : '.'}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-surface-subtle p-3 rounded-lg border border-border">
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-muted">Actionable Content</span>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${actionableRatio >= 0.85 ? 'bg-green-600 text-white' : 'bg-amber-500 text-white'}`}>
                {(actionableRatio * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-text-subtle mt-1">Estimated share of concrete actions and key insights.</p>
          </div>
          <div className="bg-surface-subtle p-3 rounded-lg border border-border">
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-muted">Bias Check</span>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${bias?.balanced ? 'bg-green-600 text-white' : 'bg-amber-500 text-white'}`}>
                {bias?.balanced ? 'Balanced' : 'Needs Balance'}
              </span>
            </div>
            <p className="text-xs text-text-subtle mt-1">
              Domain diversity {(bias?.domain_diversity ? (bias.domain_diversity * 100).toFixed(0) : '0')}%
              {bias?.dominant_domain !== undefined && bias?.dominant_share !== undefined
                ? `, dominant: ${bias.dominant_domain} ${(bias.dominant_share * 100).toFixed(0)}%`
                : ''}
            </p>
          </div>
          <div className="bg-surface-subtle p-3 rounded-lg border border-border">
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-muted">Paradigm Fit</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-600 text-white">
                {((metadata?.paradigm_fit?.confidence || 0) * 100).toFixed(0)}% conf · margin {((metadata?.paradigm_fit?.margin || 0) * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-text-subtle mt-1">Primary: {metadata?.paradigm_fit?.primary || '-'}</p>
          </div>
        </div>

        <div className="mt-3 text-sm text-text">
          <span className="font-medium">Analyzed</span> {Number(metadata?.total_sources_analyzed ?? 0)} sources
          <span> → </span>
          <span className="font-medium">{Number(metadata?.high_quality_sources ?? 0)}</span> high‑quality
          {metadata?.credibility_summary?.average_score !== undefined && (
            <span className="ml-2 text-xs text-text-subtle">avg credibility {(Number(metadata.credibility_summary.average_score) * 100).toFixed(0)}%</span>
          )}
        </div>

        <div className="mt-3 p-3 rounded-lg bg-surface-subtle border border-border">
          <p className="text-xs text-text-muted">
            <span className="font-semibold">Scales:</span> Confidence — High ≥ 80%, Medium 60–79%, Low &lt; 60%. Quality — Strong ≥ 0.80, Moderate 0.60–0.79, Weak &lt; 0.60.
          </p>
        </div>

        <div className="prose dark:prose-invert max-w-none mt-3">
          <p className="text-text">{integrated_synthesis?.integrated_summary || summary}</p>
        </div>

        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="bg-surface-subtle rounded-lg p-3 transition-colors duration-200">
            <p className="text-text-muted">Sources Analyzed</p>
            <p className="font-semibold text-lg text-text">{Number(metadata?.total_sources_analyzed ?? 0)}</p>
          </div>
          <div className="bg-surface-subtle rounded-lg p-3 transition-colors duration-200">
            <p className="text-text-muted">High Quality</p>
            <p className="font-semibold text-lg text-text">{Number(metadata?.high_quality_sources ?? 0)}</p>
          </div>
          <div className="bg-surface-subtle rounded-lg p-3 transition-colors duration-200">
            <p className="text-text-muted">Processing Time</p>
            <p className="font-semibold text-lg text-text">{Number(metadata?.processing_time_seconds ?? 0)}s</p>
          </div>
          <div className="bg-surface-subtle rounded-lg p-3 transition-colors duration-200">
            <p className="text-text-muted">Paradigms Used</p>
            <p className="font-semibold text-lg text-text">{Array.isArray(metadata.paradigms_used) ? metadata.paradigms_used.length : 0}</p>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="bg-surface-subtle rounded-lg p-3">
            <p className="text-text mb-2">Source categories</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(categoryDistribution).map(([cat, count]) => (
                <span key={cat} className="px-2 py-1 rounded bg-surface border border-border text-xs text-text">
                  {cat}: <span className="font-semibold">{String(count)}</span>
                </span>
              ))}
              {Object.keys(categoryDistribution).length === 0 && (
                <span className="text-text-subtle">No category data</span>
              )}
            </div>
          </div>
          <div className="bg-surface-subtle rounded-lg p-3">
            <p className="text-text mb-2">Credibility Distribution</p>
            {(() => {
              const totals = Object.values(credibilityDistribution).map((value) => Number(value || 0))
              const sum = totals.reduce((acc, cur) => acc + cur, 0) || 1
              const pct = (key: keyof typeof credibilityDistribution) => Math.round(((credibilityDistribution[key] ?? 0) / sum) * 100)
              return (
                <div>
                  <div className="h-2 w-full rounded bg-surface-muted overflow-hidden flex">
                    <div className="h-2 bg-green-600" style={{ width: `${pct('high') || 0}%` }} />
                    <div className="h-2 bg-yellow-500" style={{ width: `${pct('medium') || 0}%` }} />
                    <div className="h-2 bg-red-500" style={{ width: `${pct('low') || 0}%` }} />
                  </div>
                  <div className="mt-2 flex gap-2 text-xs text-text">
                    <span className="px-2 py-0.5 rounded bg-green-600 text-white">High {pct('high') || 0}%</span>
                    <span className="px-2 py-0.5 rounded bg-yellow-500 text-white">Medium {pct('medium') || 0}%</span>
                    <span className="px-2 py-0.5 rounded bg-red-500 text-white">Low {pct('low') || 0}%</span>
                  </div>
                </div>
              )
            })()}
          </div>
        </div>

        <div className="mt-4 bg-surface-subtle rounded-lg p-3">
          <p className="text-text mb-2">Bias Distribution</p>
          <div className="flex flex-wrap gap-2 text-xs">
            {Object.entries(biasDistribution).map(([k, v]) => (
              <span key={k} className="px-2 py-0.5 rounded bg-surface border border-border text-text">
                {k}: <span className="font-semibold">{String(v)}</span>
              </span>
            ))}
            {Object.keys(biasDistribution).length === 0 && (
              <span className="text-text-subtle">No bias data</span>
            )}
          </div>
        </div>

        {['deep', 'deep_research'].includes(String(metadata?.research_depth || '')) && (
          <div className="mt-4 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-lg border border-purple-200 dark:border-purple-800 transition-colors duration-200">
            <div className="flex items-center gap-2">
              <FiZap className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              <h4 className="text-sm font-semibold text-purple-900 dark:text-purple-100">Advanced Research Depth</h4>
              <span className="ml-auto text-xs font-bold bg-purple-600 text-white px-2 py-1 rounded">
                {String(metadata.research_depth)}
              </span>
            </div>
            <p className="text-sm text-purple-700 dark:text-purple-300 mt-2">
              This result was generated using an advanced depth setting with broader search and synthesis.
            </p>
          </div>
        )}

        <div className="mt-4 p-4 bg-surface-subtle rounded-lg border border-border transition-colors duration-200">
          <h4 className="text-sm font-semibold text-text mb-2">Executive Summary</h4>
          <ul className="list-disc list-inside text-sm text-text space-y-1">
            {(integrated_synthesis?.primary_answer?.action_items || (results as unknown as { answer?: PrimaryAnswer }).answer?.action_items || []).slice(0, 3).map((a: ActionItem, i: number) => (
              <li key={i}>{a.action || ''}{a.timeframe ? ` (${a.timeframe})` : ''}</li>
            ))}
            {(integrated_synthesis?.primary_answer?.action_items || (results as unknown as { answer?: PrimaryAnswer }).answer?.action_items || []).length === 0 && (
              <li>No immediate actions extracted.</li>
            )}
          </ul>
        </div>

        {integrated_synthesis && (
          <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800 transition-colors duration-200">
            <h4 className="text-sm font-semibold text-green-900 dark:text-green-100 mb-3">Strategic Framework</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-surface rounded-lg border border-border p-4">
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-sm font-semibold text-text">Immediate Opportunities (Maeve)</h5>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${getParadigmClass('maeve')}`}>Strategic</span>
                </div>
                {Array.isArray(integrated_synthesis.primary_answer?.action_items) && integrated_synthesis.primary_answer.action_items.length > 0 ? (
                  <ul className="list-disc list-inside text-sm text-text space-y-1">
                    {integrated_synthesis.primary_answer.action_items.map((it: ActionItem, idx: number) => (
                      <li key={idx}><span className="font-medium capitalize">{it.priority}</span>: {it.action} {it.timeframe ? (<em className="text-xs text-text-subtle">({it.timeframe})</em>) : null}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-text">No immediate actions extracted.</p>
                )}
              </div>

              <div className="bg-surface rounded-lg border border-border p-4">
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-sm font-semibold text-text">Systemic Context (Dolores)</h5>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${getParadigmClass('dolores')}`}>Revolutionary</span>
                </div>
                <div className="text-sm text-text whitespace-pre-wrap">
                  {integrated_synthesis.secondary_perspective?.content || '—'}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800 transition-colors duration-200">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-semibold text-yellow-900 dark:text-yellow-100">Action Items</h4>
            <span className="text-xs text-yellow-800 dark:text-yellow-200">{actionItems.length} found</span>
          </div>
          {actionItems.length > 0 ? (
            <ul className="divide-y divide-yellow-200 dark:divide-yellow-800">
              {actionItems.map((it: ActionItem, idx: number) => (
                <li key={idx} className="py-2 flex items-start gap-3">
                  <div className="mt-0.5" aria-hidden>
                    {getPriorityIcon(String(it.priority || 'low'))}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm text-text">
                      {it.action || '—'}
                    </div>
                    <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-text-muted">
                      {it.timeframe && (
                        <span className="px-2 py-0.5 rounded bg-surface border border-border">{it.timeframe}</span>
                      )}
                      {it.paradigm && (
                        <span className={`px-2 py-0.5 rounded border ${getParadigmClass(it.paradigm || 'bernard')}`}>{String(it.paradigm).charAt(0).toUpperCase() + String(it.paradigm).slice(1)}</span>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-text">No action items identified.</p>
          )}
        </div>

        {contextLayers ? (
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 transition-colors duration-200">
            <h4 className="text-sm font-semibold text-text mb-2">Context engineering pipeline</h4>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
              {contextLayers.write_focus ? (
                <div>
                  <p className="text-text-muted">Write Focus</p>
                  <p className="font-semibold text-text">{contextLayers.write_focus}</p>
                </div>
              ) : null}
              <div>
                <p className="text-text-muted">Compression Ratio</p>
                <p className="font-semibold text-text">
                  {typeof contextLayers.compression_ratio === 'number'
                    ? `${(contextLayers.compression_ratio * 100).toFixed(0)}%`
                    : '—'}
                </p>
              </div>
              <div>
                <p className="text-text-muted">Token Budget</p>
                <p className="font-semibold text-text">
                  {typeof contextLayers.token_budget === 'number'
                    ? contextLayers.token_budget.toLocaleString()
                    : '—'}
                </p>
              </div>
              <div>
                <p className="text-text-muted">Search Queries</p>
                <p className="font-semibold text-text">
                  {typeof contextLayers.search_queries_count === 'number'
                    ? contextLayers.search_queries_count
                    : '—'}
                </p>
              </div>
              <div>
                <p className="text-text-muted">Isolation Strategy</p>
                <p className="font-semibold capitalize text-text">{contextLayers.isolation_strategy || '—'}</p>
              </div>
            </div>
            {contextLayers.layer_times ? (
              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                {Object.entries(contextLayers.layer_times).map(([k, v]) => {
                  const label = k.charAt(0).toUpperCase() + k.slice(1)
                  const value = typeof v === 'number' ? v : Number(v)
                  return (
                    <div key={k} className="bg-blue-100/60 dark:bg-blue-900/30 rounded p-2">
                      <p className="text-text">{label} Time</p>
                      <p className="font-semibold text-text">{Number.isFinite(value) ? `${value.toFixed(2)}s` : '—'}</p>
                    </div>
                  )
                })}
              </div>
            ) : null}
            {contextLayers.budget_plan && Object.keys(contextLayers.budget_plan).length > 0 ? (
              <div className="mt-3">
                <p className="text-xs text-text mb-1">Token Budget Plan</p>
                <div className="flex items-center gap-1">
                  {Object.entries(contextLayers.budget_plan).map(([k, v]) => {
                    const totalBudget = typeof contextLayers.token_budget === 'number' ? contextLayers.token_budget : 0
                    const numericValue = typeof v === 'number' ? v : Number(v)
                    const widthPercent = totalBudget > 0 ? Math.min(100, (numericValue / totalBudget) * 100) : 0
                    return (
                      <div key={k} className="flex-1">
                        <div className="h-2 rounded" style={{ width: '100%', background: 'rgba(59,130,246,0.15)' }}>
                          <div className="h-2 rounded bg-blue-600" style={{ width: `${widthPercent}%` }} />
                        </div>
                        <div className="text-[10px] text-text mt-0.5">{k} · {Number.isFinite(numericValue) ? numericValue.toLocaleString() : '0'}t</div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ) : null}
            {(contextLayers.rewrite_primary || contextLayers.optimize_primary) ? (
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {contextLayers.rewrite_primary ? (
                  <div className="bg-blue-100/60 dark:bg-blue-900/30 rounded p-2">
                    <p className="text-text mb-1">
                      Rewritten Query
                      {typeof contextLayers.rewrite_alternatives === 'number' ? ` (${contextLayers.rewrite_alternatives} alts)` : ''}
                    </p>
                    <p className="font-mono text-[11px] break-words text-text">{contextLayers.rewrite_primary}</p>
                  </div>
                ) : null}
                {contextLayers.optimize_primary ? (
                  <div className="bg-blue-100/60 dark:bg-blue-900/30 rounded p-2">
                    <p className="text-text mb-1">
                      Optimized Primary
                      {typeof contextLayers.optimize_variations_count === 'number' ? ` (${contextLayers.optimize_variations_count} vars)` : ''}
                    </p>
                    <p className="font-mono text-[11px] break-words text-text">{contextLayers.optimize_primary}</p>
                  </div>
                ) : null}
              </div>
            ) : null}
            {typeof contextLayers.refined_queries_count === 'number' ? (
              <p className="mt-2 text-[11px] text-text">Refined queries: {contextLayers.refined_queries_count}</p>
            ) : null}
            {contextLayers.isolated_findings ? (
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                <div className="bg-blue-100/60 dark:bg-blue-900/30 rounded p-2">
                  <p className="text-text">Isolation Focus Areas</p>
                  <p className="font-semibold text-text truncate">
                    {(contextLayers.isolated_findings.focus_areas ?? []).join(', ') || '—'}
                  </p>
                </div>
                <div className="bg-blue-100/60 dark:bg-blue-900/30 rounded p-2">
                  <p className="text-text">Extraction Patterns</p>
                  <p className="font-semibold text-text">
                    {typeof contextLayers.isolated_findings.patterns === 'number'
                      ? contextLayers.isolated_findings.patterns
                      : '—'}
                  </p>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {results.paradigm_analysis?.primary?.paradigm === 'bernard' && typeof ((baseAnswer?.metadata as AnswerMetadataType | undefined)?.statistical_insights) === 'number' && (
          <div className="mt-4 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800 transition-colors duration-200">
            <h4 className="text-sm font-semibold text-indigo-900 dark:text-indigo-100 mb-1">Analytical signals</h4>
            <p className="text-sm text-indigo-800 dark:text-indigo-200">
              {(baseAnswer?.metadata as AnswerMetadataType).statistical_insights} statistical insights detected across sources.
            </p>
          </div>
        )}

        {Array.isArray(evidenceQuotes) && evidenceQuotes.length > 0 && (
          <EvidencePanel quotes={evidenceQuotes} />
        )}

        {results.cost_info && (
          <div className="mt-4 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800 transition-colors duration-200">
            <h4 className="text-sm font-semibold text-amber-900 dark:text-amber-100 mb-2">Research costs</h4>
            <div className="grid grid-cols-3 gap-3 text-sm">
              {results.cost_info.search_api_costs !== undefined && (
                <div>
                  <p className="text-amber-700 dark:text-amber-300">Search API</p>
                  <p className="font-semibold text-amber-900 dark:text-amber-100">${results.cost_info.search_api_costs.toFixed(3)}</p>
                </div>
              )}
              {results.cost_info.llm_costs !== undefined && (
                <div>
                  <p className="text-amber-700 dark:text-amber-300">LLM Processing</p>
                  <p className="font-semibold text-amber-900 dark:text-amber-100">${results.cost_info.llm_costs.toFixed(3)}</p>
                </div>
              )}
              {results.cost_info.total !== undefined && (
                <div>
                  <p className="text-amber-700 dark:text-amber-300">Total Cost</p>
                  <p className="font-semibold text-amber-900 dark:text-amber-100">${results.cost_info.total.toFixed(3)}</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {Array.isArray(metadata?.agent_trace) && metadata.agent_trace.length > 0 && (
        <div className="bg-surface rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up border border-border" style={{ animationDelay: '0.55s' }}>
          <button
            onClick={() => setTraceOpen(!traceOpen)}
            className="w-full text-left flex items-center justify-between"
            aria-expanded={traceOpen}
          >
            <h3 className="text-lg font-semibold text-text">Agentic research trace</h3>
            {traceOpen ? <FiChevronUp className="h-5 w-5" /> : <FiChevronDown className="h-5 w-5" />}
          </button>
          {traceOpen && (
            <div className="mt-3 space-y-2 text-sm text-text">
              {(metadata.agent_trace as unknown as AgentTraceEntry[]).map((entry, idx) => (
                <div key={idx} className="border border-border rounded p-3 bg-surface">
                  <div className="flex items-center gap-2 text-text-muted">
                    <FiClock className="h-4 w-4" />
                    <span className="uppercase tracking-wide text-xs font-semibold">{String(entry.step || 'revise')}</span>
                    {typeof entry.iteration === 'number' && (
                      <span className="text-xs">iter {entry.iteration}</span>
                    )}
                    {typeof entry.coverage === 'number' && (
                      <span className="ml-auto text-xs">coverage {(entry.coverage * 100).toFixed(0)}%</span>
                    )}
                  </div>
                  {Array.isArray(entry.proposed_queries) && entry.proposed_queries.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs text-text-subtle">Proposed Queries</p>
                      <ul className="list-disc list-inside space-y-1">
                        {entry.proposed_queries.map((q, i) => (
                          <li key={i} className="break-all">{q}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {Array.isArray(entry.warnings) && entry.warnings.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs text-amber-600 dark:text-amber-400">Warnings</p>
                      <ul className="list-disc list-inside space-y-1">
                        {entry.warnings.map((w, i) => (
                          <li key={i}>{w}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <ContextMetricsPanel />

      {integrated_synthesis && (
        <div className="bg-surface rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up border border-border" style={{ animationDelay: '0.1s' }}>
          <h3 className="text-lg font-semibold text-text mb-4">Mesh network analysis</h3>
          {Array.isArray(integrated_synthesis.synergies) && integrated_synthesis.synergies.length > 0 && (
            <div className="mb-4">
              <h4 className="font-semibold text-text flex items-center"><FiGitMerge className="h-5 w-5 mr-2 text-success" />Synergies</h4>
              <ul className="list-disc list-inside mt-2 text-text">
                {integrated_synthesis.synergies.map((synergy, i) => <li key={i}>{synergy}</li>)}
              </ul>
            </div>
          )}
          {Array.isArray(integrated_synthesis.conflicts_identified) && integrated_synthesis.conflicts_identified.length > 0 && (
            <div>
              <h4 className="font-semibold text-text flex items-center"><FiZap className="h-5 w-5 mr-2 text-error" />Conflicts</h4>
              <ul className="list-disc list-inside mt-2 text-text">
                {integrated_synthesis.conflicts_identified.map((conflict, i) => <li key={i}>{conflict.description}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}

      <div className="space-y-4">
        {allSections.map((section: AnswerSection, index) => (
          <div key={`${section.title}-${index}`} className="bg-surface rounded-lg shadow-lg overflow-hidden transition-all duration-300 animate-slide-up border border-border" style={{ animationDelay: `${0.2 + index * 0.05}s` }}>
            <button
              onClick={() => toggleSection(index)}
              className="w-full px-6 py-4 flex items-center justify-between hover:bg-surface-subtle transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
            >
              <div className="flex items-center gap-3">
                <h3 className="text-lg font-semibold text-text">{section.title}</h3>
                <span className={`px-2 py-1 rounded text-xs font-medium ${getParadigmClass((section as any)?.paradigm || 'bernard')}`}>
                  {getParadigmDescription((section as any)?.paradigm || 'bernard')}
                </span>
                <span className="text-sm text-text-muted">
                  {Number((section as any)?.sources_count ?? 0)} sources • {Math.round(Number((section as any)?.confidence ?? 0) * 100)}% confidence
                </span>
              </div>
              {expandedSections.has(index) ? (
                <FiChevronUp className="h-5 w-5 text-text-muted transition-transform duration-200" />
              ) : (
                <FiChevronDown className="h-5 w-5 text-text-muted transition-transform duration-200" />
              )}
            </button>

            {expandedSections.has(index) && (
              <div className="px-6 pb-4 border-t border-border animate-slide-down">
                <div className="prose dark:prose-invert max-w-none mt-4">
                  <p className="text-text whitespace-pre-wrap">{section.content}</p>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {actionItems.length > 0 && (
        <div className="bg-surface rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up border border-border" style={{ animationDelay: '0.3s' }}>
          <h3 className="text-lg font-semibold text-text mb-4">Detailed action items</h3>
          <div className="space-y-3">
            {actionItems.map((item: ActionItem, index: number) => (
              <div key={`${item.action || 'action'}-${index}`} className="flex items-start gap-3 p-3 bg-surface-subtle rounded-lg transition-colors duration-200 hover:bg-surface">
                <div className="mt-0.5">
                  {getPriorityIcon(item.priority || 'low')}
                </div>
                <div className="flex-1">
                  <p className="text-text font-medium">{item.action}</p>
                  <div className="flex items-center gap-4 mt-1 text-sm text-text-muted">
                    <span>Timeframe: {item.timeframe || '—'}</span>
                    <span>Owner: {item.owner || 'Unassigned'}</span>
                    <span>Due: {item.due_date ? new Date(item.due_date).toLocaleDateString() : 'Set due date'}</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${getParadigmClass(item.paradigm || 'bernard')}`}>
                      {getParadigmDescription(item.paradigm || 'bernard')}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {sourcesList.length > 0 && (
        <div className="bg-surface rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up border border-border" style={{ animationDelay: '0.4s' }}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text">Research sources</h3>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <FiFilter className="h-4 w-4 text-text-subtle" />
                <div className="flex flex-wrap gap-2">
                  {['all', ...Object.keys(categoryDistribution)].map((cat) => (
                    <Button
                      key={cat}
                      size="sm"
                      variant={selectedCategories.has(cat) ? 'primary' : 'ghost'}
                      onClick={() => {
                        setSelectedCategories(prev => {
                          const next = new Set(prev)
                          if (next.has(cat)) next.delete(cat); else next.add(cat)
                          if (cat === 'all') return new Set(['all'])
                          next.delete('all')
                          if (next.size === 0) next.add('all')
                          return next
                        })
                      }}
                    >
                      {cat}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2 text-xs">
                {(['high', 'medium', 'low'] as const).map(band => (
                  <Button
                    key={band}
                    size="sm"
                    variant={
                      selectedCredBands.has(band)
                        ? band === 'high'
                          ? 'success'
                          : band === 'medium'
                            ? 'primary'
                            : 'danger'
                        : 'ghost'
                    }
                    onClick={() =>
                      setSelectedCredBands(prev => {
                        const next = new Set(prev)
                        if (next.has(band)) next.delete(band)
                        else next.add(band)
                        if (next.size === 0) return new Set(['high', 'medium', 'low'])
                        return next
                      })
                    }
                  >
                    {band}
                  </Button>
                ))}
              </div>
            </div>
          </div>

          <>
            <div className="flex items-center justify-between mb-3 text-sm text-text-muted">
              <div>
                {totalSourcesFiltered === 0 ? 'No sources' : `Showing ${pageStart + 1}–${pageEnd} of ${totalSourcesFiltered} sources`}
              </div>
              <div className="flex items-center gap-2">
                <label className="whitespace-nowrap">Rows per page</label>
                <select
                  className="border border-border rounded px-2 py-1 bg-surface text-text"
                  value={sourcesPageSize}
                  onChange={(e) => {
                    const val = (e.target.value === 'all') ? (totalSourcesFiltered || 1) : Number(e.target.value)
                    setSourcesPageSize(val)
                    setSourcesPage(1)
                  }}
                >
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                  <option value="all">All</option>
                </select>
                <div className="flex items-center gap-1 ml-2">
                  <button
                    className="px-2 py-1 rounded border border-border disabled:opacity-50"
                    onClick={() => setSourcesPage(p => Math.max(1, p - 1))}
                    disabled={currentPage <= 1}
                    aria-label="Previous page"
                  >
                    ‹
                  </button>
                  <span className="px-2">{currentPage} / {pageCount}</span>
                  <button
                    className="px-2 py-1 rounded border border-border disabled:opacity-50"
                    onClick={() => setSourcesPage(p => Math.min(pageCount, p + 1))}
                    disabled={currentPage >= pageCount}
                    aria-label="Next page"
                  >
                    ›
                  </button>
                </div>
              </div>
            </div>
            <div className="grid gap-3">
              {viewSources.map((source, index) => {
                const quote = (source.key_quote || '').trim()
                const words = quote.split(/\s+/).filter(Boolean)
                const shortQuote = quote ? (words.slice(0, 20).join(' ') + (words.length > 20 ? '…' : '')) : ''
                const score = Number(source.credibility_score || 0)
                const qual = qualityLabel(score)
                const why = source.credibility_explanation || `Credibility: ${qual} (${(score * 100).toFixed(0)}%)`
                const itemKey = source.url || `${source.title || ''}-${source.domain || ''}-${source.published_date || ''}-${index}`

                return (
                  <div
                    key={itemKey}
                    className="border border-border rounded-lg p-4 hover:border-border transition-all duration-200 hover:shadow-md bg-surface"
                    role="article"
                    aria-label={source.title}
                  >
                    <h4 className="font-medium text-text">{source.title}</h4>
                    <div className="mt-2 grid md:grid-cols-3 gap-3 text-sm">
                      <div>
                        <p className="text-text-subtle">What it says</p>
                        <p className="text-text">{source.snippet || '—'}</p>
                      </div>
                      <div>
                        <p className="text-text-subtle">Key quote</p>
                        <p className="italic text-text">{quote ? `“${shortQuote}”` : '—'}</p>
                      </div>
                      <div>
                        <p className="text-text-subtle">Why it matters</p>
                        <p className="text-text">{why}</p>
                      </div>
                    </div>
                    <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
                      <span className="text-text-subtle">{getDomain(source.url) || source.domain}</span>
                      {source.source_category && (
                        <span className="px-2 py-0.5 rounded bg-surface-subtle text-text border border-border">{source.source_category}</span>
                      )}
                      <span className="px-2 py-0.5 rounded bg-surface-subtle text-text border border-border">
                        Quality: {qual}{typeof source.credibility_score === 'number' ? ` (${(source.credibility_score * 100).toFixed(0)}%)` : ''}
                      </span>
                      {source.published_date && (
                        <span className="text-text-subtle">Published {new Date(source.published_date).toLocaleDateString()}</span>
                      )}
                      <span className="text-text-subtle">Indexed {new Date(fetchedAtRef.current).toLocaleDateString()}</span>
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="ml-auto inline-flex items-center gap-1 px-2 py-1 text-primary hover:underline"
                        aria-label="Open source"
                      >
                        <FiExternalLink className="h-3.5 w-3.5" /> Open source
                      </a>
                    </div>
                  </div>
                )
              })}
            </div>
            {totalSourcesFiltered > 0 && (
              <div className="mt-4 flex items-center justify-between text-sm text-text-muted">
                <div>
                  Page {currentPage} of {pageCount}
                </div>
                <div className="flex items-center gap-1">
                  <button
                    className="px-2 py-1 rounded border border-border disabled:opacity-50"
                    onClick={() => setSourcesPage(1)}
                    disabled={currentPage === 1}
                    aria-label="First page"
                  >
                    «
                  </button>
                  <button
                    className="px-2 py-1 rounded border border-border disabled:opacity-50"
                    onClick={() => setSourcesPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    aria-label="Previous page"
                  >
                    ‹
                  </button>
                  <button
                    className="px-2 py-1 rounded border border-border disabled:opacity-50"
                    onClick={() => setSourcesPage(p => Math.min(pageCount, p + 1))}
                    disabled={currentPage === pageCount}
                    aria-label="Next page"
                  >
                    ›
                  </button>
                  <button
                    className="px-2 py-1 rounded border border-border disabled:opacity-50"
                    onClick={() => setSourcesPage(pageCount)}
                    disabled={currentPage === pageCount}
                    aria-label="Last page"
                  >
                    »
                  </button>
                </div>
              </div>
            )}
          </>
        </div>
      )}
      {Array.isArray(results.sources) && results.sources.length === 0 && (
        <div className="bg-surface rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up border border-border" style={{ animationDelay: '0.4s' }}>
          <h3 className="text-lg font-semibold text-text mb-2">Research sources</h3>
          <p className="text-sm text-text-muted">No sources available. Try broadening the query or enabling real search.</p>
        </div>
      )}

      {results.cost_info && (
        <div className="bg-surface rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up border border-border" style={{ animationDelay: '0.5s' }}>
          <h3 className="text-lg font-semibold text-text mb-4">Research metrics</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(results.cost_info).map(([key, value]) => (
              <div key={key} className="bg-surface-subtle rounded-lg p-3">
                <p className="text-sm text-text-muted">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                <p className="text-lg font-semibold text-text">
                  {typeof value === 'number'
                    ? (key.toLowerCase().includes('cost') || key.toLowerCase() === 'total' ? `$${value.toFixed(4)}` : value.toFixed(2))
                    : (typeof value === 'string'
                      ? value
                      : JSON.stringify(value))}
                </p>
              </div>
            ))}
            {metadata?.processing_time_seconds !== undefined && (
              <div className="bg-surface-subtle rounded-lg p-3">
                <p className="text-sm text-text-muted">Processing Time</p>
                <p className="text-lg font-semibold text-text">
                  {Number(metadata.processing_time_seconds).toFixed(1)}s
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="bg-surface rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up border border-border" style={{ animationDelay: '0.55s' }}>
        <h3 className="text-lg font-semibold text-text mb-4">Feedback</h3>
        <div className="grid grid-cols-1">
          <AnswerFeedback researchId={results.research_id} />
        </div>
      </div>

      <div className="bg-surface rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up border border-border" style={{ animationDelay: '0.6s' }}>
        <h3 className="text-lg font-semibold text-text mb-4">Answer citations</h3>
        <div className="space-y-3">
          {displayedCitations.map((citation: AnswerCitation, idx: number) => {
            const d = getDomain(citation.url) || getDomain(citation.source)
            const info = domainInfo[d] || {}
            const parsed = parseExplanation(info.explanation)
            const score = Number((citation.credibility_score ?? info.score ?? 0))
            const trustLabel = score >= 0.8 ? 'Very High' : score >= 0.6 ? 'High' : score >= 0.4 ? 'Moderate' : score >= 0.2 ? 'Low' : 'Very Low'
            const key = citation.id || citation.url || `${citation.title || 'citation'}-${idx}`
            const paradigmAlign = citation.paradigm_alignment || 'bernard'
            return (
              <div key={key} className="border border-border rounded-lg p-4 hover:border-border transition-all duration-200 hover:shadow-md bg-surface">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium text-text">{citation.title}</h4>
                    <p className="text-sm text-text-muted mt-1">{d || citation.source}</p>

                    <div className="flex items-center gap-4 mt-2">
                      <div className={`flex items-center gap-1 text-sm ${getCredibilityColor(score)}`}>
                        {getCredibilityIcon(score)}
                        <span className="font-medium">
                          {trustLabel} ({(score * 100).toFixed(0)}%)
                        </span>
                      </div>
                      {info.category && (
                        <span className="px-2 py-0.5 rounded text-xs bg-surface-subtle text-text border border-border">
                          {info.category}
                        </span>
                      )}
                      {parsed.bias && (
                        <span className="px-2 py-0.5 rounded text-xs bg-surface-subtle text-text border border-border">
                          bias: {parsed.bias}
                        </span>
                      )}
                      {parsed.fact && (
                        <span className="px-2 py-0.5 rounded text-xs bg-surface-subtle text-text border border-border">
                          factual: {parsed.fact}
                        </span>
                      )}

                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${getParadigmClass(paradigmAlign || 'bernard')}`}>
                        {getParadigmDescription(paradigmAlign || 'bernard')}
                      </span>
                    </div>
                  </div>

                  <a
                    href={citation.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-4 p-2 text-text-muted hover:text-text hover:bg-surface-subtle rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    aria-label="Open citation in new tab"
                  >
                    <FiExternalLink className="h-4 w-4" />
                  </a>
                </div>
              </div>
            )
          })}
        </div>

        {citations.length > 5 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAllCitations(!showAllCitations)}
          >
            {showAllCitations ? 'Show less' : `Show all ${citations.length} citations`}
          </Button>
        )}
      </div>
    </div>
  )
}
