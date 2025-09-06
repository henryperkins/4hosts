import React, { useState, useRef, useEffect } from 'react'
import { FiDownload, FiExternalLink, FiShield, FiAlertTriangle, FiChevronDown, FiChevronUp, FiZap, FiGitMerge, FiLoader, FiCheckCircle, FiAlertCircle, FiClock, FiFilter } from 'react-icons/fi'
import toast from 'react-hot-toast'
import api from '../services/api'
import type { ResearchResult, AnswerSection } from '../types'
import { getParadigmClass, getParadigmDescription } from '../constants/paradigm'
import { ContextMetricsPanel } from './ContextMetricsPanel'
import { EvidencePanel } from './EvidencePanel'

interface ResultsDisplayEnhancedProps {
  results: ResearchResult
}


export const ResultsDisplayEnhanced: React.FC<ResultsDisplayEnhancedProps> = ({ results }) => {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set([0]))
  const [isExporting, setIsExporting] = useState(false)
  // Track currently-exporting format
  const [exportFormat, setExportFormat] = useState<string | null>(null)
  const [showAllCitations, setShowAllCitations] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const [traceOpen, setTraceOpen] = useState(false)
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set(['all']))
  const [selectedCredBands, setSelectedCredBands] = useState<Set<'high'|'medium'|'low'>>(() => new Set(['high','medium','low']))
  // Stable timestamp for this render (used for "As of" labels)
  const fetchedAtRef = useRef<string>(new Date().toISOString())

  // Safely handle cases where answer might be undefined (failed research)
  const answer = results.integrated_synthesis 
    ? results.integrated_synthesis.primary_answer 
    : results.answer;
  const { integrated_synthesis } = results;
  // Prefer SSOTA `metadata.context_layers`, fall back to legacy `paradigm_analysis.context_engineering`
  const contextLayers = (results as any)?.metadata?.context_layers || (results as any)?.paradigm_analysis?.context_engineering;
  const actionableRatio = Number(results.metadata?.actionable_content_ratio || 0)
  const bias = (results.metadata as any)?.bias_check as (ResearchResult['metadata'] & { bias_check?: any })['bias_check']

  // Build domain info map for citation credibility mini-cards
  const domainInfo = React.useMemo(() => {
    const map: Record<string, { category?: string; explanation?: string; score?: number }> = {}
    for (const s of results.sources || []) {
      const d = (s.domain || '').toLowerCase()
      if (d && !map[d]) {
        map[d] = { category: s.source_category, explanation: s.credibility_explanation, score: s.credibility_score }
      }
    }
    return map
  }, [results.sources])

  // Helper: map credibility score to quality band
  const qualityLabel = (score?: number) => {
    if (typeof score !== 'number') return 'Unknown'
    if (score >= 0.8) return 'Strong'
    if (score >= 0.6) return 'Moderate'
    return 'Weak'
  }

  // Compute evidence snapshot and timeframe window
  const evidenceSnapshot = React.useMemo(() => {
    const total = (results.sources || []).length
    let strong = 0, moderate = 0, weak = 0
    let minDate: number | null = null
    let maxDate: number | null = null
    for (const s of results.sources || []) {
      const sc = s.credibility_score
      if (typeof sc === 'number') {
        if (sc >= 0.8) strong++
        else if (sc >= 0.6) moderate++
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
  }, [results.sources])

  // Derive a concise bottom‑line sentence from the summary
  function bottomLine(text: string, maxWords = 20): string {
    if (!text) return 'No summary available.'
    const firstSentence = text.split(/(?<=[.!?])\s+/)[0] || text
    const words = firstSentence.trim().split(/\s+/)
    if (words.length <= maxWords) return firstSentence.trim()
    return words.slice(0, maxWords).join(' ') + '…'
  }

  // Build a human reason for confidence
  const confidenceInfo = React.useMemo(() => {
    const conf = (results.integrated_synthesis?.confidence_score ?? results.paradigm_analysis?.primary?.confidence ?? 0) * 100
    const summary = results.metadata?.credibility_summary
    const highShare = summary?.high_credibility_ratio ?? (summary?.high_credibility_count && results.sources?.length ? summary.high_credibility_count / Math.max(1, results.sources.length) : undefined)
    const uniqueCats = Object.keys(results.metadata?.category_distribution || {}).length
    let band: 'High' | 'Medium' | 'Low' = 'Low'
    if (conf >= 80) band = 'High'
    else if (conf >= 60) band = 'Medium'
    const because: string[] = []
    if (typeof highShare === 'number') because.push(`${Math.round(highShare * 100)}% high‑credibility sources`)
    if (uniqueCats >= 3) because.push(`${uniqueCats} source types`)
    if ((results.metadata?.actionable_content_ratio || 0) >= 0.85) because.push('high actionable content')
    return { conf, band, because: because.join('; ') }
  }, [results])

  // Parse bias/factual from explanation like "bias=left, fact=high, cat=academic"
  function parseExplanation(expl?: string): { bias?: string; fact?: string; cat?: string } {
    if (!expl) return {}
    const out: any = {}
    const pairs = expl.split(',')
    for (const p of pairs) {
      const [k, v] = p.split('=').map(s => (s || '').trim())
      if (!k || !v) continue
      if (k.startsWith('bias')) out.bias = v
      if (k.startsWith('fact')) out.fact = v
      if (k.startsWith('cat')) out.cat = v
    }
    return out
  }

  // Close dropdown when clicking outside
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

  // If there's no answer data, show error state
  if (!answer) {
    return (
      <div className="mt-8 animate-fade-in">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 text-center transition-colors duration-200">
          <FiAlertCircle className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Research Incomplete
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            This research could not be completed due to an error during processing.
          </p>
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <p className="text-sm text-red-600 dark:text-red-400">
              Status: {results.status || 'Unknown'}
            </p>
            {results.metadata && (
              <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                Research ID: {results.research_id}
              </p>
            )}
          </div>
        </div>
      </div>
    )
  }

  const toggleSection = (index: number) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  const handleExport = async (format: string) => {
    setIsExporting(true)
    setExportFormat(format)
    setDropdownOpen(false)

    try {
      const blob = await api.exportResearch(results.research_id, format as 'pdf' | 'json' | 'csv' | 'markdown' | 'excel')
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `research-${results.research_id}.${format}`
      link.click()
      URL.revokeObjectURL(url)

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

  const getCredibilityColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 dark:text-green-400'
    if (score >= 0.6) return 'text-yellow-600 dark:text-yellow-400'
    if (score >= 0.4) return 'text-orange-600 dark:text-orange-400'
    return 'text-red-600 dark:text-red-400'
  }

  const getCredibilityIcon = (score: number) => {
    if (score >= 0.8) return <FiShield className="h-4 w-4" aria-label="High credibility" />
    if (score >= 0.4) return <FiAlertTriangle className="h-4 w-4" aria-label="Medium credibility" />
    return <FiAlertTriangle className="h-4 w-4" aria-label="Low credibility" />
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

  // Safely handle all answer properties that might be undefined
  const citations = answer.citations || []
  const sections = answer.sections || []
  const actionItems = answer.action_items || []
  const summary = answer.summary || 'No summary available'
  const answerMetadata: Record<string, any> | undefined = (answer as any)?.metadata
  const evidenceQuotes: any[] = Array.isArray(answerMetadata?.evidence_quotes)
    ? (answerMetadata!.evidence_quotes as any[])
    : (Array.isArray((results as any)?.metadata?.evidence_quotes) ? (results as any).metadata.evidence_quotes : [])
  
  const displayedCitations = showAllCitations
    ? citations
    : citations.slice(0, 5)

  const allSections = integrated_synthesis?.secondary_perspective
    ? [...sections, integrated_synthesis.secondary_perspective]
    : sections;

  return (
    <div className="mt-8 space-y-6 animate-fade-in">
      {/* Summary Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Research results</h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Query: "{results.query}"</p>
            <p className="text-xs text-gray-500 dark:text-gray-400">As of {new Date(fetchedAtRef.current).toLocaleString()}</p>
          </div>

          <div className="flex items-center gap-2">
            <span className={`px-3 py-1 rounded-full text-sm font-medium border ${
              getParadigmClass(results.paradigm_analysis.primary.paradigm)
            }`}>
              {getParadigmDescription(results.paradigm_analysis.primary.paradigm)}
            </span>
            {results.paradigm_analysis.secondary && (
                <span className={`px-3 py-1 rounded-full text-sm font-medium border ${
                    getParadigmClass(results.paradigm_analysis.secondary.paradigm)
                }`}>
                    + {getParadigmDescription(results.paradigm_analysis.secondary.paradigm)}
                </span>
            )}

            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                disabled={isExporting}
                className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
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

              {/* Build export list dynamically from backend-provided URLs when available */}
              <div
                className={`absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 transition-all duration-200 z-10 ${
                  dropdownOpen ? 'opacity-100 visible translate-y-0' : 'opacity-0 invisible -translate-y-2'
                }`}
              >
                {
                  (() => {
                    // backend may provide export_formats mapping; fall back to defaults
                    const allowed = ['json', 'csv', 'pdf', 'markdown', 'excel'] as const
                    const map: Record<string, string> = (results as any).export_formats || {}
                    if (Object.keys(map).length === 0) {
                      allowed.forEach((f) => {
                        map[f] = `/v1/research/${results.research_id}/export/${f}`
                      })
                    }
                    return allowed.filter((f) => f in map).map((fmt) => (
                      <button
                        key={fmt}
                        onClick={() => handleExport(fmt)}
                        className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 flex items-center gap-2 capitalize"
                        disabled={isExporting}
                      >
                        {exportFormat === fmt && isExporting ? (
                          <FiLoader className="h-4 w-4 animate-spin" />
                        ) : null}
                        Export as {fmt.toUpperCase()}
                      </button>
                    ))
                  })()
                }
              </div>
            </div>
          </div>
        </div>

        {/* Bottom line and evidence snapshot */}
        <div className="mt-3">
          <p className="text-gray-900 dark:text-gray-100 text-base"><span className="font-semibold">Bottom line:</span> {bottomLine(summary)}</p>
          <div className="mt-2 text-sm text-gray-700 dark:text-gray-300">
            <span className="font-medium">Evidence:</span> {evidenceSnapshot.total.toLocaleString()} sources ({evidenceSnapshot.strong} strong, {evidenceSnapshot.moderate} moderate, {evidenceSnapshot.weak} weak){evidenceSnapshot.window ? ` · timeframe ${evidenceSnapshot.window}` : ''}.
          </div>
          <div className="mt-1 text-sm text-gray-700 dark:text-gray-300">
            <span className="font-medium">Confidence:</span> {confidenceInfo.band} ({Math.round(confidenceInfo.conf)}%){confidenceInfo.because ? ` — because ${confidenceInfo.because}.` : '.'}
          </div>
        </div>

        {/* Quality checks & paradigm fit */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg border border-gray-200 dark:border-gray-600">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-300">Actionable Content</span>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${actionableRatio >= 0.85 ? 'bg-green-600 text-white' : 'bg-amber-500 text-white'}`}>
                {(actionableRatio * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Estimated share of concrete actions and key insights.</p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg border border-gray-200 dark:border-gray-600">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-300">Bias Check</span>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${bias?.balanced ? 'bg-green-600 text-white' : 'bg-amber-500 text-white'}`}>
                {bias?.balanced ? 'Balanced' : 'Needs Balance'}
              </span>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Domain diversity {(bias?.domain_diversity ? (bias.domain_diversity * 100).toFixed(0) : '0')}%{bias?.dominant_domain ? `, dominant: ${bias.dominant_domain} ${(bias.dominant_share! * 100).toFixed(0)}%` : ''}</p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg border border-gray-200 dark:border-gray-600">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-300">Paradigm Fit</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-600 text-white">
                {((results.metadata?.paradigm_fit?.confidence || 0) * 100).toFixed(0)}% conf · margin {((results.metadata?.paradigm_fit?.margin || 0) * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Primary: {results.metadata?.paradigm_fit?.primary || '-'}</p>
          </div>
        </div>

        {/* Analyzed → High-Quality banner */}
        <div className="mt-3 text-sm text-gray-700 dark:text-gray-300">
          <span className="font-medium">Analyzed</span> {results.metadata.total_sources_analyzed} sources
          <span> → </span>
          <span className="font-medium">{results.metadata.high_quality_sources}</span> high‑quality
          {results.metadata?.credibility_summary?.average_score !== undefined && (
            <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">avg credibility {(results.metadata.credibility_summary.average_score * 100).toFixed(0)}%</span>
          )}
        </div>

        {/* Scales used */}
        <div className="mt-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600">
          <p className="text-xs text-gray-600 dark:text-gray-300">
            <span className="font-semibold">Scales:</span> Confidence — High ≥ 80%, Medium 60–79%, Low &lt; 60%. Quality — Strong ≥ 0.80, Moderate 0.60–0.79, Weak &lt; 0.60.
          </p>
        </div>

        <div className="prose dark:prose-invert max-w-none mt-3">
          <p className="text-gray-700 dark:text-gray-300">{integrated_synthesis ? integrated_synthesis.integrated_summary : summary}</p>
        </div>

        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 transition-colors duration-200">
            <p className="text-gray-600 dark:text-gray-400">Sources Analyzed</p>
            <p className="font-semibold text-lg text-gray-900 dark:text-gray-100">{results.metadata.total_sources_analyzed}</p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 transition-colors duration-200">
            <p className="text-gray-600 dark:text-gray-400">High Quality</p>
            <p className="font-semibold text-lg text-gray-900 dark:text-gray-100">{results.metadata.high_quality_sources}</p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 transition-colors duration-200">
            <p className="text-gray-600 dark:text-gray-400">Processing Time</p>
            <p className="font-semibold text-lg text-gray-900 dark:text-gray-100">{results.metadata.processing_time_seconds}s</p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 transition-colors duration-200">
            <p className="text-gray-600 dark:text-gray-400">Paradigms Used</p>
            <p className="font-semibold text-lg text-gray-900 dark:text-gray-100">{results.metadata.paradigms_used.length}</p>
          </div>
        </div>

        {/* Source Category & Credibility Distribution */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <p className="text-gray-600 dark:text-gray-300 mb-2">Source categories</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries((results as any)?.metadata?.category_distribution || {}).map(([cat, count]) => (
                <span key={cat} className="px-2 py-1 rounded bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 text-xs text-gray-700 dark:text-gray-200">
                  {cat}: <span className="font-semibold">{String(count)}</span>
                </span>
              ))}
              {Object.keys((results as any)?.metadata?.category_distribution || {}).length === 0 && (
                <span className="text-gray-500 dark:text-gray-400">No category data</span>
              )}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <p className="text-gray-600 dark:text-gray-300 mb-2">Credibility Distribution</p>
            {(() => {
              const dist = ((results as any)?.metadata?.credibility_summary?.score_distribution) || {}
              const total: number = (Object.values(dist) as any[]).reduce((a:number,b:any)=>a+Number(b||0),0) || 1
              const pct = (k: string) => Math.round(((Number((dist as any)[k]||0))/total)*100)
              return (
                <div>
                  <div className="h-2 w-full rounded bg-gray-200 dark:bg-gray-600 overflow-hidden">
                    <div className="h-2 bg-green-600" style={{width:`${pct('high')}%`}} />
                    <div className="h-2 bg-yellow-500" style={{width:`${pct('medium')}%`}} />
                    <div className="h-2 bg-red-500" style={{width:`${pct('low')}%`}} />
                  </div>
                  <div className="mt-2 flex gap-2 text-xs text-gray-700 dark:text-gray-300">
                    <span className="px-2 py-0.5 rounded bg-green-600 text-white">High {pct('high')}%</span>
                    <span className="px-2 py-0.5 rounded bg-yellow-500 text-white">Medium {pct('medium')}%</span>
                    <span className="px-2 py-0.5 rounded bg-red-500 text-white">Low {pct('low')}%</span>
                  </div>
                </div>
              )
            })()}
          </div>
        </div>

        {/* Bias Distribution */}
        <div className="mt-4 bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
          <p className="text-gray-600 dark:text-gray-300 mb-2">Bias Distribution</p>
          <div className="flex flex-wrap gap-2 text-xs">
            {Object.entries((results as any)?.metadata?.bias_distribution || {}).map(([k,v]) => (
              <span key={k} className="px-2 py-0.5 rounded bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 text-gray-700 dark:text-gray-200">
                {k}: <span className="font-semibold">{String(v)}</span>
              </span>
            ))}
            {Object.keys((results as any)?.metadata?.bias_distribution || {}).length === 0 && (
              <span className="text-gray-500 dark:text-gray-400">No bias data</span>
            )}
          </div>
        </div>

        {/* Deep/Advanced Depth Indicator (based on backend research_depth) */}
        {['deep', 'deep_research'].includes(String(results.metadata?.research_depth || '')) && (
          <div className="mt-4 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-lg border border-purple-200 dark:border-purple-800 transition-colors duration-200">
            <div className="flex items-center gap-2">
              <FiZap className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              <h4 className="text-sm font-semibold text-purple-900 dark:text-purple-100">Advanced Research Depth</h4>
              <span className="ml-auto text-xs font-bold bg-purple-600 text-white px-2 py-1 rounded">
                {String(results.metadata.research_depth)}
              </span>
            </div>
            <p className="text-sm text-purple-700 dark:text-purple-300 mt-2">
              This result was generated using an advanced depth setting with broader search and synthesis.
            </p>
          </div>
        )}

        {/* Executive Summary (Maeve focus) */}
        <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800/40 rounded-lg border border-gray-200 dark:border-gray-700 transition-colors duration-200">
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">Executive Summary</h4>
          <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
            {((integrated_synthesis?.primary_answer?.action_items || results.answer?.action_items || []) as any[]).slice(0,3).map((a, i) => (
              <li key={i}>{a.action || ''}{a.timeframe ? ` (${a.timeframe})` : ''}</li>
            ))}
            {((integrated_synthesis?.primary_answer?.action_items || results.answer?.action_items || []) as any[]).length === 0 && (
              <li>No immediate actions extracted.</li>
            )}
          </ul>
        </div>

        {/* Strategic Framework (Maeve + Dolores) */}
        {integrated_synthesis && (
          <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800 transition-colors duration-200">
            <h4 className="text-sm font-semibold text-green-900 dark:text-green-100 mb-3">Strategic Framework</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Immediate Opportunities (Maeve) */}
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-sm font-semibold text-gray-900 dark:text-gray-100">Immediate Opportunities (Maeve)</h5>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${getParadigmClass('maeve')}`}>Strategic</span>
                </div>
                {Array.isArray(integrated_synthesis.primary_answer?.action_items) && integrated_synthesis.primary_answer.action_items.length > 0 ? (
                  <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    {integrated_synthesis.primary_answer.action_items.map((it: any, idx: number) => (
                      <li key={idx}><span className="font-medium capitalize">{it.priority}</span>: {it.action} {it.timeframe ? (<em className="text-xs text-gray-500 dark:text-gray-400">({it.timeframe})</em>) : null}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-gray-700 dark:text-gray-300">No immediate actions extracted.</p>
                )}
              </div>

              {/* Systemic Context (Dolores) */}
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-sm font-semibold text-gray-900 dark:text-gray-100">Systemic Context (Dolores)</h5>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${getParadigmClass('dolores')}`}>Revolutionary</span>
                </div>
                <div className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                  {integrated_synthesis.secondary_perspective?.content || '—'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Context Engineering Info (unified view) */}
        {contextLayers && (
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 transition-colors duration-200">
            <h4 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">Context engineering pipeline</h4>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
              {('write_focus' in contextLayers) && (
                <div>
                  <p className="text-blue-700 dark:text-blue-300">Write Focus</p>
                  <p className="font-semibold text-blue-900 dark:text-blue-100">{contextLayers.write_focus || '—'}</p>
                </div>
              )}
              <div>
                <p className="text-blue-700 dark:text-blue-300">Compression Ratio</p>
                <p className="font-semibold text-blue-900 dark:text-blue-100">{(contextLayers.compression_ratio * 100).toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-blue-700 dark:text-blue-300">Token Budget</p>
                <p className="font-semibold text-blue-900 dark:text-blue-100">{Number(contextLayers.token_budget).toLocaleString()}</p>
              </div>
              <div>
                <p className="text-blue-700 dark:text-blue-300">Search Queries</p>
                <p className="font-semibold text-blue-900 dark:text-blue-100">{contextLayers.search_queries_count}</p>
              </div>
              <div>
                <p className="text-blue-700 dark:text-blue-300">Isolation Strategy</p>
                <p className="font-semibold capitalize text-blue-900 dark:text-blue-100">{contextLayers.isolation_strategy}</p>
              </div>
            </div>
            {contextLayers.layer_times && (
              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                {Object.entries(contextLayers.layer_times).map(([k, v]) => (
                  <div key={k} className="bg-blue-100/60 dark:bg-blue-900/30 rounded p-2">
                    <p className="text-blue-800 dark:text-blue-200">{k.charAt(0).toUpperCase() + k.slice(1)} Time</p>
                    <p className="font-semibold text-blue-900 dark:text-blue-100">{Number(v).toFixed(2)}s</p>
                  </div>
                ))}
              </div>
            )}
            {contextLayers.budget_plan && Object.keys(contextLayers.budget_plan).length > 0 && (
              <div className="mt-3">
                <p className="text-xs text-blue-800 dark:text-blue-200 mb-1">Token Budget Plan</p>
                <div className="flex items-center gap-1">
                  {Object.entries(contextLayers.budget_plan).map(([k,v]) => (
                    <div key={k} className="flex-1">
                      <div className="h-2 rounded" style={{ width: '100%', background: 'rgba(59,130,246,0.15)' }}>
                        <div className="h-2 rounded bg-blue-600" style={{ width: `${Math.min(100, (Number(v) / Math.max(1, Number(contextLayers.token_budget))) * 100)}%` }} />
                      </div>
                      <div className="text-[10px] text-blue-900 dark:text-blue-100 mt-0.5">{k} · {Number(v).toLocaleString()}t</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {(contextLayers.rewrite_primary || contextLayers.optimize_primary) && (
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {contextLayers.rewrite_primary && (
                  <div className="bg-blue-100/60 dark:bg-blue-900/30 rounded p-2">
                    <p className="text-blue-800 dark:text-blue-200 mb-1">Rewritten Query {typeof contextLayers.rewrite_alternatives === 'number' ? `(${contextLayers.rewrite_alternatives} alts)` : ''}</p>
                    <p className="font-mono text-[11px] break-words text-blue-900 dark:text-blue-100">{contextLayers.rewrite_primary}</p>
                  </div>
                )}
                {contextLayers.optimize_primary && (
                  <div className="bg-blue-100/60 dark:bg-blue-900/30 rounded p-2">
                    <p className="text-blue-800 dark:text-blue-200 mb-1">Optimized Primary {typeof contextLayers.optimize_variations_count === 'number' ? `(${contextLayers.optimize_variations_count} vars)` : ''}</p>
                    <p className="font-mono text-[11px] break-words text-blue-900 dark:text-blue-100">{contextLayers.optimize_primary}</p>
                  </div>
                )}
              </div>
            )}
            {typeof contextLayers.refined_queries_count === 'number' && (
              <p className="mt-2 text-[11px] text-blue-800 dark:text-blue-200">Refined queries: {contextLayers.refined_queries_count}</p>
            )}
            {contextLayers.isolated_findings && (
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                <div className="bg-blue-100/60 dark:bg-blue-900/30 rounded p-2">
                  <p className="text-blue-800 dark:text-blue-200">Isolation Focus Areas</p>
                  <p className="font-semibold text-blue-900 dark:text-blue-100 truncate">
                    {(contextLayers.isolated_findings.focus_areas || []).join(', ') || '—'}
                  </p>
                </div>
                <div className="bg-blue-100/60 dark:bg-blue-900/30 rounded p-2">
                  <p className="text-blue-800 dark:text-blue-200">Extraction Patterns</p>
                  <p className="font-semibold text-blue-900 dark:text-blue-100">{Number(contextLayers.isolated_findings.patterns || 0)}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Analytical Signals (Bernard) */}
        {results.paradigm_analysis?.primary?.paradigm === 'bernard' && typeof (answerMetadata?.statistical_insights) === 'number' && (
          <div className="mt-4 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800 transition-colors duration-200">
            <h4 className="text-sm font-semibold text-indigo-900 dark:text-indigo-100 mb-1">Analytical signals</h4>
            <p className="text-sm text-indigo-800 dark:text-indigo-200">
              {answerMetadata.statistical_insights} statistical insights detected across sources.
            </p>
          </div>
        )}

        {/* Evidence Quotes */}
        {Array.isArray(evidenceQuotes) && evidenceQuotes.length > 0 && (
          <EvidencePanel quotes={evidenceQuotes} />
        )}

        {/* Cost Information */}
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

      {/* Agent Trace (transparency) */}
      {Array.isArray(results.metadata?.agent_trace) && results.metadata.agent_trace.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.55s' }}>
          <button
            onClick={() => setTraceOpen(!traceOpen)}
            className="w-full text-left flex items-center justify-between"
            aria-expanded={traceOpen}
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Agentic research trace</h3>
            {traceOpen ? <FiChevronUp className="h-5 w-5" /> : <FiChevronDown className="h-5 w-5" />}
          </button>
          {traceOpen && (
            <div className="mt-3 space-y-2 text-sm text-gray-700 dark:text-gray-300">
              {results.metadata.agent_trace.map((entry: any, idx: number) => (
                <div key={idx} className="border border-gray-200 dark:border-gray-700 rounded p-3">
                  <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
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
                      <p className="text-xs text-gray-500 dark:text-gray-400">Proposed Queries</p>
                      <ul className="list-disc list-inside space-y-1">
                        {entry.proposed_queries.map((q: string, i: number) => (
                          <li key={i} className="break-all">{q}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {Array.isArray(entry.warnings) && entry.warnings.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs text-amber-600 dark:text-amber-400">Warnings</p>
                      <ul className="list-disc list-inside space-y-1">
                        {entry.warnings.map((w: string, i: number) => (
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

      {/* Context Metrics (global W‑S‑C‑I telemetry) */}
      <ContextMetricsPanel />

      {/* Mesh Network Analysis */}
      {integrated_synthesis && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.1s' }}>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Mesh network analysis</h3>
            {integrated_synthesis.synergies.length > 0 && (
                <div className="mb-4">
                    <h4 className="font-semibold text-gray-800 dark:text-gray-200 flex items-center"><FiGitMerge className="h-5 w-5 mr-2 text-green-500 dark:text-green-400" />Synergies</h4>
                    <ul className="list-disc list-inside mt-2 text-gray-700 dark:text-gray-300">
                        {integrated_synthesis.synergies.map((synergy, i) => <li key={i}>{synergy}</li>)}
                    </ul>
                </div>
            )}
            {integrated_synthesis.conflicts_identified.length > 0 && (
                <div>
                    <h4 className="font-semibold text-gray-800 dark:text-gray-200 flex items-center"><FiZap className="h-5 w-5 mr-2 text-red-500 dark:text-red-400" />Conflicts</h4>
                    <ul className="list-disc list-inside mt-2 text-gray-700 dark:text-gray-300">
                        {integrated_synthesis.conflicts_identified.map((conflict, i) => <li key={i}>{conflict.description}</li>)}
                    </ul>
                </div>
            )}
        </div>
      )}

      {/* Detailed Sections */}
      <div className="space-y-4">
        {allSections.map((section: AnswerSection, index) => (
          <div key={index} className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden transition-all duration-300 animate-slide-up" style={{ animationDelay: `${0.2 + index * 0.05}s` }}>
            <button
              onClick={() => toggleSection(index)}
              className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 dark:focus:ring-blue-400"
            >
              <div className="flex items-center gap-3">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">{section.title}</h3>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  getParadigmClass(section.paradigm)
                }`}>
                  {getParadigmDescription(section.paradigm)}
                </span>
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {section.sources_count} sources • {Math.round(section.confidence * 100)}% confidence
                </span>
              </div>
              {expandedSections.has(index) ? (
                <FiChevronUp className="h-5 w-5 text-gray-400 transition-transform duration-200" />
              ) : (
                <FiChevronDown className="h-5 w-5 text-gray-400 transition-transform duration-200" />
              )}
            </button>

            {expandedSections.has(index) && (
              <div className="px-6 pb-4 border-t border-gray-200 dark:border-gray-700 animate-slide-down">
                <div className="prose dark:prose-invert max-w-none mt-4">
                  <p className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{section.content}</p>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Action items */}
      {actionItems.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.3s' }}>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Action items</h3>
          <div className="space-y-3">
            {actionItems.map((item, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg transition-colors duration-200 hover:bg-gray-100 dark:hover:bg-gray-600">
                <div className="mt-0.5">
                  {getPriorityIcon(item.priority)}
                </div>
                <div className="flex-1">
                  <p className="text-gray-900 dark:text-gray-100 font-medium">{item.action}</p>
                  <div className="flex items-center gap-4 mt-1 text-sm text-gray-600 dark:text-gray-400">
                    <span>Timeframe: {item.timeframe || '—'}</span>
                    <span>Owner: {item.owner || 'Unassigned'}</span>
                    <span>Due: {item.due_date ? new Date(item.due_date).toLocaleDateString() : 'Set due date'}</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      getParadigmClass(item.paradigm)
                    }`}>
                      {getParadigmDescription(item.paradigm)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sources Overview with Category Filter */}
      {results.sources && results.sources.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.4s' }}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Research sources</h3>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <FiFilter className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                <div className="flex flex-wrap gap-2">
                  {['all', ...Object.keys((results as any)?.metadata?.category_distribution || {})].map((cat) => (
                    <button
                      key={cat}
                      className={`px-2 py-1 rounded text-xs ${selectedCategories.has(cat) ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
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
                    >{cat}</button>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2 text-xs">
                {(['high','medium','low'] as const).map(band => (
                  <button
                    key={band}
                    className={`px-2 py-1 rounded ${selectedCredBands.has(band) ? (band==='high'?'bg-green-600':band==='medium'?'bg-yellow-500':'bg-red-500')+' text-white':''} ${!selectedCredBands.has(band)?'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300':''}`}
                    onClick={() => setSelectedCredBands(prev => {
                      const next = new Set(prev)
                      if (next.has(band)) next.delete(band); else next.add(band)
                      if (next.size === 0) return new Set(['high','medium','low'])
                      return next
                    })}
                  >{band}</button>
                ))}
              </div>
            </div>
          </div>

          {(() => {
            const byCat = selectedCategories.has('all') ? results.sources : results.sources.filter(s => selectedCategories.has(s.source_category || 'general'))
            const credBand = (score:number) => score >= 0.7 ? 'high' : score >= 0.4 ? 'medium' : 'low'
            const filtered = byCat.filter(s => selectedCredBands.has(credBand(s.credibility_score)))
            const view = filtered.slice(0, 5)
            return (
              <>
                <div className="grid gap-3">
                  {view.map((source, index) => {
                    const quote = (source.snippet || '').trim()
                    const words = quote.split(/\s+/).filter(Boolean)
                    const shortQuote = words.slice(0, 20).join(' ') + (words.length > 20 ? '…' : '')
                    const qual = qualityLabel(source.credibility_score)
                    const whyByCategory: Record<string, string> = {
                      academic: 'Peer‑reviewed evidence; higher methodological rigor.',
                      government: 'Official guidance or data with legal/operational relevance.',
                      news: 'Current reporting; useful for recent developments.',
                      industry: 'Practical insights; may carry vendor bias.',
                    }
                    const why = whyByCategory[(source.source_category || '').toLowerCase()] || 'Adds perspective relevant to the decision.'
                    return (
                      <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:border-gray-300 dark:hover:border-gray-600 transition-all duration-200 hover:shadow-md">
                        <h4 className="font-medium text-gray-900 dark:text-gray-100">{source.title}</h4>
                        <div className="mt-2 grid md:grid-cols-3 gap-3 text-sm">
                          <div>
                            <p className="text-gray-500 dark:text-gray-400">What it says</p>
                            <p className="text-gray-800 dark:text-gray-200">{source.snippet || '—'}</p>
                          </div>
                          <div>
                            <p className="text-gray-500 dark:text-gray-400">Key quote</p>
                            <p className="italic text-gray-800 dark:text-gray-200">{quote ? `“${shortQuote}”` : '—'}</p>
                          </div>
                          <div>
                            <p className="text-gray-500 dark:text-gray-400">Why it matters</p>
                            <p className="text-gray-800 dark:text-gray-200">{why}</p>
                          </div>
                        </div>
                        <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
                          <span className="text-gray-500 dark:text-gray-400">{source.domain}</span>
                          {source.source_category && (
                            <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600">{source.source_category}</span>
                          )}
                          <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600">
                            Quality: {qual}{typeof source.credibility_score === 'number' ? ` (${(source.credibility_score * 100).toFixed(0)}%)` : ''}
                          </span>
                          {source.published_date && (
                            <span className="text-gray-500 dark:text-gray-400">Published {new Date(source.published_date).toLocaleDateString()}</span>
                          )}
                          <span className="text-gray-500 dark:text-gray-400">Indexed {new Date(fetchedAtRef.current).toLocaleDateString()}</span>
                          {source.credibility_explanation && (
                            <span className="text-gray-500 dark:text-gray-400">{source.credibility_explanation}</span>
                          )}
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="ml-auto inline-flex items-center gap-1 px-2 py-1 text-blue-700 dark:text-blue-300 hover:underline"
                            aria-label="Open source"
                          >
                            <FiExternalLink className="h-3.5 w-3.5" /> Open source
                          </a>
                        </div>
                      </div>
                    )
                  })}
                </div>
                {filtered.length > 5 && (
                  <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                    Showing 5 of {filtered.length} sources
                  </p>
                )}
              </>
            )})()}
        </div>
      )}
      {Array.isArray(results.sources) && results.sources.length === 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.4s' }}>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Research sources</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">No sources available. Try broadening the query or enabling real search.</p>
        </div>
      )}

      {/* Cost and Metadata Information */}
      {results.cost_info && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.5s' }}>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Research metrics</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(results.cost_info).map(([key, value]) => (
              <div key={key} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                <p className="text-sm text-gray-600 dark:text-gray-400">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {typeof value === 'number'
                    ? (key.includes('cost') ? `$${value.toFixed(4)}` : value.toFixed(2))
                    : (typeof value === 'string'
                        ? value
                        : JSON.stringify(value))}
                </p>
              </div>
            ))}
            {results.metadata?.processing_time_seconds && (
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                <p className="text-sm text-gray-600 dark:text-gray-400">Processing Time</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {results.metadata.processing_time_seconds.toFixed(1)}s
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Citations with Credibility */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.6s' }}>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Answer citations</h3>
        <div className="space-y-3">
          {displayedCitations.map((citation) => {
            const info = domainInfo[(citation.source || '').toLowerCase()] || {}
            const parsed = parseExplanation(info.explanation)
            const trustLabel = citation.credibility_score >= 0.8 ? 'Very High' : citation.credibility_score >= 0.6 ? 'High' : citation.credibility_score >= 0.4 ? 'Moderate' : citation.credibility_score >= 0.2 ? 'Low' : 'Very Low'
            return (
            <div key={citation.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:border-gray-300 dark:hover:border-gray-600 transition-all duration-200 hover:shadow-md">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100">{citation.title}</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{citation.source}</p>

                  <div className="flex items-center gap-4 mt-2">
                    <div className={`flex items-center gap-1 text-sm ${getCredibilityColor(citation.credibility_score)}`}>
                      {getCredibilityIcon(citation.credibility_score)}
                      <span className="font-medium">
                        {trustLabel} ({(citation.credibility_score * 100).toFixed(0)}%)
                      </span>
                    </div>
                    {info.category && (
                      <span className="px-2 py-0.5 rounded text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600">
                        {info.category}
                      </span>
                    )}
                    {parsed.bias && (
                      <span className="px-2 py-0.5 rounded text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600">
                        bias: {parsed.bias}
                      </span>
                    )}
                    {parsed.fact && (
                      <span className="px-2 py-0.5 rounded text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600">
                        factual: {parsed.fact}
                      </span>
                    )}

                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      getParadigmClass(citation.paradigm_alignment)
                    }`}>
                      {getParadigmDescription(citation.paradigm_alignment)}
                    </span>
                  </div>
                </div>

                <a
                  href={citation.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-4 p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
                  aria-label="Open citation in new tab"
                >
                  <FiExternalLink className="h-4 w-4" />
                </a>
              </div>
            </div>
          )})}
        </div>

        {citations.length > 5 && (
          <button
            onClick={() => setShowAllCitations(!showAllCitations)}
            className="mt-4 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium transition-colors duration-200 focus:outline-none focus:underline"
          >
            {showAllCitations ? 'Show less' : `Show all ${citations.length} citations`}
          </button>
        )}
      </div>
    </div>
  )
}

// Component is already exported above
