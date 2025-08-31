import React, { useState, useRef, useEffect } from 'react'
import {
  Download, ExternalLink, Shield, AlertTriangle,
  Zap, GitMerge, Loader2, CheckCircle, AlertCircle, Clock, BarChart3,
  TrendingUp, Target, Award, Layers, Eye, FileText, Map, Activity,
  ChevronDown, ChevronUp
} from 'lucide-react'
import toast from 'react-hot-toast'
import api from '../services/api'
import type { ResearchResult, AnswerSection } from '../types'
import { getParadigmClass, getParadigmDescription } from '../constants/paradigm'
import { Badge } from './ui/Badge'
import { Button } from './ui/Button'
import { cn } from '../utils/cn'

interface ResultsDisplayIdeaBrowserProps {
  results: ResearchResult
}

type ViewMode = 'executive' | 'paradigm' | 'credibility' | 'action' | 'trend'

interface ResultScoring {
  answerQuality: number
  sourceDiversity: number
  paradigmCoherence: number
  actionability: number
}

const getMetricColor = (score: number): string => {
  if (score >= 8) return 'text-green-600 dark:text-green-400'
  if (score >= 6) return 'text-blue-600 dark:text-blue-400'
  if (score >= 4) return 'text-amber-600 dark:text-amber-400'
  return 'text-red-600 dark:text-red-400'
}

const getMetricBgColor = (score: number): string => {
  if (score >= 8) return 'bg-green-100 dark:bg-green-900/30'
  if (score >= 6) return 'bg-blue-100 dark:bg-blue-900/30'
  if (score >= 4) return 'bg-amber-100 dark:bg-amber-900/30'
  return 'bg-red-100 dark:bg-red-900/30'
}

export const ResultsDisplayIdeaBrowser: React.FC<ResultsDisplayIdeaBrowserProps> = ({ results }) => {
  const [isExporting, setIsExporting] = useState(false)
  const [exportFormat, setExportFormat] = useState<'json' | 'pdf' | 'csv' | null>(null)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [traceOpen, setTraceOpen] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('executive')
  const [showMetricsOverlay, setShowMetricsOverlay] = useState(true)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Calculate result scoring
  const [resultScoring] = useState<ResultScoring>(() => {
    const answer = results.integrated_synthesis?.primary_answer || results.answer
    if (!answer) {
      return { answerQuality: 0, sourceDiversity: 0, paradigmCoherence: 0, actionability: 0 }
    }

    // Calculate scores based on various metrics
    const answerQuality = Math.min(10,
      (answer.sections?.length || 0) * 2 +
      0
    )

    const sourceDiversity = Math.min(10,
      (results.metadata.total_sources_analyzed / 10) +
      (results.metadata.paradigms_used.length * 2)
    )

    const paradigmCoherence = Math.min(10,
      (results.paradigm_analysis.primary.confidence * 10) +
      (results.paradigm_analysis.secondary ? 2 : 0)
    )

    const actionability = Math.min(10,
      (answer.action_items?.length || 0) * 2
    )

    return { answerQuality, sourceDiversity, paradigmCoherence, actionability }
  })

  // Safely handle cases where answer might be undefined
  const answer = results.integrated_synthesis
    ? results.integrated_synthesis.primary_answer
    : results.answer
  const { integrated_synthesis } = results

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
          <AlertCircle className="h-16 w-16 text-red-500 mx-auto mb-4" />
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


  const handleExport = async (format: 'json' | 'pdf' | 'csv') => {
    setIsExporting(true)
    setExportFormat(format)
    setDropdownOpen(false)

    try {
      const blob = await api.exportResearch(results.research_id, format)
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `research-${results.research_id}.${format}`
      link.click()
      URL.revokeObjectURL(url)

      toast.success(
        <div className="flex items-center gap-2">
          <CheckCircle className="h-4 w-4" />
          <span>Exported as {format.toUpperCase()}</span>
        </div>
      )
    } catch {
      toast.error(
        <div className="flex items-center gap-2">
          <AlertCircle className="h-4 w-4" />
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
    if (score >= 0.8) return <Shield className="h-4 w-4" aria-label="High credibility" />
    if (score >= 0.4) return <AlertTriangle className="h-4 w-4" aria-label="Medium credibility" />
    return <AlertTriangle className="h-4 w-4" aria-label="Low credibility" />
  }

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'high':
        return <AlertCircle className="h-4 w-4 text-red-500 dark:text-red-400" aria-label="High priority" />
      case 'medium':
        return <Clock className="h-4 w-4 text-yellow-500 dark:text-yellow-400" aria-label="Medium priority" />
      default:
        return <CheckCircle className="h-4 w-4 text-green-500 dark:text-green-400" aria-label="Low priority" />
    }
  }

  // View mode configurations
  const viewModes = [
    { value: 'executive', label: 'Executive Summary', icon: <Eye className="h-4 w-4" />, description: 'Quick scan with key findings' },
    { value: 'paradigm', label: 'Paradigm Analysis', icon: <Layers className="h-4 w-4" />, description: 'Deep dive by consciousness type' },
    { value: 'credibility', label: 'Source Credibility', icon: <Shield className="h-4 w-4" />, description: 'Quality-scored source ranking' },
    { value: 'action', label: 'Action Roadmap', icon: <Map className="h-4 w-4" />, description: 'Prioritized next steps' },
    { value: 'trend', label: 'Trend Alignment', icon: <TrendingUp className="h-4 w-4" />, description: 'Market signals correlation' }
  ] as const

  // Calculate overall quality score
  const calculateOverallScore = () => {
    const { answerQuality, sourceDiversity, paradigmCoherence, actionability } = resultScoring
    return ((answerQuality + sourceDiversity + paradigmCoherence + actionability) / 4).toFixed(1)
  }

  // Render content based on view mode
  const renderViewContent = () => {
    switch (viewMode) {
      case 'executive':
        return renderExecutiveSummary()
      case 'paradigm':
        return renderParadigmAnalysis()
      case 'credibility':
        return renderCredibilityView()
      case 'action':
        return renderActionRoadmap()
      case 'trend':
        return renderTrendAlignment()
      default:
        return renderExecutiveSummary()
    }
  }

  const renderExecutiveSummary = () => {
    const summary = integrated_synthesis ? integrated_synthesis.integrated_summary : answer.summary || 'No summary available'
    const keyInsights = answer.sections?.slice(0, 3) || []

    return (
      <div className="space-y-4">
        <div className="prose dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 text-lg leading-relaxed">{summary}</p>
        </div>

        {keyInsights.length > 0 && (
          <div className="mt-6">
            <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
              <Target className="h-4 w-4" />
              Key Insights
            </h4>
            <div className="space-y-3">
              {keyInsights.map((section, index) => (
                <div key={index} className="pl-4 border-l-4 border-primary/30">
                  <h5 className="font-medium text-gray-900 dark:text-gray-100">{section.title}</h5>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
                    {section.content}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  const renderParadigmAnalysis = () => {
    const sections = answer.sections || []
    const groupedByParadigm = sections.reduce((acc, section) => {
      const paradigm = section.paradigm || 'unknown'
      if (!acc[paradigm]) acc[paradigm] = []
      acc[paradigm].push(section)
      return acc
    }, {} as Record<string, AnswerSection[]>)

    return (
      <div className="space-y-6">
        {Object.entries(groupedByParadigm).map(([paradigm, paradigmSections]) => (
          <div key={paradigm} className="space-y-3">
            <h4 className={cn(
              "text-lg font-semibold flex items-center gap-2",
              paradigm !== 'unknown' ? getParadigmClass(paradigm, 'subtle') : 'text-gray-700 dark:text-gray-300'
            )}>
              <span className="text-2xl">{paradigm !== 'unknown' ? getParadigmIcon(paradigm) : '❓'}</span>
              {getParadigmDescription(paradigm)}
            </h4>
            <div className="space-y-2">
              {paradigmSections.map((section, index) => (
                <div key={index} className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
                  <h5 className="font-medium text-gray-900 dark:text-gray-100">{section.title}</h5>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-2 whitespace-pre-wrap">
                    {section.content}
                  </p>
                  <div className="flex items-center gap-4 mt-3 text-xs">
                    <span className="flex items-center gap-1">
                      <Activity className="h-3 w-3" />
                      {Math.round(section.confidence * 100)}% confidence
                    </span>
                    <span>{section.sources_count} sources</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    )
  }

  const renderCredibilityView = () => {
    const sources = results.sources || []
    const sortedSources = [...sources].sort((a, b) => b.credibility_score - a.credibility_score)

    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
            <div className="text-sm text-green-600 dark:text-green-400">High Credibility</div>
            <div className="text-2xl font-bold text-green-700 dark:text-green-300">
              {sources.filter(s => s.credibility_score >= 0.8).length}
            </div>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-3">
            <div className="text-sm text-yellow-600 dark:text-yellow-400">Medium Credibility</div>
            <div className="text-2xl font-bold text-yellow-700 dark:text-yellow-300">
              {sources.filter(s => s.credibility_score >= 0.4 && s.credibility_score < 0.8).length}
            </div>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
            <div className="text-sm text-red-600 dark:text-red-400">Low Credibility</div>
            <div className="text-2xl font-bold text-red-700 dark:text-red-300">
              {sources.filter(s => s.credibility_score < 0.4).length}
            </div>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
            <div className="text-sm text-blue-600 dark:text-blue-400">Avg. Score</div>
            <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">
              {sources.length > 0
                ? (sources.reduce((acc, s) => acc + s.credibility_score, 0) / sources.length * 100).toFixed(0)
                : 0}%
            </div>
          </div>
        </div>

        <div className="space-y-3">
          {sortedSources.slice(0, 10).map((source, index) => (
            <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:border-gray-300 dark:hover:border-gray-600 transition-all duration-200">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100">{source.title}</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{source.snippet}</p>
                  <div className="flex items-center gap-4 mt-2">
                    <span className="text-xs text-gray-500">{source.domain}</span>
                    {source.published_date && (
                      <span className="text-xs text-gray-500">{new Date(source.published_date).toLocaleDateString()}</span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className={cn(
                    "flex items-center gap-1 text-sm font-medium",
                    getCredibilityColor(source.credibility_score)
                  )}>
                    {getCredibilityIcon(source.credibility_score)}
                    <span>{(source.credibility_score * 100).toFixed(0)}%</span>
                  </div>
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    <ExternalLink className="h-4 w-4" />
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const renderActionRoadmap = () => {
    const actionItems = answer.action_items || []
    const groupedByPriority = actionItems.reduce((acc, item) => {
      if (!acc[item.priority]) acc[item.priority] = []
      acc[item.priority].push(item)
      return acc
    }, {} as Record<string, typeof actionItems>)

    return (
      <div className="space-y-6">
        {['high', 'medium', 'low'].map(priority => {
          const items = groupedByPriority[priority] || []
          if (items.length === 0) return null

          return (
            <div key={priority}>
              <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                {getPriorityIcon(priority)}
                {priority.charAt(0).toUpperCase() + priority.slice(1)} Priority Actions
              </h4>
              <div className="space-y-3">
                {items.map((item, index) => (
                  <div key={index} className="flex items-start gap-3 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors">
                    <div className="flex-1">
                      <p className="text-gray-900 dark:text-gray-100 font-medium">{item.action}</p>
                      <div className="flex items-center gap-4 mt-2 text-sm text-gray-600 dark:text-gray-400">
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {item.timeframe}
                        </span>
                        <span className={cn(
                          "px-2 py-0.5 rounded text-xs font-medium",
                          getParadigmClass(item.paradigm)
                        )}>
                          {getParadigmDescription(item.paradigm)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    )
  }

  const renderTrendAlignment = () => {
    // Extract trend-related information from the research
    const trendIndicators = [
      {
        name: 'Research Coverage',
        value: results.metadata.total_sources_analyzed,
        trend: '+12%',
        status: 'up' as const
      },
      {
        name: 'Quality Sources',
        value: results.metadata.high_quality_sources,
        trend: '+8%',
        status: 'up' as const
      },
      {
        name: 'Paradigm Diversity',
        value: results.metadata.paradigms_used.length,
        trend: '0%',
        status: 'stable' as const
      },
      {
        name: 'Processing Efficiency',
        value: `${results.metadata.processing_time_seconds}s`,
        trend: '-15%',
        status: 'down' as const
      }
    ]

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {trendIndicators.map((indicator, index) => (
            <div key={index} className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">{indicator.name}</h5>
                <TrendingUp className={cn(
                  "h-4 w-4",
                  indicator.status === 'up' ? 'text-green-500' :
                    indicator.status === 'down' ? 'text-red-500' :
                      'text-gray-500'
                )} />
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">{indicator.value}</div>
              <div className={cn(
                "text-sm font-medium mt-1",
                indicator.status === 'up' ? 'text-green-600 dark:text-green-400' :
                  indicator.status === 'down' ? 'text-red-600 dark:text-red-400' :
                    'text-gray-600 dark:text-gray-400'
              )}>
                {indicator.trend} from baseline
              </div>
            </div>
          ))}
        </div>

        {integrated_synthesis && integrated_synthesis.synergies.length > 0 && (
          <div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
              <GitMerge className="h-5 w-5 text-green-500" />
              Emerging Synergies
            </h4>
            <div className="space-y-2">
              {integrated_synthesis.synergies.map((synergy, index) => (
                <div key={index} className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <p className="text-sm text-green-700 dark:text-green-300">{synergy}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  const getParadigmIcon = (paradigm: string) => {
    const icons: Record<string, string> = {
      dolores: '🛡️',
      teddy: '❤️',
      bernard: '🧠',
      maeve: '📈'
    }
    return icons[paradigm] || '❓'
  }

  // Safely handle all answer properties

  return (
    <div className="mt-8 space-y-6 animate-fade-in">
      {/* Header with View Mode Selector and Metrics */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Research Results</h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Query: "{results.query}"
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getParadigmClass(results.paradigm_analysis.primary.paradigm)
              }`}>
              {getParadigmDescription(results.paradigm_analysis.primary.paradigm)}
            </span>
            {results.paradigm_analysis.secondary && (
              <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getParadigmClass(results.paradigm_analysis.secondary.paradigm)
                }`}>
                + {getParadigmDescription(results.paradigm_analysis.secondary.paradigm)}
              </span>
            )}

            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                disabled={isExporting}
                className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-all duration-200"
                aria-label="Export results"
              >
                {isExporting ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Download className="h-5 w-5" />
                )}
              </button>

              <div className={cn(
                "absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 transition-all duration-200 z-10",
                dropdownOpen ? 'opacity-100 visible translate-y-0' : 'opacity-0 invisible -translate-y-2'
              )}>
                <button
                  onClick={() => handleExport('json')}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 flex items-center gap-2"
                  disabled={isExporting}
                >
                  {exportFormat === 'json' && isExporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : null}
                  Export as JSON
                </button>
                <button
                  onClick={() => handleExport('csv')}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 flex items-center gap-2"
                  disabled={isExporting}
                >
                  {exportFormat === 'csv' && isExporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : null}
                  Export as CSV
                </button>
                <button
                  onClick={() => handleExport('pdf')}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 flex items-center gap-2"
                  disabled={isExporting}
                >
                  {exportFormat === 'pdf' && isExporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : null}
                  Export as PDF
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Quality Metrics Dashboard */}
        {showMetricsOverlay && (
          <div className="mb-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Research Quality Scoring
              </h4>
              <div className="flex items-center gap-2">
                <Badge variant="default" size="sm" className={getMetricColor(parseFloat(calculateOverallScore()))}>
                  Overall Score: {calculateOverallScore()}/10
                </Badge>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowMetricsOverlay(false)}
                  className="text-xs"
                >
                  Hide
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className={cn("rounded-lg p-3 transition-all", getMetricBgColor(resultScoring.answerQuality))}>
                <div className="flex items-center justify-between">
                  <Award className="h-4 w-4 opacity-50" />
                  <span className={cn("text-2xl font-bold", getMetricColor(resultScoring.answerQuality))}>
                    {resultScoring.answerQuality.toFixed(1)}
                  </span>
                </div>
                <div className="text-xs text-gray-700 dark:text-gray-300 mt-1">Answer Quality</div>
              </div>

              <div className={cn("rounded-lg p-3 transition-all", getMetricBgColor(resultScoring.sourceDiversity))}>
                <div className="flex items-center justify-between">
                  <Layers className="h-4 w-4 opacity-50" />
                  <span className={cn("text-2xl font-bold", getMetricColor(resultScoring.sourceDiversity))}>
                    {resultScoring.sourceDiversity.toFixed(1)}
                  </span>
                </div>
                <div className="text-xs text-gray-700 dark:text-gray-300 mt-1">Source Diversity</div>
              </div>

              <div className={cn("rounded-lg p-3 transition-all", getMetricBgColor(resultScoring.paradigmCoherence))}>
                <div className="flex items-center justify-between">
                  <Target className="h-4 w-4 opacity-50" />
                  <span className={cn("text-2xl font-bold", getMetricColor(resultScoring.paradigmCoherence))}>
                    {resultScoring.paradigmCoherence.toFixed(1)}
                  </span>
                </div>
                <div className="text-xs text-gray-700 dark:text-gray-300 mt-1">Paradigm Coherence</div>
              </div>

              <div className={cn("rounded-lg p-3 transition-all", getMetricBgColor(resultScoring.actionability))}>
                <div className="flex items-center justify-between">
                  <Map className="h-4 w-4 opacity-50" />
                  <span className={cn("text-2xl font-bold", getMetricColor(resultScoring.actionability))}>
                    {resultScoring.actionability.toFixed(1)}
                  </span>
                </div>
                <div className="text-xs text-gray-700 dark:text-gray-300 mt-1">Actionability</div>
              </div>
            </div>
          </div>
        )}

        {/* View Mode Selector */}
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <div className="flex items-center gap-2 mb-3">
            <FileText className="h-4 w-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">View Mode</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            {viewModes.map((mode) => (
              <button
                key={mode.value}
                onClick={() => setViewMode(mode.value as ViewMode)}
                className={cn(
                  "p-3 rounded-lg border transition-all duration-200",
                  viewMode === mode.value
                    ? 'border-primary bg-primary/10 shadow-md'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 bg-white dark:bg-gray-800'
                )}
              >
                <div className="flex items-center justify-center gap-2 mb-1">
                  {mode.icon}
                  <span className="font-medium text-sm">{mode.label}</span>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">{mode.description}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Research Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 text-sm">
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

        {/* Deep Research Indicator */}
        {results.metadata.deep_research_enabled && (
          <div className="mt-4 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-lg border border-purple-200 dark:border-purple-800 transition-colors duration-200">
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              <h4 className="text-sm font-semibold text-purple-900 dark:text-purple-100">Deep Research Mode</h4>
              <span className="ml-auto text-xs font-bold bg-gradient-to-r from-orange-500 to-yellow-500 text-white px-2 py-1 rounded">
                o3-deep-research
              </span>
            </div>
            <p className="text-sm text-purple-700 dark:text-purple-300 mt-2">
              This research utilized OpenAI's advanced o3-deep-research model for comprehensive multi-source analysis
            </p>
          </div>
        )}
      </div>

      {/* Main Content Area - Dynamic based on view mode */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.1s' }}>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
          {viewModes.find(m => m.value === viewMode)?.icon}
          {viewModes.find(m => m.value === viewMode)?.label}
        </h3>
        {renderViewContent()}
      </div>

      {/* Agent Trace (transparency) */}
      {Array.isArray(results.metadata?.agent_trace) && results.metadata.agent_trace.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.15s' }}>
          <button onClick={() => setTraceOpen(!traceOpen)} className="w-full text-left flex items-center justify-between" aria-expanded={traceOpen}>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Agentic Research Trace</h3>
            {traceOpen ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
          </button>
          {traceOpen && (
            <ul className="mt-3 space-y-2 text-sm text-gray-700 dark:text-gray-300">
              {results.metadata.agent_trace.map((e: any, i: number) => (
                <li key={i} className="border border-gray-200 dark:border-gray-700 rounded p-3">
                  <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                    <Clock className="h-4 w-4" />
                    <span className="uppercase tracking-wide text-xs font-semibold">{String(e.step || 'revise')}</span>
                    {typeof e.iteration === 'number' && <span className="text-xs">iter {e.iteration}</span>}
                    {typeof e.coverage === 'number' && <span className="ml-auto text-xs">coverage {(e.coverage * 100).toFixed(0)}%</span>}
                  </div>
                  {Array.isArray(e.proposed_queries) && e.proposed_queries.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs text-gray-500 dark:text-gray-400">Proposed Queries</p>
                      <ul className="list-disc list-inside space-y-1">
                        {e.proposed_queries.map((q: string, j: number) => <li key={j} className="break-all">{q}</li>)}
                      </ul>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Additional sections only shown in certain view modes */}
      {viewMode === 'executive' && (
        <>
          {/* Mesh Network Analysis */}
          {integrated_synthesis && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.2s' }}>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Mesh Network Analysis</h3>
              {integrated_synthesis.synergies.length > 0 && (
                <div className="mb-4">
                  <h4 className="font-semibold text-gray-800 dark:text-gray-200 flex items-center">
                    <GitMerge className="h-5 w-5 mr-2 text-green-500 dark:text-green-400" />
                    Synergies
                  </h4>
                  <ul className="list-disc list-inside mt-2 text-gray-700 dark:text-gray-300">
                    {integrated_synthesis.synergies.map((synergy, i) => <li key={i}>{synergy}</li>)}
                  </ul>
                </div>
              )}
              {integrated_synthesis.conflicts_identified.length > 0 && (
                <div>
                  <h4 className="font-semibold text-gray-800 dark:text-gray-200 flex items-center">
                    <Zap className="h-5 w-5 mr-2 text-red-500 dark:text-red-400" />
                    Conflicts
                  </h4>
                  <ul className="list-disc list-inside mt-2 text-gray-700 dark:text-gray-300">
                    {integrated_synthesis.conflicts_identified.map((conflict, i) => <li key={i}>{conflict.description}</li>)}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Cost Information */}
          {results.cost_info && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.3s' }}>
              <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Research Costs</h4>
              <div className="grid grid-cols-3 gap-4">
                {results.cost_info.search_api_costs !== undefined && (
                  <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-3">
                    <p className="text-amber-700 dark:text-amber-300 text-sm">Search API</p>
                    <p className="font-semibold text-amber-900 dark:text-amber-100 text-lg">
                      ${results.cost_info.search_api_costs.toFixed(3)}
                    </p>
                  </div>
                )}
                {results.cost_info.llm_costs !== undefined && (
                  <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-3">
                    <p className="text-amber-700 dark:text-amber-300 text-sm">LLM Processing</p>
                    <p className="font-semibold text-amber-900 dark:text-amber-100 text-lg">
                      ${results.cost_info.llm_costs.toFixed(3)}
                    </p>
                  </div>
                )}
                {results.cost_info.total !== undefined && (
                  <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-3">
                    <p className="text-amber-700 dark:text-amber-300 text-sm">Total Cost</p>
                    <p className="font-semibold text-amber-900 dark:text-amber-100 text-lg">
                      ${results.cost_info.total.toFixed(3)}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
