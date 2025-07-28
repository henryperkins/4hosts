import React, { useState, useRef, useEffect } from 'react'
import { Download, ExternalLink, Shield, AlertTriangle, ChevronDown, ChevronUp, Zap, GitMerge, Loader2, CheckCircle, AlertCircle, Clock } from 'lucide-react'
import toast from 'react-hot-toast'
import api from '../services/api'
import type { ResearchResult, AnswerSection } from '../types'
import { getParadigmClass, getParadigmDescription } from '../constants/paradigm'

interface ResultsDisplayEnhancedProps {
  results: ResearchResult
}


export const ResultsDisplayEnhanced: React.FC<ResultsDisplayEnhancedProps> = ({ results }) => {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set([0]))
  const [isExporting, setIsExporting] = useState(false)
  const [exportFormat, setExportFormat] = useState<'json' | 'pdf' | 'markdown' | null>(null)
  const [showAllCitations, setShowAllCitations] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const answer = results.integrated_synthesis ? results.integrated_synthesis.primary_answer : results.answer;
  const { integrated_synthesis } = results;

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

  const handleExport = async (format: 'json' | 'pdf' | 'markdown') => {
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

  const displayedCitations = showAllCitations
    ? answer.citations
    : answer.citations.slice(0, 5)

  const allSections = integrated_synthesis?.secondary_perspective
    ? [...answer.sections, integrated_synthesis.secondary_perspective]
    : answer.sections;

  return (
    <div className="mt-8 space-y-6 animate-fade-in">
      {/* Summary Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Research Results</h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Query: "{results.query}"
            </p>
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
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Download className="h-5 w-5" />
                )}
              </button>

              <div className={`absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 transition-all duration-200 z-10 ${
                dropdownOpen ? 'opacity-100 visible translate-y-0' : 'opacity-0 invisible -translate-y-2'
              }`}>
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
                  onClick={() => handleExport('markdown')}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 flex items-center gap-2"
                  disabled={isExporting}
                >
                  {exportFormat === 'markdown' && isExporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : null}
                  Export as Markdown
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

        <div className="prose dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300">{integrated_synthesis ? integrated_synthesis.integrated_summary : answer.summary}</p>
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

        {/* Context Engineering Info */}
        {results.paradigm_analysis.context_engineering && (
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 transition-colors duration-200">
            <h4 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">Context Engineering Pipeline</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <p className="text-blue-700 dark:text-blue-300">Compression Ratio</p>
                <p className="font-semibold text-blue-900 dark:text-blue-100">{(results.paradigm_analysis.context_engineering.compression_ratio * 100).toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-blue-700 dark:text-blue-300">Token Budget</p>
                <p className="font-semibold text-blue-900 dark:text-blue-100">{results.paradigm_analysis.context_engineering.token_budget.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-blue-700 dark:text-blue-300">Search Queries</p>
                <p className="font-semibold text-blue-900 dark:text-blue-100">{results.paradigm_analysis.context_engineering.search_queries_count}</p>
              </div>
              <div>
                <p className="text-blue-700 dark:text-blue-300">Isolation Strategy</p>
                <p className="font-semibold capitalize text-blue-900 dark:text-blue-100">{results.paradigm_analysis.context_engineering.isolation_strategy}</p>
              </div>
            </div>
          </div>
        )}

        {/* Cost Information */}
        {results.cost_info && (
          <div className="mt-4 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800 transition-colors duration-200">
            <h4 className="text-sm font-semibold text-amber-900 dark:text-amber-100 mb-2">Research Costs</h4>
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

      {/* Mesh Network Analysis */}
      {integrated_synthesis && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.1s' }}>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Mesh Network Analysis</h3>
            {integrated_synthesis.synergies.length > 0 && (
                <div className="mb-4">
                    <h4 className="font-semibold text-gray-800 dark:text-gray-200 flex items-center"><GitMerge className="h-5 w-5 mr-2 text-green-500 dark:text-green-400" />Synergies</h4>
                    <ul className="list-disc list-inside mt-2 text-gray-700 dark:text-gray-300">
                        {integrated_synthesis.synergies.map((synergy, i) => <li key={i}>{synergy}</li>)}
                    </ul>
                </div>
            )}
            {integrated_synthesis.conflicts_identified.length > 0 && (
                <div>
                    <h4 className="font-semibold text-gray-800 dark:text-gray-200 flex items-center"><Zap className="h-5 w-5 mr-2 text-red-500 dark:text-red-400" />Conflicts</h4>
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
                <ChevronUp className="h-5 w-5 text-gray-400 transition-transform duration-200" />
              ) : (
                <ChevronDown className="h-5 w-5 text-gray-400 transition-transform duration-200" />
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

      {/* Action Items */}
      {answer.action_items.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.3s' }}>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Action Items</h3>
          <div className="space-y-3">
            {answer.action_items.map((item, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg transition-colors duration-200 hover:bg-gray-100 dark:hover:bg-gray-600">
                <div className="mt-0.5">
                  {getPriorityIcon(item.priority)}
                </div>
                <div className="flex-1">
                  <p className="text-gray-900 dark:text-gray-100 font-medium">{item.action}</p>
                  <div className="flex items-center gap-4 mt-1 text-sm text-gray-600 dark:text-gray-400">
                    <span>Timeframe: {item.timeframe}</span>
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

      {/* Sources Overview */}
      {results.sources && results.sources.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.4s' }}>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Research Sources</h3>
          <div className="grid gap-3">
            {results.sources.slice(0, 5).map((source, index) => (
              <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:border-gray-300 dark:hover:border-gray-600 transition-all duration-200 hover:shadow-md">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900 dark:text-gray-100">{source.title}</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{source.snippet}</p>
                    <div className="flex items-center gap-4 mt-2">
                      <span className="text-xs text-gray-500 dark:text-gray-500">{source.domain}</span>
                      {source.published_date && (
                        <span className="text-xs text-gray-500 dark:text-gray-500">{new Date(source.published_date).toLocaleDateString()}</span>
                      )}
                      <div className={`flex items-center gap-1 text-sm ${getCredibilityColor(source.credibility_score)}`}>
                        {getCredibilityIcon(source.credibility_score)}
                        <span className="font-medium">{(source.credibility_score * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-4 p-2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
                    aria-label="Open source in new tab"
                  >
                    <ExternalLink className="h-4 w-4" />
                  </a>
                </div>
              </div>
            ))}
          </div>
          {results.sources.length > 5 && (
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
              Showing 5 of {results.sources.length} sources analyzed
            </p>
          )}
        </div>
      )}

      {/* Citations with Credibility */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200 animate-slide-up" style={{ animationDelay: '0.5s' }}>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Answer Citations</h3>
        <div className="space-y-3">
          {displayedCitations.map((citation) => (
            <div key={citation.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:border-gray-300 dark:hover:border-gray-600 transition-all duration-200 hover:shadow-md">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100">{citation.title}</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{citation.source}</p>

                  <div className="flex items-center gap-4 mt-2">
                    <div className={`flex items-center gap-1 text-sm ${getCredibilityColor(citation.credibility_score)}`}>
                      {getCredibilityIcon(citation.credibility_score)}
                      <span className="font-medium">
                        {citation.credibility_score * 100}% credibility
                      </span>
                    </div>

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
                  <ExternalLink className="h-4 w-4" />
                </a>
              </div>
            </div>
          ))}
        </div>

        {answer.citations.length > 5 && (
          <button
            onClick={() => setShowAllCitations(!showAllCitations)}
            className="mt-4 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium transition-colors duration-200 focus:outline-none focus:underline"
          >
            {showAllCitations ? 'Show less' : `Show all ${answer.citations.length} citations`}
          </button>
        )}
      </div>
    </div>
  )
}
