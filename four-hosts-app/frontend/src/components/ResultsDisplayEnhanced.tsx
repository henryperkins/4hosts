import React, { useState } from 'react'
import { Download, ExternalLink, Shield, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react'
import toast from 'react-hot-toast'
import api from '../services/api'
import type { ResearchResult } from '../types'

interface ResultsDisplayEnhancedProps {
  results: ResearchResult
}

const paradigmColors = {
  dolores: 'bg-red-100 text-red-800 border-red-200',
  teddy: 'bg-blue-100 text-blue-800 border-blue-200',
  bernard: 'bg-green-100 text-green-800 border-green-200',
  maeve: 'bg-purple-100 text-purple-800 border-purple-200',
}

const paradigmDescriptions = {
  dolores: 'Truth & Justice',
  teddy: 'Care & Support',
  bernard: 'Analysis & Logic',
  maeve: 'Strategy & Power',
}

export const ResultsDisplayEnhanced: React.FC<ResultsDisplayEnhancedProps> = ({ results }) => {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set([0]))
  const [isExporting, setIsExporting] = useState(false)
  const [showAllCitations, setShowAllCitations] = useState(false)

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
    try {
      const blob = await api.exportResearch(results.research_id, format)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `research-${results.research_id}.${format}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast.success(`Exported as ${format.toUpperCase()}`)
    } catch (error) {
      toast.error('Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  const getCredibilityColor = (score: number) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 60) return 'text-yellow-600'
    if (score >= 40) return 'text-orange-600'
    return 'text-red-600'
  }

  const getCredibilityIcon = (score: number) => {
    if (score >= 80) return <Shield className="h-4 w-4" />
    if (score >= 40) return <AlertTriangle className="h-4 w-4" />
    return <AlertTriangle className="h-4 w-4" />
  }

  const displayedCitations = showAllCitations 
    ? results.answer.citations 
    : results.answer.citations.slice(0, 5)

  return (
    <div className="mt-8 space-y-6">
      {/* Summary Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Research Results</h2>
            <p className="text-sm text-gray-600 mt-1">
              Query: "{results.query}"
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <span className={`px-3 py-1 rounded-full text-sm font-medium border ${
              paradigmColors[results.paradigm_analysis.primary.paradigm]
            }`}>
              {paradigmDescriptions[results.paradigm_analysis.primary.paradigm]}
            </span>
            
            <div className="relative group">
              <button
                onClick={() => handleExport('json')}
                disabled={isExporting}
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                title="Export results"
              >
                <Download className="h-5 w-5" />
              </button>
              
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                <button
                  onClick={() => handleExport('json')}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  Export as JSON
                </button>
                <button
                  onClick={() => handleExport('markdown')}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  Export as Markdown
                </button>
                <button
                  onClick={() => handleExport('pdf')}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  Export as PDF
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="prose max-w-none">
          <p className="text-gray-700">{results.answer.summary}</p>
        </div>

        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-gray-600">Sources Analyzed</p>
            <p className="font-semibold text-lg">{results.metadata.total_sources_analyzed}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-gray-600">High Quality</p>
            <p className="font-semibold text-lg">{results.metadata.high_quality_sources}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-gray-600">Processing Time</p>
            <p className="font-semibold text-lg">{results.metadata.processing_time_seconds}s</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-gray-600">Paradigms Used</p>
            <p className="font-semibold text-lg">{results.metadata.paradigms_used.length}</p>
          </div>
        </div>
      </div>

      {/* Detailed Sections */}
      <div className="space-y-4">
        {results.answer.sections.map((section, index) => (
          <div key={index} className="bg-white rounded-lg shadow-md overflow-hidden">
            <button
              onClick={() => toggleSection(index)}
              className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <h3 className="text-lg font-semibold text-gray-900">{section.title}</h3>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  paradigmColors[section.paradigm]
                }`}>
                  {paradigmDescriptions[section.paradigm]}
                </span>
                <span className="text-sm text-gray-600">
                  {section.sources_count} sources • {Math.round(section.confidence * 100)}% confidence
                </span>
              </div>
              {expandedSections.has(index) ? (
                <ChevronUp className="h-5 w-5 text-gray-400" />
              ) : (
                <ChevronDown className="h-5 w-5 text-gray-400" />
              )}
            </button>
            
            {expandedSections.has(index) && (
              <div className="px-6 pb-4 border-t">
                <div className="prose max-w-none mt-4">
                  <p className="text-gray-700 whitespace-pre-wrap">{section.content}</p>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Action Items */}
      {results.answer.action_items.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Action Items</h3>
          <div className="space-y-3">
            {results.answer.action_items.map((item, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                <div className={`mt-0.5 w-2 h-2 rounded-full ${
                  item.priority === 'high' ? 'bg-red-500' :
                  item.priority === 'medium' ? 'bg-yellow-500' :
                  'bg-green-500'
                }`} />
                <div className="flex-1">
                  <p className="text-gray-900 font-medium">{item.action}</p>
                  <div className="flex items-center gap-4 mt-1 text-sm text-gray-600">
                    <span>Timeframe: {item.timeframe}</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      paradigmColors[item.paradigm]
                    }`}>
                      {paradigmDescriptions[item.paradigm]}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Citations with Credibility */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Sources & Citations</h3>
        <div className="space-y-3">
          {displayedCitations.map((citation) => (
            <div key={citation.id} className="border border-gray-200 rounded-lg p-4 hover:border-gray-300 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900">{citation.title}</h4>
                  <p className="text-sm text-gray-600 mt-1">{citation.source}</p>
                  
                  <div className="flex items-center gap-4 mt-2">
                    <div className={`flex items-center gap-1 text-sm ${getCredibilityColor(citation.credibility_score)}`}>
                      {getCredibilityIcon(citation.credibility_score)}
                      <span className="font-medium">
                        {citation.credibility_score}% credibility
                      </span>
                    </div>
                    
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      paradigmColors[citation.paradigm_alignment]
                    }`}>
                      {paradigmDescriptions[citation.paradigm_alignment]}
                    </span>
                  </div>
                </div>
                
                <a
                  href={citation.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-4 p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <ExternalLink className="h-4 w-4" />
                </a>
              </div>
            </div>
          ))}
        </div>
        
        {results.answer.citations.length > 5 && (
          <button
            onClick={() => setShowAllCitations(!showAllCitations)}
            className="mt-4 text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            {showAllCitations ? 'Show less' : `Show all ${results.answer.citations.length} citations`}
          </button>
        )}
      </div>
    </div>
  )
}