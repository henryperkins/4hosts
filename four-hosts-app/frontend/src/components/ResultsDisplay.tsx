import { useState, useEffect } from 'react'
import type { ResearchResult } from '../types'
import { CheckCircle, AlertCircle, Clock, ExternalLink, TrendingUp, ChevronDown, Shield } from 'lucide-react'

interface ResultsDisplayProps {
  results: ResearchResult
}

const paradigmColors = {
  dolores: 'text-red-700 bg-red-50 border-red-200 dark:text-red-300 dark:bg-red-900/20 dark:border-red-800',
  teddy: 'text-orange-700 bg-orange-50 border-orange-200 dark:text-orange-300 dark:bg-orange-900/20 dark:border-orange-800',
  bernard: 'text-blue-700 bg-blue-50 border-blue-200 dark:text-blue-300 dark:bg-blue-900/20 dark:border-blue-800',
  maeve: 'text-green-700 bg-green-50 border-green-200 dark:text-green-300 dark:bg-green-900/20 dark:border-green-800'
}

const paradigmDescriptions = {
  dolores: 'Truth & Justice',
  teddy: 'Care & Support',
  bernard: 'Analysis & Logic',
  maeve: 'Strategy & Power'
}

function ResultsDisplay({ results }: ResultsDisplayProps) {
  const [visibleSections, setVisibleSections] = useState<number[]>([])
  const [expandedSections, setExpandedSections] = useState<number[]>([])

  useEffect(() => {
    // Animate sections appearing one by one
    results.answer.sections.forEach((_, index) => {
      setTimeout(() => {
        setVisibleSections(prev => [...prev, index])
      }, index * 150)
    })
  }, [results.answer.sections])

  const toggleSection = (index: number) => {
    setExpandedSections(prev => 
      prev.includes(index) 
        ? prev.filter(i => i !== index)
        : [...prev, index]
    )
  }

  return (
    <div className="mt-8 space-y-6 animate-fade-in">
      <div className="bg-white dark:bg-gray-800 shadow-lg rounded-lg p-6 hover-lift animate-slide-up transition-colors duration-200">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4">Research Results</h2>
        
        <div className="prose max-w-none animate-fade-in" style={{ animationDelay: '0.1s' }}>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2 flex items-center">
            <TrendingUp className="h-5 w-5 mr-2 text-blue-600 dark:text-blue-400" />
            Summary
          </h3>
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">{results.answer.summary}</p>
        </div>
        
        <div className="mt-6 space-y-4">
          {results.answer.sections.map((section, index) => (
            <div 
              key={index} 
              className={`border-2 rounded-lg overflow-hidden transition-all duration-300 ${
                visibleSections.includes(index) 
                  ? 'opacity-100 translate-y-0' 
                  : 'opacity-0 translate-y-4'
              } ${paradigmColors[section.paradigm]} hover:shadow-lg`}
              style={{ 
                animationDelay: `${index * 0.1}s`,
                transitionDelay: `${index * 0.1}s`
              }}
            >
              <button 
                className="w-full p-4 text-left focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 dark:focus:ring-blue-400"
                onClick={() => toggleSection(index)}
                aria-expanded={expandedSections.includes(index)}
                aria-controls={`section-content-${index}`}
              >
                <h4 className="font-medium mb-2 flex items-center justify-between">
                  <span className="flex items-center">
                    <span className="text-xs font-medium px-2 py-1 rounded-full mr-2 bg-current/10">
                      {paradigmDescriptions[section.paradigm]}
                    </span>
                    {section.title}
                  </span>
                  <ChevronDown className={`h-5 w-5 transform transition-transform duration-200 ${
                    expandedSections.includes(index) ? 'rotate-180' : ''
                  }`} />
                </h4>
                <div 
                  id={`section-content-${index}`}
                  className={`text-sm transition-all duration-300 overflow-hidden ${
                    expandedSections.includes(index) 
                      ? 'max-h-[500px]' 
                      : 'max-h-20'
                  }`}
                >
                  <p className={expandedSections.includes(index) ? '' : 'line-clamp-3'}>
                    {section.content}
                  </p>
                </div>
                <div className="mt-3 flex items-center justify-between text-xs">
                  <span className="flex items-center space-x-3 opacity-75">
                    <span>{section.sources_count} sources analyzed</span>
                    <span>•</span>
                    <span className="flex items-center">
                      <div className={`h-2 w-20 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mr-1`}>
                        <div 
                          className={`h-full bg-current transition-all duration-1000 ease-out`}
                          style={{ 
                            width: `${section.confidence * 100}%`,
                            transitionDelay: `${index * 0.2}s`
                          }}
                        />
                      </div>
                      {(section.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </span>
                </div>
              </button>
            </div>
          ))}
        </div>
      </div>
      
      {results.answer.action_items.length > 0 && (
        <div className="bg-white dark:bg-gray-800 shadow-lg rounded-lg p-6 hover-lift animate-slide-up transition-colors duration-200" style={{ animationDelay: '0.3s' }}>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4 flex items-center">
            <CheckCircle className="h-5 w-5 mr-2 text-green-600 dark:text-green-400" />
            Action Items
          </h3>
          <div className="space-y-3">
            {results.answer.action_items.map((item, index) => (
              <div 
                key={index} 
                className="flex items-start p-3 rounded-lg transition-all duration-200 hover:bg-gray-50 dark:hover:bg-gray-700 animate-slide-up"
                style={{ animationDelay: `${0.4 + index * 0.1}s` }}
              >
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium transition-all duration-200 hover:scale-110 ${
                  item.priority === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' :
                  item.priority === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300' :
                  'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                }`}>
                  {item.priority === 'high' ? <AlertCircle className="h-3 w-3 mr-1" /> : 
                   item.priority === 'medium' ? <Clock className="h-3 w-3 mr-1" /> : 
                   <CheckCircle className="h-3 w-3 mr-1" />} 
                  {item.priority}
                </span>
                <div className="ml-3 flex-1">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{item.action}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center mt-1">
                    <Clock className="h-3 w-3 mr-1" />
                    Timeframe: {item.timeframe}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div className="bg-white dark:bg-gray-800 shadow-lg rounded-lg p-6 hover-lift animate-slide-up transition-colors duration-200" style={{ animationDelay: '0.5s' }}>
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4 flex items-center">
            <Shield className="h-5 w-5 mr-2 text-blue-600 dark:text-blue-400" />
            Sources & Citations
          </h3>
        <div className="space-y-2">
          {results.answer.citations.map((citation, index) => (
            <div 
              key={citation.id} 
              className="border-l-4 border-gray-200 dark:border-gray-700 pl-4 py-2 transition-all duration-200 hover:border-blue-500 dark:hover:border-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-r animate-slide-up"
              style={{ animationDelay: `${0.6 + index * 0.05}s` }}
            >
              <a
                href={citation.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 flex items-center group focus:outline-none focus:underline"
              >
                {citation.title}
                <ExternalLink className="h-3 w-3 ml-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
              </a>
              <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center mt-1">
                <span className="font-medium">{citation.source}</span>
                <span className="mx-2">•</span>
                <span className="flex items-center">
                  <Shield className={`h-3 w-3 mr-1 ${
                    citation.credibility_score >= 0.8 ? 'text-green-600 dark:text-green-400' :
                    citation.credibility_score >= 0.6 ? 'text-yellow-600 dark:text-yellow-400' :
                    'text-red-600 dark:text-red-400'
                  }`} aria-label="Credibility score" />
                  <span className={`font-medium ${
                    citation.credibility_score >= 0.8 ? 'text-green-600 dark:text-green-400' :
                    citation.credibility_score >= 0.6 ? 'text-yellow-600 dark:text-yellow-400' :
                    'text-red-600 dark:text-red-400'
                  }`}>
                    {(citation.credibility_score * 100).toFixed(0)}%
                  </span>
                </span>
              </p>
            </div>
          ))}
        </div>
        
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 animate-fade-in" style={{ animationDelay: '0.8s' }}>
          <div className="flex flex-wrap gap-4 text-xs text-gray-500 dark:text-gray-400">
            <span className="flex items-center">
              <span className="inline-block w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full mr-1 animate-pulse"></span>
              {results.metadata.total_sources_analyzed} sources analyzed
            </span>
            <span className="flex items-center">
              <span className="inline-block w-2 h-2 bg-green-500 dark:bg-green-400 rounded-full mr-1"></span>
              {results.metadata.high_quality_sources} high-quality sources
            </span>
            <span className="flex items-center">
              <span className="inline-block w-2 h-2 bg-yellow-500 dark:bg-yellow-400 rounded-full mr-1"></span>
              {results.metadata.search_queries_executed} searches executed
            </span>
            <span className="flex items-center">
              <span className="inline-block w-2 h-2 bg-purple-500 dark:bg-purple-400 rounded-full mr-1"></span>
              {results.metadata.processing_time_seconds.toFixed(1)}s processing time
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResultsDisplay