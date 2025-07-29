import { useState, useEffect } from 'react'
import type { ResearchResult } from '../types'
import { Clock, ExternalLink, TrendingUp, ChevronDown, Shield } from 'lucide-react'
import { getParadigmClass, getParadigmDescription } from '../constants/paradigm'
import { Card, CardHeader, CardContent } from './ui/Card'
import { Badge } from './ui/Badge'
import { StatusIcon } from './ui/StatusIcon'
import { ProgressBar } from './ui/ProgressBar'

interface ResultsDisplayProps {
  results: ResearchResult
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
      <Card className="hover-lift animate-slide-up">
        <CardHeader icon={TrendingUp}>
          <h2 className="text-xl font-semibold">Research Results</h2>
        </CardHeader>
        <CardContent>
          <div className="prose max-w-none animate-fade-in delay-100">
            <h3 className="text-lg font-medium text-text mb-2">
              Summary
            </h3>
            <p className="text-text-muted leading-relaxed">{results.answer.summary}</p>
          </div>
        
          <div className="mt-6 space-y-4">
            {results.answer.sections.map((section, index) => (
              <div 
                key={index} 
                className={`border-2 rounded-lg overflow-hidden transition-all duration-300 ${
                  visibleSections.includes(index) 
                    ? 'opacity-100 translate-y-0' 
                    : 'opacity-0 translate-y-4'
                } ${getParadigmClass(section.paradigm)} hover:shadow-lg stagger-delay-${Math.min(index * 100, 300)}`}
              >
                <button 
                  className="w-full p-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500"
                  onClick={() => toggleSection(index)}
                  aria-expanded={expandedSections.includes(index)}
                  aria-controls={`section-content-${index}`}
                >
                  <h4 className="font-medium mb-2 flex items-center justify-between">
                    <span className="flex items-center">
                      <span className="text-xs font-medium px-2 py-1 rounded-full mr-2 bg-current/10">
                        {getParadigmDescription(section.paradigm)}
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
                      <span className="flex items-center gap-2">
                        <ProgressBar
                          value={section.confidence * 100}
                          max={100}
                          size="sm"
                          variant="info"
                          className="w-20"
                        />
                        <span className="text-xs">{(section.confidence * 100).toFixed(0)}% confidence</span>
                      </span>
                    </span>
                  </div>
                </button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
      
      {results.answer.action_items.length > 0 && (
        <Card className="hover-lift animate-slide-up delay-300">
          <CardHeader>
            <h3 className="text-lg font-medium flex items-center gap-2">
              <StatusIcon status="completed" size="sm" />
              Action Items
            </h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {results.answer.action_items.map((item, index) => (
                <div 
                  key={index} 
                  className={`flex items-start p-3 rounded-lg transition-all duration-200 hover:bg-surface-subtle animate-slide-up stagger-delay-${Math.min(index * 100, 300)}`}
                >
                  <Badge
                    variant={item.priority === 'high' ? 'error' : item.priority === 'medium' ? 'warning' : 'success'}
                    size="sm"
                  >
                    {item.priority}
                  </Badge>
                  <div className="ml-3 flex-1">
                    <p className="text-sm font-medium text-text">{item.action}</p>
                    <p className="text-xs text-text-muted flex items-center mt-1">
                      <Clock className="h-3 w-3 mr-1" />
                      Timeframe: {item.timeframe}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
      
      <Card className="hover-lift animate-slide-up delay-500">
        <CardHeader icon={Shield}>
          <h3 className="text-lg font-medium">Sources & Citations</h3>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {results.answer.citations.map((citation, index) => (
              <div 
                key={citation.id} 
                className={`border-l-4 border-border pl-4 py-2 transition-all duration-200 hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-r animate-slide-up stagger-delay-${Math.min(index * 50, 300)}`}
              >
                <a
                  href={citation.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 flex items-center group focus-visible:outline-none focus-visible:underline"
                >
                  {citation.title}
                  <ExternalLink className="h-3 w-3 ml-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                </a>
                <p className="text-xs text-text-muted flex items-center mt-1">
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
          
          <div className="mt-4 pt-4 border-t border-border animate-fade-in delay-800">
            <div className="flex flex-wrap gap-4 text-xs text-text-muted">
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
        </CardContent>
      </Card>
    </div>
  )
}

export default ResultsDisplay