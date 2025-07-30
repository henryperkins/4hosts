import React, { useEffect, useState, useRef } from 'react'
import { Clock, X, Search, Database, Brain, CheckCircle, Zap, TrendingUp, BarChart3, Activity, Target, Award, AlertTriangle } from 'lucide-react'
import { format } from 'date-fns'
import { Button } from './ui/Button'
import { Card } from './ui/Card'
import { StatusBadge, type StatusType } from './ui/StatusIcon'
import { ProgressBar } from './ui/ProgressBar'
import { LoadingSpinner } from './ui/LoadingSpinner'
import { Badge } from './ui/Badge'
import { cn } from '../utils/cn'
import { getParadigmColorValue } from '../constants/paradigm'
import api from '../services/api'

interface ResearchProgressIdeaBrowserProps {
  researchId: string
  onComplete?: () => void
  onCancel?: () => void
}

interface ProgressUpdate {
  status: 'pending' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
  progress?: number
  message?: string
  timestamp: string
}

interface WebSocketData {
  status?: 'pending' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
  progress?: number
  message?: string
  query?: string
  engine?: string
  index?: number
  total?: number
  results_count?: number
  domain?: string
  score?: number
  before_count?: number
  after_count?: number
  removed?: number
  phase?: string
  old_phase?: string
  new_phase?: string
  source?: any
  total_sources?: number
  source_id?: string
  analyzed_count?: number
  duration_seconds?: number
  error?: string
}

interface ResearchPhase {
  name: string
  icon: React.ReactNode
  progress: [number, number]
  isActive?: boolean
  isCompleted?: boolean
  score?: number
  metrics?: {
    accuracy?: number
    time?: string
    efficiency?: number
  }
}

interface ResearchStats {
  sourcesFound: number
  sourcesAnalyzed: number
  searchesCompleted: number
  totalSearches: number
  highQualitySources: number
}

interface QualityMetrics {
  sourceQualityScore: number
  paradigmAlignment: number
  searchEffectiveness: number
  answerConfidence: number
}

interface PhaseScores {
  classification: { accuracy: number; time: string }
  contextEngineering: { compression: number; relevance: number }
  search: { coverage: number; quality: number }
  synthesis: { coherence: number; paradigmFit: number }
}

interface SourcePreview {
  title: string
  domain: string
  snippet?: string
  credibility?: number
}

const getMetricColor = (score: number): string => {
  if (score >= 80) return 'text-green-600 dark:text-green-400'
  if (score >= 60) return 'text-blue-600 dark:text-blue-400'
  if (score >= 40) return 'text-amber-600 dark:text-amber-400'
  return 'text-red-600 dark:text-red-400'
}

const getMetricBgColor = (score: number): string => {
  if (score >= 80) return 'bg-green-100 dark:bg-green-900/30'
  if (score >= 60) return 'bg-blue-100 dark:bg-blue-900/30'
  if (score >= 40) return 'bg-amber-100 dark:bg-amber-900/30'
  return 'bg-red-100 dark:bg-red-900/30'
}

export const ResearchProgressIdeaBrowser: React.FC<ResearchProgressIdeaBrowserProps> = ({ researchId, onComplete, onCancel }) => {
  const [updates, setUpdates] = useState<ProgressUpdate[]>([])
  const [currentStatus, setCurrentStatus] = useState<ProgressUpdate['status']>('pending')
  const [progress, setProgress] = useState(0)
  const [isConnecting, setIsConnecting] = useState(true)
  const [isCancelling, setIsCancelling] = useState(false)
  const [stats, setStats] = useState<ResearchStats>({
    sourcesFound: 0,
    sourcesAnalyzed: 0,
    searchesCompleted: 0,
    totalSearches: 0,
    highQualitySources: 0
  })
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics>({
    sourceQualityScore: 0,
    paradigmAlignment: 0,
    searchEffectiveness: 0,
    answerConfidence: 0
  })
  const [phaseScores, setPhaseScores] = useState<PhaseScores>({
    classification: { accuracy: 0, time: '0s' },
    contextEngineering: { compression: 0, relevance: 0 },
    search: { coverage: 0, quality: 0 },
    synthesis: { coherence: 0, paradigmFit: 0 }
  })
  const [sourcePreviews, setSourcePreviews] = useState<SourcePreview[]>([])
  const [showSourcePreviews, setShowSourcePreviews] = useState(true)
  const [showMetricsDashboard, setShowMetricsDashboard] = useState(true)
  const updatesContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setIsConnecting(true)
      
      api.connectWebSocket(researchId, (message) => {
        setIsConnecting(false)
        const data = message.data as any
        
        let statusUpdate: WebSocketData | undefined
        
        switch (message.type) {
          case 'research_progress':
            statusUpdate = {
              status: data.status,
              progress: data.progress,
              message: data.message || `Phase: ${data.phase || 'processing'}`
            }
            break
          case 'research_phase_change':
            statusUpdate = {
              message: `Phase changed: ${data.old_phase} → ${data.new_phase}`
            }
            updatePhaseScores(data.old_phase, data)
            break
          case 'source_found':
            statusUpdate = {
              message: `Found source: ${data.source?.title || 'New source'} (${data.total_sources} total)`
            }
            if (data.source) {
              setSourcePreviews(prev => [...prev.slice(-4), {
                title: data.source.title,
                domain: data.source.domain,
                snippet: data.source.snippet,
                credibility: data.source.credibility_score
              }])
              updateQualityMetrics('source', data.source.credibility_score)
            }
            break
          case 'source_analyzed':
            statusUpdate = {
              message: `Analyzed source ${data.source_id} (${data.analyzed_count} analyzed)`
            }
            setStats(prev => ({ ...prev, sourcesAnalyzed: data.analyzed_count || prev.sourcesAnalyzed }))
            break
          case 'research_completed':
            statusUpdate = {
              status: 'completed',
              progress: 100,
              message: `Research completed in ${data.duration_seconds}s`
            }
            break
          case 'research_failed':
            statusUpdate = {
              status: 'failed',
              message: `Research failed: ${data.error}`
            }
            break
          case 'research_started':
            statusUpdate = {
              status: 'processing',
              message: `Research started for query: ${data.query}`
            }
            break
          case 'search.started':
            statusUpdate = {
              message: `Searching (${data.index}/${data.total}): ${data.query}`
            }
            setStats(prev => ({ ...prev, totalSearches: data.total || prev.totalSearches }))
            break
          case 'search.completed':
            statusUpdate = {
              message: `Found ${data.results_count} results`
            }
            setStats(prev => ({ 
              ...prev, 
              searchesCompleted: prev.searchesCompleted + 1,
              sourcesFound: prev.sourcesFound + (data.results_count || 0)
            }))
            updateQualityMetrics('search', data.results_count)
            break
          case 'credibility.check':
            statusUpdate = {
              message: `Checking credibility: ${data.domain} (${(data.score * 100).toFixed(0)}%)`
            }
            if (data.score > 0.7) {
              setStats(prev => ({ ...prev, highQualitySources: prev.highQualitySources + 1 }))
            }
            break
          case 'deduplication.progress':
            statusUpdate = {
              message: `Removed ${data.removed} duplicate results`
            }
            break
          default:
            statusUpdate = {
              status: data.status,
              progress: data.progress,
              message: data.message
            }
        }
        
        const update: ProgressUpdate = {
          status: statusUpdate.status || currentStatus,
          progress: statusUpdate.progress,
          message: statusUpdate.message,
          timestamp: new Date().toISOString(),
        }

        setUpdates(prev => {
          const newUpdates = [...prev, update]
          setTimeout(() => {
            if (updatesContainerRef.current) {
              updatesContainerRef.current.scrollTop = updatesContainerRef.current.scrollHeight
            }
          }, 100)
          return newUpdates
        })
        
        if (data.status) {
          setCurrentStatus(data.status)
        }
        
        if (data.progress !== undefined) {
          setProgress(data.progress)
        }

        if (data.status === 'completed' && onComplete) {
          onComplete()
        }
        
        if (data.status === 'cancelled' && onCancel) {
          onCancel()
        }
      })

    return () => {
      api.unsubscribeFromResearch(researchId)
    }
  }, [researchId, currentStatus, onComplete, onCancel])

  const updateQualityMetrics = (type: string, value: any) => {
    setQualityMetrics(prev => {
      const newMetrics = { ...prev }
      
      switch (type) {
        case 'source':
          const credibilityScore = value * 100
          newMetrics.sourceQualityScore = Math.round(
            (prev.sourceQualityScore * stats.sourcesAnalyzed + credibilityScore) / 
            (stats.sourcesAnalyzed + 1)
          )
          break
        case 'search':
          if (value > 0) {
            newMetrics.searchEffectiveness = Math.min(100, 
              Math.round((stats.sourcesFound + value) / (stats.totalSearches * 10) * 100)
            )
          }
          break
        case 'paradigm':
          newMetrics.paradigmAlignment = Math.round(value * 100)
          break
        case 'confidence':
          newMetrics.answerConfidence = Math.round(value * 100)
          break
      }
      
      return newMetrics
    })
  }

  const updatePhaseScores = (phase: string, data: any) => {
    setPhaseScores(prev => {
      const newScores = { ...prev }
      
      switch (phase) {
        case 'classification':
          newScores.classification = {
            accuracy: data.accuracy || 95,
            time: data.time || '0.8s'
          }
          break
        case 'context_engineering':
          newScores.contextEngineering = {
            compression: data.compression || 84,
            relevance: data.relevance || 92
          }
          break
        case 'search':
          newScores.search = {
            coverage: data.coverage || 87,
            quality: data.quality || 91
          }
          break
        case 'synthesis':
          newScores.synthesis = {
            coherence: data.coherence || 94,
            paradigmFit: data.paradigm_fit || 96
          }
          break
      }
      
      return newScores
    })
  }

  const handleCancel = async () => {
    setIsCancelling(true)
    try {
      await api.cancelResearch(researchId)
    } catch (error) {
      console.error('Failed to cancel research:', error)
    } finally {
      setIsCancelling(false)
    }
  }

  const getStatusType = (): StatusType => {
    if (isConnecting) return 'processing'
    
    switch (currentStatus) {
      case 'pending': return 'pending'
      case 'processing':
      case 'in_progress': return 'processing'
      case 'completed': return 'completed'
      case 'failed': return 'failed'
      case 'cancelled': return 'cancelled'
      default: return 'pending'
    }
  }

  const getStatusLabel = () => {
    if (isConnecting) return 'Connecting...'
    return currentStatus.charAt(0).toUpperCase() + currentStatus.slice(1)
  }

  const canCancel = () => {
    return currentStatus === 'processing' || currentStatus === 'in_progress' || currentStatus === 'pending'
  }

  const researchPhases: ResearchPhase[] = [
    {
      name: 'Classification',
      icon: <Brain className="h-4 w-4" />,
      progress: [0, 10],
      isActive: progress >= 0 && progress < 10,
      isCompleted: progress >= 10,
      score: phaseScores.classification.accuracy,
      metrics: {
        accuracy: phaseScores.classification.accuracy,
        time: phaseScores.classification.time
      }
    },
    {
      name: 'Context Engineering',
      icon: <Zap className="h-4 w-4" />,
      progress: [10, 20],
      isActive: progress >= 10 && progress < 20,
      isCompleted: progress >= 20,
      score: (phaseScores.contextEngineering.compression + phaseScores.contextEngineering.relevance) / 2,
      metrics: {
        efficiency: phaseScores.contextEngineering.compression,
        accuracy: phaseScores.contextEngineering.relevance
      }
    },
    {
      name: 'Search & Retrieval',
      icon: <Search className="h-4 w-4" />,
      progress: [20, 60],
      isActive: progress >= 20 && progress < 60,
      isCompleted: progress >= 60,
      score: (phaseScores.search.coverage + phaseScores.search.quality) / 2,
      metrics: {
        efficiency: phaseScores.search.coverage,
        accuracy: phaseScores.search.quality
      }
    },
    {
      name: 'Analysis',
      icon: <Database className="h-4 w-4" />,
      progress: [60, 80],
      isActive: progress >= 60 && progress < 80,
      isCompleted: progress >= 80,
      score: qualityMetrics.sourceQualityScore
    },
    {
      name: 'Synthesis',
      icon: <Brain className="h-4 w-4" />,
      progress: [80, 100],
      isActive: progress >= 80 && progress < 100,
      isCompleted: progress >= 100,
      score: (phaseScores.synthesis.coherence + phaseScores.synthesis.paradigmFit) / 2,
      metrics: {
        efficiency: phaseScores.synthesis.coherence,
        accuracy: phaseScores.synthesis.paradigmFit
      }
    }
  ]

  const calculateOverallScore = () => {
    const weights = {
      sourceQuality: 0.3,
      paradigmAlignment: 0.2,
      searchEffectiveness: 0.3,
      answerConfidence: 0.2
    }
    
    const score = 
      qualityMetrics.sourceQualityScore * weights.sourceQuality +
      qualityMetrics.paradigmAlignment * weights.paradigmAlignment +
      qualityMetrics.searchEffectiveness * weights.searchEffectiveness +
      qualityMetrics.answerConfidence * weights.answerConfidence
    
    return Math.round(score)
  }

  return (
    <Card className="mt-6 animate-slide-up">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Research Analytics
        </h3>
        <div className="flex items-center gap-3">
          {canCancel() && (
            <Button
              variant="danger"
              size="sm"
              icon={X}
              loading={isCancelling}
              onClick={handleCancel}
              className="text-xs"
            >
              {isCancelling ? 'Cancelling...' : 'Cancel'}
            </Button>
          )}
          <StatusBadge
            status={getStatusType()}
            label={getStatusLabel()}
            variant="solid"
            size="md"
          />
        </div>
      </div>

      {/* Quality Metrics Dashboard */}
      {showMetricsDashboard && progress > 0 && (
        <div className="mb-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Research Quality Metrics
            </h4>
            <div className="flex items-center gap-2">
              <Badge variant="default" size="sm" className={getMetricColor(calculateOverallScore())}>
                Overall Score: {calculateOverallScore()}/100
              </Badge>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowMetricsDashboard(false)}
                className="text-xs"
              >
                Hide
              </Button>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className={cn("rounded-lg p-3 transition-all", getMetricBgColor(qualityMetrics.sourceQualityScore))}>
              <div className="flex items-center justify-between">
                <Award className="h-4 w-4 opacity-50" />
                <span className={cn("text-2xl font-bold", getMetricColor(qualityMetrics.sourceQualityScore))}>
                  {qualityMetrics.sourceQualityScore}%
                </span>
              </div>
              <div className="text-xs text-gray-700 dark:text-gray-300 mt-1">Source Quality</div>
              <div className="mt-2 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-current transition-all duration-500"
                  style={{ width: `${qualityMetrics.sourceQualityScore}%`, color: getParadigmColorValue('bernard') }}
                />
              </div>
            </div>
            
            <div className={cn("rounded-lg p-3 transition-all", getMetricBgColor(qualityMetrics.paradigmAlignment))}>
              <div className="flex items-center justify-between">
                <Target className="h-4 w-4 opacity-50" />
                <span className={cn("text-2xl font-bold", getMetricColor(qualityMetrics.paradigmAlignment))}>
                  {qualityMetrics.paradigmAlignment}%
                </span>
              </div>
              <div className="text-xs text-gray-700 dark:text-gray-300 mt-1">Paradigm Fit</div>
              <div className="mt-2 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-current transition-all duration-500"
                  style={{ width: `${qualityMetrics.paradigmAlignment}%`, color: getParadigmColorValue('maeve') }}
                />
              </div>
            </div>
            
            <div className={cn("rounded-lg p-3 transition-all", getMetricBgColor(qualityMetrics.searchEffectiveness))}>
              <div className="flex items-center justify-between">
                <TrendingUp className="h-4 w-4 opacity-50" />
                <span className={cn("text-2xl font-bold", getMetricColor(qualityMetrics.searchEffectiveness))}>
                  {qualityMetrics.searchEffectiveness}%
                </span>
              </div>
              <div className="text-xs text-gray-700 dark:text-gray-300 mt-1">Search Effect</div>
              <div className="mt-2 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-current transition-all duration-500"
                  style={{ width: `${qualityMetrics.searchEffectiveness}%`, color: getParadigmColorValue('dolores') }}
                />
              </div>
            </div>
            
            <div className={cn("rounded-lg p-3 transition-all", getMetricBgColor(qualityMetrics.answerConfidence))}>
              <div className="flex items-center justify-between">
                <Brain className="h-4 w-4 opacity-50" />
                <span className={cn("text-2xl font-bold", getMetricColor(qualityMetrics.answerConfidence))}>
                  {qualityMetrics.answerConfidence}%
                </span>
              </div>
              <div className="text-xs text-gray-700 dark:text-gray-300 mt-1">Confidence</div>
              <div className="mt-2 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-current transition-all duration-500"
                  style={{ width: `${qualityMetrics.answerConfidence}%`, color: getParadigmColorValue('teddy') }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {progress > 0 && (currentStatus === 'processing' || currentStatus === 'in_progress') && (
        <>
          <ProgressBar
            value={progress}
            max={100}
            variant="default"
            showLabel
            label="Progress"
            shimmer
            className="mb-4"
          />
          
          {/* Enhanced Research Phases with Scores */}
          <div className="mb-4 bg-gray-50 dark:bg-gray-800/30 rounded-lg p-4">
            <div className="flex justify-between items-center gap-2">
              {researchPhases.map((phase, index) => (
                <div key={phase.name} className="flex items-center flex-1">
                  <div className="flex flex-col items-center flex-1">
                    <div className={cn(
                      "p-2 rounded-full mb-1 transition-all duration-300 relative",
                      phase.isCompleted ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400' : 
                      phase.isActive ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 animate-pulse' : 
                      'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-600'
                    )}>
                      {phase.isCompleted ? <CheckCircle className="h-4 w-4" /> : phase.icon}
                      {phase.score !== undefined && phase.score > 0 && (
                        <Badge 
                          variant="default" 
                          size="sm" 
                          className={cn(
                            "absolute -top-2 -right-2 text-xs",
                            getMetricColor(phase.score)
                          )}
                        >
                          {Math.round(phase.score)}
                        </Badge>
                      )}
                    </div>
                    <span className={cn("text-xs font-medium", phase.isActive ? 'text-text' : 'text-text-muted')}>
                      {phase.name}
                    </span>
                    {phase.metrics && phase.isCompleted && (
                      <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        {phase.metrics.time || `${phase.metrics.accuracy}%`}
                      </div>
                    )}
                  </div>
                  {index < researchPhases.length - 1 && (
                    <div className={cn(
                      "h-0.5 flex-1 mx-2 transition-colors",
                      phase.isCompleted ? 'bg-green-500' : 'bg-gray-200 dark:bg-gray-700'
                    )} />
                  )}
                </div>
              ))}
            </div>
          </div>
          
          {/* Enhanced Statistics with Trends */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 relative overflow-hidden">
              <div className="text-2xl font-bold text-text">{stats.sourcesFound}</div>
              <div className="text-xs text-text-muted">Sources Found</div>
              {stats.sourcesFound > 0 && (
                <TrendingUp className="absolute top-2 right-2 h-3 w-3 text-green-500 opacity-50" />
              )}
            </div>
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 relative overflow-hidden">
              <div className="text-2xl font-bold text-text">
                {stats.totalSearches > 0 ? `${stats.searchesCompleted}/${stats.totalSearches}` : '-'}
              </div>
              <div className="text-xs text-text-muted">Searches</div>
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-200 dark:bg-gray-700">
                <div
                  className="h-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${stats.totalSearches > 0 ? (stats.searchesCompleted / stats.totalSearches) * 100 : 0}%` }}
                />
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 relative overflow-hidden">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {stats.highQualitySources}
              </div>
              <div className="text-xs text-text-muted">High Quality</div>
              {stats.highQualitySources > 0 && (
                <Award className="absolute top-2 right-2 h-3 w-3 text-green-500 opacity-50" />
              )}
            </div>
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 relative overflow-hidden">
              <div className="text-2xl font-bold text-text">
                {stats.sourcesFound > 0 ? `${Math.round((stats.highQualitySources / stats.sourcesFound) * 100)}%` : '-'}
              </div>
              <div className="text-xs text-text-muted">Quality Rate</div>
              <div className="absolute top-2 right-2">
                {stats.sourcesFound > 0 && stats.highQualitySources / stats.sourcesFound > 0.7 ? (
                  <CheckCircle className="h-3 w-3 text-green-500 opacity-50" />
                ) : (
                  <AlertTriangle className="h-3 w-3 text-amber-500 opacity-50" />
                )}
              </div>
            </div>
          </div>
        </>
      )}

      <div 
        ref={updatesContainerRef}
        className="space-y-2 max-h-64 overflow-y-auto"
        role="log"
        aria-label="Research progress updates"
        aria-live="polite"
      >
        {updates.map((update, index) => {
          const isSearchCompleted = update.message?.includes('Found') && update.message?.includes('results')
          const resultsMatch = update.message?.match(/Found (\d+) results/)
          const resultsCount = resultsMatch ? parseInt(resultsMatch[1]) : 0
          
          return (
            <div
              key={index}
              className="flex items-start gap-3 text-sm py-2 border-b border-border last:border-0 animate-slide-up"
            >
              <span className="text-text-muted text-xs font-mono">
                {format(new Date(update.timestamp), 'HH:mm:ss')}
              </span>
              <div className="flex-1">
                <p className="text-text">
                  {update.message || `Status: ${update.status}`}
                </p>
                {isSearchCompleted && resultsCount > 0 && (
                  <div className="mt-1">
                    <Badge 
                      variant={resultsCount > 10 ? 'success' : resultsCount > 5 ? 'warning' : 'default'} 
                      size="sm"
                    >
                      {resultsCount} results
                    </Badge>
                  </div>
                )}
              </div>
              {update.message?.includes('Searching') && (
                <Search className="h-4 w-4 text-blue-500 animate-pulse" />
              )}
              {update.message?.includes('credibility') && (
                <CheckCircle className="h-4 w-4 text-green-500" />
              )}
              {update.message?.includes('duplicate') && (
                <Database className="h-4 w-4 text-yellow-500" />
              )}
            </div>
          )
        })}
      </div>

      {updates.length === 0 && !isConnecting && (
        <div className="text-center py-8 animate-fade-in">
          <Clock className="h-12 w-12 text-text-muted opacity-30 mx-auto mb-3" />
          <p className="text-text-muted">
            Waiting for updates...
          </p>
        </div>
      )}
      
      {updates.length === 0 && isConnecting && (
        <div className="text-center py-8">
          <LoadingSpinner
            size="xl"
            variant="primary"
            text="Connecting to research stream..."
          />
        </div>
      )}
      
      {/* Enhanced Source Previews with Quality Indicators */}
      {sourcePreviews.length > 0 && showSourcePreviews && (
        <div className="mt-4 border-t border-border pt-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-text flex items-center gap-2">
              <Database className="h-4 w-4" />
              Recent Sources
            </h4>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSourcePreviews(false)}
              className="text-xs"
            >
              Hide
            </Button>
          </div>
          <div className="space-y-2">
            {sourcePreviews.map((source, index) => (
              <div
                key={index}
                className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 animate-slide-up border border-transparent hover:border-gray-300 dark:hover:border-gray-600 transition-all"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <h5 className="text-sm font-medium text-text truncate">
                      {source.title}
                    </h5>
                    <p className="text-xs text-text-muted mt-1">
                      {source.domain}
                    </p>
                    {source.snippet && (
                      <p className="text-xs text-text-muted mt-1 line-clamp-2">
                        {source.snippet}
                      </p>
                    )}
                  </div>
                  {source.credibility !== undefined && (
                    <div className="flex flex-col items-end gap-1">
                      <Badge
                        variant={source.credibility > 0.7 ? 'success' : source.credibility > 0.4 ? 'warning' : 'error'}
                        size="sm"
                        className="flex items-center gap-1"
                      >
                        {source.credibility > 0.7 ? (
                          <CheckCircle className="h-3 w-3" />
                        ) : (
                          <AlertTriangle className="h-3 w-3" />
                        )}
                        {Math.round(source.credibility * 100)}%
                      </Badge>
                      <span className="text-xs text-gray-500">Credibility</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}