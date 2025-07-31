import React, { useEffect, useState, useRef } from 'react'
import { Clock, X, Search, Database, Brain, CheckCircle, Zap } from 'lucide-react'
import { format } from 'date-fns'
import { Button } from './ui/Button'
import { Card } from './ui/Card'
import { StatusBadge, type StatusType } from './ui/StatusIcon'
import { ProgressBar } from './ui/ProgressBar'
import { LoadingSpinner } from './ui/LoadingSpinner'
import { Badge } from './ui/Badge'
import api from '../services/api'

interface ResearchProgressProps {
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
  source?: {
    title: string
    domain: string
    snippet: string
    credibility_score: number
  }
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
}

interface ResearchStats {
  sourcesFound: number
  sourcesAnalyzed: number
  searchesCompleted: number
  totalSearches: number
  highQualitySources: number
}

interface SourcePreview {
  title: string
  domain: string
  snippet?: string
  credibility?: number
}

export const ResearchProgress: React.FC<ResearchProgressProps> = ({ researchId, onComplete, onCancel }) => {
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
  const [sourcePreviews, setSourcePreviews] = useState<SourcePreview[]>([])
  const [showSourcePreviews, setShowSourcePreviews] = useState(true)
  const updatesContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Use WebSocket from api
    setIsConnecting(true)
      
      api.connectWebSocket(researchId, (message) => {
        setIsConnecting(false)
        const data = message.data as WebSocketData
        
        // Handle different message types
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
            break
          case 'source_found':
            statusUpdate = {
              message: `Found source: ${data.source?.title || 'New source'} (${data.total_sources} total)`
            }
            // Add source to previews
            if (data.source) {
              setSourcePreviews(prev => [...prev.slice(-4), {
                title: data.source!.title,
                domain: data.source!.domain,
                snippet: data.source!.snippet,
                credibility: data.source!.credibility_score
              }])
            }
            break
          case 'source_analyzed':
            statusUpdate = {
              message: `Analyzed source ${data.source_id} (${data.analyzed_count} analyzed)`
            }
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
            break
          case 'credibility.check':
            statusUpdate = {
              message: `Checking credibility: ${data.domain} (${((data.score ?? 0) * 100).toFixed(0)}%)`
            }
            if ((data.score ?? 0) > 0.7) {
              setStats(prev => ({ ...prev, highQualitySources: prev.highQualitySources + 1 }))
            }
            break
          case 'deduplication.progress':
            statusUpdate = {
              message: `Removed ${data.removed} duplicate results`
            }
            break
          default:
            // Handle legacy message format
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
          // Auto-scroll to bottom when new update arrives
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

  const handleCancel = async () => {
    setIsCancelling(true)
    try {
      await api.cancelResearch(researchId)
      // Status will be updated via WebSocket
    } catch {
      // Failed to cancel research
      // You might want to show a toast error here
    } finally {
      setIsCancelling(false)
    }
  }

  const getStatusType = (): StatusType => {
    if (isConnecting) {
      return 'processing'
    }
    
    switch (currentStatus) {
      case 'pending':
        return 'pending'
      case 'processing':
      case 'in_progress':
        return 'processing'
      case 'completed':
        return 'completed'
      case 'failed':
        return 'failed'
      case 'cancelled':
        return 'cancelled'
      default:
        return 'pending'
    }
  }

  const getStatusLabel = () => {
    if (isConnecting) {
      return 'Connecting...'
    }
    return currentStatus.charAt(0).toUpperCase() + currentStatus.slice(1)
  }

  const canCancel = () => {
    return currentStatus === 'processing' || currentStatus === 'in_progress' || currentStatus === 'pending'
  }

  // Define research phases
  const researchPhases: ResearchPhase[] = [
    {
      name: 'Classification',
      icon: <Brain className="h-4 w-4" />,
      progress: [0, 10],
      isActive: progress >= 0 && progress < 10,
      isCompleted: progress >= 10
    },
    {
      name: 'Context Engineering',
      icon: <Zap className="h-4 w-4" />,
      progress: [10, 20],
      isActive: progress >= 10 && progress < 20,
      isCompleted: progress >= 20
    },
    {
      name: 'Search & Retrieval',
      icon: <Search className="h-4 w-4" />,
      progress: [20, 60],
      isActive: progress >= 20 && progress < 60,
      isCompleted: progress >= 60
    },
    {
      name: 'Analysis',
      icon: <Database className="h-4 w-4" />,
      progress: [60, 80],
      isActive: progress >= 60 && progress < 80,
      isCompleted: progress >= 80
    },
    {
      name: 'Synthesis',
      icon: <Brain className="h-4 w-4" />,
      progress: [80, 100],
      isActive: progress >= 80 && progress < 100,
      isCompleted: progress >= 100
    }
  ]

  return (
    <Card className="mt-6 animate-slide-up">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text">Research Progress</h3>
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
            variant="subtle"
            size="md"
          />
        </div>
      </div>

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
          
          {/* Research Phases */}
          <div className="mb-4 bg-gray-50 dark:bg-gray-800/30 rounded-lg p-4">
            <div className="flex justify-between items-center gap-2">
              {researchPhases.map((phase, index) => (
                <div key={phase.name} className="flex items-center flex-1">
                  <div className="flex flex-col items-center flex-1">
                    <div className={`
                      p-2 rounded-full mb-1 transition-colors
                      ${phase.isCompleted ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400' : 
                        phase.isActive ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 animate-pulse' : 
                        'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-600'}
                    `}>
                      {phase.isCompleted ? <CheckCircle className="h-4 w-4" /> : phase.icon}
                    </div>
                    <span className={`text-xs font-medium ${phase.isActive ? 'text-text' : 'text-text-muted'}`}>
                      {phase.name}
                    </span>
                  </div>
                  {index < researchPhases.length - 1 && (
                    <div className={`h-0.5 flex-1 mx-2 transition-colors ${
                      phase.isCompleted ? 'bg-green-500' : 'bg-gray-200 dark:bg-gray-700'
                    }`} />
                  )}
                </div>
              ))}
            </div>
          </div>
          
          {/* Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3">
              <div className="text-2xl font-bold text-text">{stats.sourcesFound}</div>
              <div className="text-xs text-text-muted">Sources Found</div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3">
              <div className="text-2xl font-bold text-text">
                {stats.totalSearches > 0 ? `${stats.searchesCompleted}/${stats.totalSearches}` : '-'}
              </div>
              <div className="text-xs text-text-muted">Searches</div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {stats.highQualitySources}
              </div>
              <div className="text-xs text-text-muted">High Quality</div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3">
              <div className="text-2xl font-bold text-text">
                {stats.sourcesFound > 0 ? `${Math.round((stats.highQualitySources / stats.sourcesFound) * 100)}%` : '-'}
              </div>
              <div className="text-xs text-text-muted">Quality Rate</div>
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
          // Check if this is a search completed message with results
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
                    <Badge variant="default" size="sm">
                      {resultsCount} results
                    </Badge>
                  </div>
                )}
              </div>
              {/* Icon indicators for different message types */}
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
      
      {/* Source Previews */}
      {sourcePreviews.length > 0 && showSourcePreviews && (
        <div className="mt-4 border-t border-border pt-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-text">Recent Sources</h4>
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
                className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 animate-slide-up"
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
                    <Badge
                      variant={source.credibility > 0.7 ? 'success' : source.credibility > 0.4 ? 'warning' : 'error'}
                      size="sm"
                    >
                      {Math.round(source.credibility * 100)}%
                    </Badge>
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