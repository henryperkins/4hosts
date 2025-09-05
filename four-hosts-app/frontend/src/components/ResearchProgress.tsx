import React, { useEffect, useState, useRef, useLayoutEffect } from 'react'
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
  status: StatusType
  progress?: number
  message?: string
  timestamp: string
}

interface WebSocketData {
  status?: StatusType
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
  // Determinate progress fields
  items_done?: number
  items_total?: number
  eta_seconds?: number
  phase?: string
  old_phase?: string
  new_phase?: string
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
  // Limit the number of progress updates retained to prevent unbounded growth
  const MAX_UPDATES = 100
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
  const [startTime, setStartTime] = useState<number | null>(null)
  const [elapsedSec, setElapsedSec] = useState<number>(0)
  // Track granular layer progress within Context-Engineering (Write/Rewrite/Select/Optimize/Compress/Isolate)
  const CE_LAYERS = ['Write', 'Rewrite', 'Select', 'Optimize', 'Compress', 'Isolate'] as const
  const [ceLayerProgress, setCeLayerProgress] = useState<{ done: number; total: number }>({ done: 0, total: 6 })
  const [determinateProgress, setDeterminateProgress] = useState<{ done: number; total: number } | null>(null)
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null)
  const [currentPhase, setCurrentPhase] = useState<string>('initialization')
  const updatesContainerRef = useRef<HTMLDivElement>(null)
  // Refs to avoid re-subscribing WebSocket when these change
  const currentStatusRef = useRef<ProgressUpdate['status']>('pending')
  const onCompleteRef = useRef<typeof onComplete>(onComplete)
  const onCancelRef = useRef<typeof onCancel>(onCancel)

  useEffect(() => {
    currentStatusRef.current = currentStatus
  }, [currentStatus])

  useEffect(() => {
    onCompleteRef.current = onComplete
  }, [onComplete])

  useEffect(() => {
    onCancelRef.current = onCancel
  }, [onCancel])

  useEffect(() => {
    // Establish a single WebSocket connection per researchId
    setIsConnecting(true)

    api.connectWebSocket(researchId, (message) => {
      setIsConnecting(false)
      const data = message.data

      // Handle different message types
      let statusUpdate: WebSocketData | undefined

      switch (message.type) {
        case 'research_progress':
          statusUpdate = {
            status: data.status,
            progress: data.progress,
            message: data.message || `Phase: ${data.phase || 'processing'}`,
            items_done: data.items_done,
            items_total: data.items_total,
            eta_seconds: data.eta_seconds,
            phase: data.phase
          }
          // Update determinate progress if backend sends ratios
          if (typeof data.items_done === 'number' && typeof data.items_total === 'number') {
            setDeterminateProgress({ done: data.items_done, total: data.items_total })

            // If we are in Context-Engineering phase, keep an internal copy so we can
            // color W-R-S-O-C-I badges accurately.
            if (data.phase === 'context_engineering') {
              setCeLayerProgress({ done: data.items_done, total: data.items_total })
            }
          }
          // Update ETA if available
          if (typeof data.eta_seconds === 'number') {
            setEtaSeconds(data.eta_seconds)
          }
          // Update phase if available
          if (data.phase) {
            setCurrentPhase(data.phase)
            if (data.phase !== 'context_engineering') {
              // Reset CE layer state when we leave context-engineering
              setCeLayerProgress({ done: 0, total: CE_LAYERS.length })
            }
          }
          break
        case 'research_phase_change':
          statusUpdate = {
            message: `Phase changed: ${data.old_phase} → ${data.new_phase}`,
            phase: data.new_phase
          }
          if (data.new_phase) {
            setCurrentPhase(data.new_phase)
          }
          // Reset determinate progress when phase changes
          setDeterminateProgress(null)
          setEtaSeconds(null)
          break
        case 'source_found':
          statusUpdate = {
            message: `Found source: ${data.source?.title || 'New source'} (${data.total_sources} total)`
          }
          if (data.source) {
            setSourcePreviews(prev => [...prev.slice(-4), {
              title: data.source?.title || '',
              domain: data.source?.domain || '',
              snippet: data.source?.snippet || '',
              credibility: data.source?.credibility_score || 0
            }])
          }
          break
        case 'source_analyzed':
          statusUpdate = {
            message: `Analyzed source ${data.source_id} (${data.analyzed_count} analyzed)`
          }
          if (typeof data.analyzed_count === 'number') {
            // Ensure we never set undefined which violates ResearchStats type
            setStats(prev => ({
              ...prev,
              sourcesAnalyzed: data.analyzed_count ?? prev.sourcesAnalyzed,
            }))
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
          if (!startTime) setStartTime(Date.now())
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
            message: `Checking credibility: ${data.domain} (${((data.score || 0) * 100).toFixed(0)}%)`
          }
          if ((data.score || 0) > 0.7) {
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
        status: statusUpdate.status || currentStatusRef.current,
        progress: statusUpdate.progress,
        message: statusUpdate.message,
        timestamp: new Date().toISOString(),
      }

      // Keep only a rolling window of updates to avoid memory leaks
      setUpdates(prev => [...prev.slice(-(MAX_UPDATES - 1)), update])

      if (data.status) {
        setCurrentStatus(data.status)
      }

      if (data.progress !== undefined) {
        setProgress(data.progress)
      }

      if (data.status === 'completed' && onCompleteRef.current) {
        onCompleteRef.current()
      }

      if (data.status === 'cancelled' && onCancelRef.current) {
        onCancelRef.current()
      }
    })

    return () => {
      api.unsubscribeFromResearch(researchId)
    }
  }, [researchId])

  // Elapsed time ticker
  useEffect(() => {
    if (!startTime) return
    const id = setInterval(() => {
      setElapsedSec(Math.max(0, Math.floor((Date.now() - startTime) / 1000)))
    }, 1000)
    return () => clearInterval(id)
  }, [startTime])

  // Auto-scroll to bottom when updates change
  useLayoutEffect(() => {
    if (updatesContainerRef.current) {
      updatesContainerRef.current.scrollTop = updatesContainerRef.current.scrollHeight
    }
  }, [updates])

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
  // Map backend phase strings → icon/component label for timeline rendering
  const PHASE_ORDER: { key: string; label: string; icon: React.ReactNode }[] = [
    { key: 'classification', label: 'Classification', icon: <Brain className="h-4 w-4" /> },
    { key: 'context_engineering', label: 'Context Engineering', icon: <Zap className="h-4 w-4" /> },
    { key: 'search', label: 'Search & Retrieval', icon: <Search className="h-4 w-4" /> },
    { key: 'analysis', label: 'Analysis', icon: <Database className="h-4 w-4" /> },
    { key: 'agentic_loop', label: 'Agentic Loop', icon: <Zap className="h-4 w-4" /> },
    { key: 'synthesis', label: 'Synthesis', icon: <Brain className="h-4 w-4" /> },
    { key: 'complete', label: 'Complete', icon: <CheckCircle className="h-4 w-4" /> }
  ]

  const researchPhases: ResearchPhase[] = PHASE_ORDER.map((p, idx) => {
    const currentIdx = PHASE_ORDER.findIndex(ph => ph.key === currentPhase)
    return {
      name: p.label,
      icon: p.icon,
      progress: [0, 0], // no longer used
      isActive: idx === currentIdx,
      isCompleted: idx < currentIdx
    }
  })

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
          <div className="mb-4">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-text-muted">
                {determinateProgress ? 
                  `${determinateProgress.done} / ${determinateProgress.total} items` : 
                  'Progress'
                }
              </span>
              {etaSeconds !== null && etaSeconds > 0 && (
                <span className="text-xs text-text-muted">
                  ~{Math.floor(etaSeconds / 60).toString().padStart(2, '0')}:{(etaSeconds % 60).toString().padStart(2, '0')} remaining
                </span>
              )}
            </div>
            <ProgressBar
              value={determinateProgress ? (determinateProgress.done / determinateProgress.total) * 100 : progress}
              max={100}
              variant="default"
              showLabel={false}
              shimmer
            />
            {currentPhase && (
              <div className="mt-2">
                <Badge variant="info" size="sm" className="capitalize">
                  {currentPhase.replace(/_/g, ' ')}
                </Badge>
              </div>
            )}
          </div>

          {/* Context-Engineering Layer Badges (Write → Isolate) */}
          <div className="mb-3 flex items-center gap-2">
            {CE_LAYERS.map((label, idx) => {
              const completed = ceLayerProgress.done > idx
              const isActiveLayer = ceLayerProgress.done === idx
              const badgeStyle = completed
                ? 'bg-blue-100 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300'
                : isActiveLayer
                  ? 'bg-yellow-100 dark:bg-yellow-900/30 border-yellow-300 dark:border-yellow-700 text-yellow-700 dark:text-yellow-300'
                  : 'bg-gray-100 dark:bg-gray-800 border-gray-300 dark:border-gray-700 text-gray-500 dark:text-gray-400'

              return (
                <span
                  key={label}
                  className={`text-[11px] px-2 py-1 rounded-full border ${badgeStyle}`}
                >
                  {label}
                </span>
              )
            })}
          </div>
          
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
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
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
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3">
              <div className="text-2xl font-bold text-text">
                {`${Math.floor(elapsedSec / 60).toString().padStart(2, '0')}:${(elapsedSec % 60).toString().padStart(2, '0')}`}
              </div>
              <div className="text-xs text-text-muted">Elapsed</div>
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
