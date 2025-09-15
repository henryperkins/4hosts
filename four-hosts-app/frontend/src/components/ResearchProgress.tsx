import React, { useEffect, useState, useRef, useLayoutEffect } from 'react'
import { FiClock, FiX, FiSearch, FiDatabase, FiCpu, FiCheckCircle, FiZap, FiChevronDown, FiChevronUp } from 'react-icons/fi'
import { format } from 'date-fns'
import { Button } from './ui/Button'
import { Card } from './ui/Card'
import { StatusBadge, type StatusType } from './ui/StatusIcon'
import { ProgressBar } from './ui/ProgressBar'
import { LoadingSpinner } from './ui/LoadingSpinner'
import { Badge } from './ui/Badge'
import { SwipeableTabs } from './ui/SwipeableTabs'
import { CollapsibleEvent } from './ui/CollapsibleEvent'
import api from '../services/api'
import { stripHtml } from '../utils/sanitize'

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
  type?: string
  priority?: 'low' | 'medium' | 'high' | 'critical'
  data?: any // Store original event data for expandable details
}

interface WebSocketData {
  status?: StatusType
  progress?: number
  message?: string
  query?: string
  engine?: string
  api?: string
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
  // MCP + system notifications
  server?: string
  tool?: string
  limit_type?: string
  remaining?: number
  reset_time?: string
}

interface RateLimitWarningData {
  message?: string
  limit_type?: string
  remaining?: number
  reset_time?: string
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
  duplicatesRemoved: number
  mcpToolsUsed: number
  apisQueried: Set<string>
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
    highQualitySources: 0,
    duplicatesRemoved: 0,
    mcpToolsUsed: 0,
    apisQueried: new Set<string>()
  })
  const [sourcePreviews, setSourcePreviews] = useState<SourcePreview[]>([])
  const [showSourcePreviews, setShowSourcePreviews] = useState(true)
  const [sourcesCollapsed, setSourcesCollapsed] = useState(true)
  const [startTime, setStartTime] = useState<number | null>(null)
  const [elapsedSec, setElapsedSec] = useState<number>(0)
  // Track granular layer progress within Context-Engineering (Write/Rewrite/Select/Optimize/Compress/Isolate)
  const CE_LAYERS = ['Write', 'Rewrite', 'Select', 'Optimize', 'Compress', 'Isolate'] as const
  const [ceLayerProgress, setCeLayerProgress] = useState<{ done: number; total: number }>({ done: 0, total: 6 })
  const CE_LEN = CE_LAYERS.length
  const [determinateProgress, setDeterminateProgress] = useState<{ done: number; total: number } | null>(null)
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null)
  const [currentPhase, setCurrentPhase] = useState<string>('initialization')
  const [analysisTotal, setAnalysisTotal] = useState<number | null>(null)
  const [showVerbose, setShowVerbose] = useState<boolean>(true)
  const [rateLimitWarning, setRateLimitWarning] = useState<RateLimitWarningData | null>(null)
  const [activeCategory, setActiveCategory] = useState<'all' | 'search' | 'sources' | 'analysis' | 'system' | 'errors'>('all')
  const [isMobile, setIsMobile] = useState(false)
  const updatesContainerRef = useRef<HTMLDivElement>(null)
  // Refs to avoid re-subscribing WebSocket when these change
  const currentStatusRef = useRef<ProgressUpdate['status']>('pending')
  const onCompleteRef = useRef<typeof onComplete>(onComplete)
  const onCancelRef = useRef<typeof onCancel>(onCancel)
  const startTimeRef = useRef<number | null>(null)
  useEffect(() => { startTimeRef.current = startTime }, [startTime])

  useEffect(() => {
    currentStatusRef.current = currentStatus
  }, [currentStatus])

  useEffect(() => {
    onCompleteRef.current = onComplete
  }, [onComplete])

  useEffect(() => {
    onCancelRef.current = onCancel
  }, [onCancel])

  // Default collapse on small screens and detect mobile
  useEffect(() => {
    try {
      if (typeof window !== 'undefined') {
        const checkMobile = () => {
          const mobile = window.innerWidth < 640
          setIsMobile(mobile)
          setSourcesCollapsed(mobile)
        }
        checkMobile()
        window.addEventListener('resize', checkMobile)
        return () => window.removeEventListener('resize', checkMobile)
      }
    } catch { /* noop */ }
  }, [])

  useEffect(() => {
    // Establish a single WebSocket connection per researchId
    setIsConnecting(true)

    api.connectWebSocket(researchId, (message) => {
      setIsConnecting(false)
      const data = message.data

      // Handle different message types
      let statusUpdate: WebSocketData | undefined

      switch (message.type) {
        case 'research_progress': {
          // Normalize status for UI
          const s = (data.status || '').toLowerCase()
          const normalized = s === 'in_progress' ? 'processing'
            : (s === 'pending' || s === 'processing' || s === 'completed' || s === 'failed' || s === 'cancelled')
              ? (s as ProgressUpdate['status'])
              : undefined
          statusUpdate = {
            status: normalized,
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
            if (data.phase === 'analysis') {
              setAnalysisTotal(data.items_total)
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
              setCeLayerProgress({ done: 0, total: CE_LEN })
            }
          }
          break
        }
        case 'system.notification':
          // Handle MCP tool events and other notifications
          if (data.server && data.tool) {
            const phase = data.status ? 'completed' : 'executing'
            statusUpdate = {
              message: `🔧 MCP ${data.server}.${data.tool} ${phase}`
            }
            // Count completed tool runs
            if (data.status) {
              setStats(prev => ({ ...prev, mcpToolsUsed: (prev.mcpToolsUsed || 0) + 1 }))
            }
          } else {
            statusUpdate = { message: data.message }
          }
          break
        case 'rate_limit.warning':
          statusUpdate = {
            message: `⚠️ Rate limit warning: ${data.message || data.limit_type}`
          }
          setRateLimitWarning(data)
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
          if (!startTimeRef.current) setStartTime(Date.now())
          break
        case 'search.started':
          statusUpdate = {
            message: `Searching (${data.index}/${data.total}): ${data.query}`
          }
          setStats(prev => {
            const apiName = (data.engine || data.api || '').trim()
            const nextSet = new Set(prev.apisQueried)
            if (apiName) nextSet.add(apiName)
            return { ...prev, totalSearches: data.total || prev.totalSearches, apisQueried: nextSet }
          })
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
          setStats(prev => ({
            ...prev,
            duplicatesRemoved: (prev.duplicatesRemoved || 0) + (data.removed || 0)
          }))
          break
        default: {
          // Handle legacy message format
          const s2 = (data.status || '').toLowerCase()
          const normalized2 = s2 === 'in_progress' ? 'processing'
            : (s2 === 'pending' || s2 === 'processing' || s2 === 'completed' || s2 === 'failed' || s2 === 'cancelled')
              ? (s2 as ProgressUpdate['status'])
              : undefined
          statusUpdate = {
            status: normalized2,
            progress: data.progress,
            message: data.message
          }
        }
      }

      // Assign priority based on event type
      const toPriority = (t?: string): ProgressUpdate['priority'] => {
        switch (t) {
          case 'research_failed':
          case 'error':
            return 'critical'
          case 'research_completed':
          case 'research_phase_change':
            return 'high'
          case 'credibility.check':
          case 'deduplication.progress':
          case 'source_found':
          case 'source_analyzed':
            return 'medium'
          default:
            return 'low'
        }
      }

      const update: ProgressUpdate = {
        status: statusUpdate.status || currentStatusRef.current,
        progress: statusUpdate.progress,
        message: statusUpdate.message,
        timestamp: new Date().toISOString(),
        type: message.type,
        priority: toPriority(message.type),
        data: data // Store original data for expandable details
      }

      // Keep only a rolling window of updates to avoid memory leaks
      setUpdates(prev => [...prev.slice(-(MAX_UPDATES - 1)), update])

      if (data.status) {
        const s = (data.status || '').toLowerCase()
        const normalized = s === 'in_progress' ? 'processing'
          : (s === 'pending' || s === 'processing' || s === 'completed' || s === 'failed' || s === 'cancelled')
            ? (s as ProgressUpdate['status'])
            : undefined
        if (normalized) setCurrentStatus(normalized)
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
  }, [researchId, CE_LEN])

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

  // Defensive cleanup for any provider HTML that might slip into messages
  const cleanMessage = (msg?: string) => {
    if (!msg) return ''
    try {
      return stripHtml(msg)
    } catch {
      return msg
    }
  }

  const isNoisy = (msg?: string) => {
    const m = (msg || '').toLowerCase()
    // Only hide truly repetitive messages
    return m.includes('heartbeat') || m.includes('still processing')
  }

  // Categorize updates for filtered views
  const categorize = (u: ProgressUpdate): 'search' | 'sources' | 'analysis' | 'system' | 'errors' | 'all' => {
    const t = (u.type || '').toLowerCase()
    if (t.includes('search.')) return 'search'
    if (t === 'source_found' || t === 'source_analyzed') return 'sources'
    if (t === 'research_phase_change' && (u.message || '').toLowerCase().includes('analysis')) return 'analysis'
    if (t === 'credibility.check' || t === 'deduplication.progress') return 'analysis'
    if (t === 'system.notification' || t === 'rate_limit.warning' || t === 'connected' || t === 'disconnected') return 'system'
    if (t === 'error' || t === 'research_failed') return 'errors'
    return 'all'
  }

  const getMessageStyle = (p?: ProgressUpdate['priority']) => {
    switch (p) {
      case 'critical':
        return 'border-l-4 border-error bg-error/5'
      case 'high':
        return 'border-l-4 border-primary bg-primary/5'
      default:
        return ''
    }
  }

  type CategoryKey = 'all' | 'search' | 'sources' | 'analysis' | 'system' | 'errors'
  const categoryCounts = React.useMemo((): Record<CategoryKey, number> => {
    const counts: Record<CategoryKey, number> = { all: updates.length, search: 0, sources: 0, analysis: 0, system: 0, errors: 0 }
    for (const u of updates) {
      const cat = categorize(u) as CategoryKey
      counts[cat] += 1
    }
    return counts
  }, [updates])

  const canCancel = () => {
    return currentStatus === 'processing' || currentStatus === 'in_progress' || currentStatus === 'pending'
  }

  // Helper function to render events
  const renderEvents = () => {
    return updates
      .filter(u => showVerbose || !isNoisy(u.message))
      .filter(u => (activeCategory === 'all' ? true : categorize(u) === activeCategory))
      .map((update, index) => {
        // Determine icon for the event
        let icon: React.ReactNode = null
        if (update.message?.includes('Searching')) {
          icon = <FiSearch className="h-4 w-4 text-primary animate-pulse" />
        } else if (update.message?.includes('credibility')) {
          icon = <FiCheckCircle className="h-4 w-4 text-green-500" />
        } else if (update.message?.includes('duplicate')) {
          icon = <FiDatabase className="h-4 w-4 text-yellow-500" />
        }

        return (
          <CollapsibleEvent
            key={index}
            message={cleanMessage(update.message) || `Status: ${update.status}`}
            timestamp={format(new Date(update.timestamp), 'HH:mm:ss')}
            details={update.data}
            priority={update.priority}
            type={update.type}
            icon={icon}
            className={getMessageStyle(update.priority)}
          />
        )
      })
  }

  // Define research phases
  // Map backend phase strings → icon/component label for timeline rendering
  const PHASE_ORDER: { key: string; label: string; icon: React.ReactNode }[] = [
    { key: 'classification', label: 'Classification', icon: <FiCpu className="h-4 w-4" /> },
    { key: 'context_engineering', label: 'Context Engineering', icon: <FiZap className="h-4 w-4" /> },
    { key: 'search', label: 'Search & Retrieval', icon: <FiSearch className="h-4 w-4" /> },
    { key: 'analysis', label: 'Analysis', icon: <FiDatabase className="h-4 w-4" /> },
    { key: 'agentic_loop', label: 'Agentic Loop', icon: <FiZap className="h-4 w-4" /> },
    { key: 'synthesis', label: 'Synthesis', icon: <FiCpu className="h-4 w-4" /> },
    { key: 'complete', label: 'Complete', icon: <FiCheckCircle className="h-4 w-4" /> }
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
        <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
          <Button
            variant={showVerbose ? 'secondary' : 'ghost'}
            size="sm"
            onClick={() => setShowVerbose(v => !v)}
            className="text-xs"
          >
            {showVerbose ? 'Verbose' : 'Concise'}
          </Button>
          {canCancel() && (
            <Button
              variant="danger"
              size="sm"
              icon={FiX}
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
      {rateLimitWarning && (
        <div className="mb-3 p-3 rounded-lg border border-error/30 bg-error/10 text-error text-sm">
          <div className="font-medium">Rate limit warning</div>
          <div className="text-xs">
            {rateLimitWarning.message || `Approaching ${rateLimitWarning.limit_type} limit. ${rateLimitWarning.remaining ?? ''} remaining.`}
          </div>
        </div>
      )}

      {/* Persistent Session Summary (compact) */}
      <div className="mb-3 p-3 bg-surface-subtle rounded-lg">
        <h4 className="text-sm font-medium mb-2">Session Summary</h4>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>✅ {stats.searchesCompleted} searches completed</div>
          <div>📊 {stats.sourcesAnalyzed} sources analyzed</div>
          <div>🔍 {stats.highQualitySources} high-quality sources</div>
          <div>🗑️ {stats.duplicatesRemoved} duplicates removed</div>
          <div>🔧 {stats.mcpToolsUsed} tools executed</div>
          <div>⚡ {stats.apisQueried.size} APIs queried</div>
        </div>
      </div>

      {/* API status indicators */}
      {stats.apisQueried.size > 0 && (
        <div className="flex gap-2 mb-2">
          {Array.from(stats.apisQueried).map(apiName => (
            <Badge
              key={apiName}
              variant={currentPhase === 'search' ? 'info' : 'default'}
              size="sm"
              className={currentPhase === 'search' ? 'animate-pulse' : ''}
            >
              {apiName}
            </Badge>
          ))}
        </div>
      )}

      {progress > 0 && (currentStatus === 'processing' || currentStatus === 'in_progress') && (
        <>
          {/* Multi-layer progress: overall + sub-phase */}
          <div className="mb-4 space-y-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-text-muted">
                {determinateProgress ? `${determinateProgress.done} / ${determinateProgress.total} items` : 'Progress'}
              </span>
              {etaSeconds !== null && etaSeconds > 0 && (
                <span className="text-xs text-text-muted">
                  ~{Math.floor(etaSeconds / 60).toString().padStart(2, '0')}:{(etaSeconds % 60).toString().padStart(2, '0')} remaining
                </span>
              )}
            </div>
            <ProgressBar
              value={determinateProgress ? (determinateProgress.done / Math.max(1, determinateProgress.total)) * 100 : progress}
              max={100}
              variant="default"
              label="Overall"
              showLabel={false}
              shimmer
            />
            {currentPhase === 'search' && stats.totalSearches > 0 && (
              <ProgressBar
                value={(Math.min(stats.searchesCompleted, stats.totalSearches) / Math.max(1, stats.totalSearches)) * 100}
                max={100}
                variant="info"
                size="sm"
                label={`Search ${Math.min(stats.searchesCompleted, stats.totalSearches)}/${stats.totalSearches}`}
                showLabel={false}
              />
            )}
            {currentPhase === 'analysis' && analysisTotal && analysisTotal > 0 && (
              <ProgressBar
                value={(Math.min(stats.sourcesAnalyzed, analysisTotal) / Math.max(1, analysisTotal)) * 100}
                max={100}
                variant="info"
                size="sm"
                label={`Analyzing ${Math.min(stats.sourcesAnalyzed, analysisTotal)}/${analysisTotal}`}
                showLabel={false}
              />
            )}
            {currentPhase && (
              <div className="pt-1">
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
                ? 'bg-primary/10 border-primary/30 text-primary'
                : isActiveLayer
                  ? 'bg-primary/10 border-primary/30 text-primary'
                  : 'bg-surface-subtle border-border text-text-muted'

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
          <div className="mb-4 bg-surface-subtle rounded-lg p-4">
            <div className="flex justify-between items-center gap-2">
              {researchPhases.map((phase, index) => (
                <div key={phase.name} className="flex items-center flex-1">
                  <div className="flex flex-col items-center flex-1">
                    <div className={`
                      p-2 rounded-full mb-1 transition-colors
                      ${phase.isCompleted ? 'bg-success/10 text-success' : 
                        phase.isActive ? 'bg-primary/10 text-primary animate-pulse' : 
                        'bg-surface-subtle text-text-muted'}
                    `}>
                      {phase.isCompleted ? <FiCheckCircle className="h-4 w-4" /> : phase.icon}
                    </div>
                    <span className={`text-xs font-medium ${phase.isActive ? 'text-text' : 'text-text-muted'}`}>
                      {phase.name}
                    </span>
                  </div>
                  {index < researchPhases.length - 1 && (
                    <div className={`h-0.5 flex-1 mx-2 transition-colors ${
                      phase.isCompleted ? 'bg-success' : 'bg-surface-muted'
                    }`} />
                  )}
                </div>
              ))}
            </div>
          </div>
          
          {/* Statistics */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-8 gap-3 mb-4">
            <div className="bg-surface-subtle rounded-lg p-3">
              <div className="text-2xl font-bold text-text">{stats.sourcesFound}</div>
              <div className="text-xs text-text-muted">Sources Found</div>
            </div>
            <div className="bg-surface-subtle rounded-lg p-3">
              <div className="text-2xl font-bold text-text">
                {stats.totalSearches > 0 ? `${stats.searchesCompleted}/${stats.totalSearches}` : '-'}
              </div>
              <div className="text-xs text-text-muted">Searches</div>
            </div>
            <div className="bg-surface-subtle rounded-lg p-3">
              <div className="text-2xl font-bold text-text">
                {analysisTotal ? `${Math.min(stats.sourcesAnalyzed, analysisTotal)}/${analysisTotal}` : '-'}
              </div>
              <div className="text-xs text-text-muted">Analyzed</div>
            </div>
            <div className="bg-surface-subtle rounded-lg p-3">
              <div className="text-2xl font-bold text-success">
                {stats.highQualitySources}
              </div>
              <div className="text-xs text-text-muted">High Quality</div>
            </div>
            <div className="bg-surface-subtle rounded-lg p-3">
              <div className="text-2xl font-bold text-text">
                {stats.sourcesFound > 0 ? `${Math.round((stats.highQualitySources / stats.sourcesFound) * 100)}%` : '-'}
              </div>
              <div className="text-xs text-text-muted">Quality Rate</div>
            </div>
            <div className="bg-surface-subtle rounded-lg p-3">
              <div className="text-2xl font-bold text-error">{stats.duplicatesRemoved || 0}</div>
              <div className="text-xs text-text-muted">Duplicates Removed</div>
            </div>
            <div className="bg-surface-subtle rounded-lg p-3">
              <div className="text-2xl font-bold text-primary">{stats.apisQueried.size || 0}</div>
              <div className="text-xs text-text-muted">APIs Used</div>
            </div>
            <div className="bg-surface-subtle rounded-lg p-3">
              <div className="text-2xl font-bold text-text">
                {`${Math.floor(elapsedSec / 60).toString().padStart(2, '0')}:${(elapsedSec % 60).toString().padStart(2, '0')}`}
              </div>
              <div className="text-xs text-text-muted">Elapsed</div>
            </div>
          </div>

        </>
      )}

      {/* Category filters - SwipeableTabs on mobile, regular tabs on desktop */}
      {isMobile ? (
        <SwipeableTabs
          tabs={[
            { key: 'all', label: 'All', badge: categoryCounts.all },
            { key: 'search', label: 'Search', badge: categoryCounts.search },
            { key: 'sources', label: 'Sources', badge: categoryCounts.sources },
            { key: 'analysis', label: 'Analysis', badge: categoryCounts.analysis },
            { key: 'system', label: 'System', badge: categoryCounts.system },
            { key: 'errors', label: 'Errors', badge: categoryCounts.errors, badgeVariant: categoryCounts.errors > 0 ? 'error' : 'default' },
          ]}
          activeTab={activeCategory}
          onTabChange={(key) => setActiveCategory(key as typeof activeCategory)}
        >
          <div
            ref={updatesContainerRef}
            className="space-y-2 max-h-48 sm:max-h-64 overflow-y-auto"
            role="log"
            aria-label="Research progress updates"
            aria-live="polite"
          >
            {renderEvents()}
          </div>
        </SwipeableTabs>
      ) : (
        <>
          <div className="mb-2 flex flex-wrap gap-2 text-xs">
            {([
              { key: 'all', label: `All (${categoryCounts.all})` },
              { key: 'search', label: `Search (${categoryCounts.search})` },
              { key: 'sources', label: `Sources (${categoryCounts.sources})` },
              { key: 'analysis', label: `Analysis (${categoryCounts.analysis})` },
              { key: 'system', label: `System (${categoryCounts.system})` },
              { key: 'errors', label: `Errors (${categoryCounts.errors})`, emphasize: categoryCounts.errors > 0 },
            ] as const).map(tab => (
              <button
                key={tab.key}
                onClick={() => setActiveCategory(tab.key)}
                className={`px-2 py-1 rounded border ${activeCategory === tab.key ? 'bg-primary/10 border-primary/30 text-primary' : 'bg-surface-subtle border-border text-text-muted'} ${'emphasize' in tab && tab.emphasize ? 'text-error border-error/30' : ''}`}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div
            ref={updatesContainerRef}
            className="space-y-2 max-h-48 sm:max-h-64 overflow-y-auto"
            role="log"
            aria-label="Research progress updates"
            aria-live="polite"
          >
            {renderEvents()}
          </div>
        </>
      )}

      {updates.length === 0 && !isConnecting && (
        <div className="text-center py-8 animate-fade-in">
          <FiClock className="h-12 w-12 text-text-muted opacity-30 mx-auto mb-3" />
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
      
      {/* Source Previews (collapsible, mobile-first) */}
      {sourcePreviews.length > 0 && showSourcePreviews && (
        <div className="mt-4 border-t border-border pt-2 sm:pt-4">
          <button
            className="w-full flex items-center justify-between py-2 text-left"
            onClick={() => setSourcesCollapsed(v => !v)}
            aria-expanded={!sourcesCollapsed}
          >
            <h4 className="text-sm font-medium text-text">Recent Sources</h4>
            <div className="flex items-center gap-2">
              <span className="text-xs text-text-muted">{sourcePreviews.length}</span>
              {sourcesCollapsed ? <FiChevronDown className="h-4 w-4" /> : <FiChevronUp className="h-4 w-4" />}
            </div>
          </button>
          {!sourcesCollapsed && (
            <div className="space-y-2">
              {sourcePreviews.map((source, index) => (
                <div
                  key={index}
                  className="bg-surface-subtle rounded-lg p-3 animate-slide-up"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <h5 className="text-sm font-medium text-text truncate">
                        {source.title}
                      </h5>
                      <p className="text-xs text-text-muted mt-1 truncate">
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
          )}
          <div className="mt-2 flex items-center justify-end">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSourcePreviews(false)}
              className="text-xs"
            >
              Hide
            </Button>
          </div>
        </div>
      )}
    </Card>
  )
}
