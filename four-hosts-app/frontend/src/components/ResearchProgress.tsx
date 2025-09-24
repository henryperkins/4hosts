import React, { useEffect, useState, useRef, useMemo } from 'react'
import { FiClock, FiX, FiChevronDown, FiChevronUp } from 'react-icons/fi'
import { Button } from './ui/Button'
import { Card } from './ui/Card'
import { StatusBadge, type StatusType } from './ui/StatusIcon'
import { ProgressBar } from './ui/ProgressBar'
import { LoadingSpinner } from './ui/LoadingSpinner'
import { Badge } from './ui/Badge'
import PhaseTracker from './research/PhaseTracker'
import ResearchStats from './research/ResearchStats'
import EventLog, { type ProgressUpdate as LogUpdate, type CategoryKey as LogCategory } from './research/EventLog'
import api from '../services/api'
import { stripHtml } from '../utils/sanitize'

const MAX_UPDATES = 100
const MAX_SOURCE_PREVIEWS = 10
const MAX_API_BADGES = 12
const RESEARCH_ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/

const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

const toNonNegativeInt = (value: unknown): number | null => {
  const finite = toFiniteNumber(value)
  if (finite === null) return null
  const normalized = Math.max(0, Math.floor(finite))
  return Number.isFinite(normalized) ? normalized : null
}

const clampPercentage = (value: number | null | undefined): number => {
  if (value === null || value === undefined) return 0
  if (!Number.isFinite(value)) return 0
  if (value < 0) return 0
  if (value > 100) return 100
  return value
}

const normalizeStatus = (status: unknown): ProgressUpdate['status'] | undefined => {
  if (typeof status !== 'string') return undefined
  const lowered = status.toLowerCase()
  if (lowered === 'in_progress') return 'processing'
  if (lowered === 'pending' || lowered === 'processing' || lowered === 'completed' || lowered === 'failed' || lowered === 'cancelled') {
    return lowered as ProgressUpdate['status']
  }
  return undefined
}

const areSetsEqual = (a: Set<string>, b: Set<string>): boolean => {
  if (a.size !== b.size) return false
  for (const value of a) {
    if (!b.has(value)) {
      return false
    }
  }
  return true
}

const createInitialStats = (): ResearchStats => ({
  sourcesFound: 0,
  sourcesAnalyzed: 0,
  searchesCompleted: 0,
  totalSearches: 0,
  highQualitySources: 0,
  duplicatesRemoved: 0,
  mcpToolsUsed: 0,
  apisQueried: new Set<string>()
})

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
  data?: Record<string, unknown> // Store original event data for expandable details
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
  total_sources?: number
  domain?: string
  score?: number
  before_count?: number
  after_count?: number
  removed?: number
  source_id?: string
  analyzed_count?: number
  source?: {
    title?: string
    domain?: string
    snippet?: string
    credibility_score?: number
  }
  sources_found?: number
  sources_analyzed?: number
  searches_completed?: number
  total_searches?: number
  high_quality_sources?: number
  duplicates_removed?: number
  mcp_tools_used?: number
  apis?: string[]
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
  // Error field for research_failed events
  error?: string
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

//

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
  const [updates, setUpdates] = useState<ProgressUpdate[]>([])
  const [currentStatus, setCurrentStatus] = useState<ProgressUpdate['status']>('pending')
  const [progress, setProgress] = useState(0)
  const [isConnecting, setIsConnecting] = useState(true)
  const [isCancelling, setIsCancelling] = useState(false)
  const [stats, setStats] = useState<ResearchStats>(() => createInitialStats())
  const [sourcePreviews, setSourcePreviews] = useState<SourcePreview[]>([])
  const [showSourcePreviews, setShowSourcePreviews] = useState(true)
  const initialMobile = typeof window !== 'undefined' ? window.innerWidth < 640 : false
  const [sourcesCollapsed, setSourcesCollapsed] = useState(initialMobile)
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
  const [validationError, setValidationError] = useState<string | null>(null)
  const [pollingTimeout, setPollingTimeout] = useState<boolean>(false)
  const [liveAnnouncement, setLiveAnnouncement] = useState<string>('')
  const safeResearchId = React.useMemo(() => {
    const trimmed = (researchId || '').trim()
    return RESEARCH_ID_PATTERN.test(trimmed) ? trimmed : null
  }, [researchId])
  // Event log container handled inside EventLog component
  // Refs to avoid re-subscribing WebSocket when these change
  const currentStatusRef = useRef<ProgressUpdate['status']>('pending')
  const onCompleteRef = useRef<typeof onComplete>(onComplete)
  const onCancelRef = useRef<typeof onCancel>(onCancel)
  const startTimeRef = useRef<number | null>(null)
  const timeoutTimerRef = useRef<NodeJS.Timeout | null>(null)
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

  useEffect(() => {
    setStats(() => createInitialStats())
    setUpdates([])
    setSourcePreviews([])
    setDeterminateProgress(null)
    setEtaSeconds(null)
    setRateLimitWarning(null)
    setCurrentPhase('initialization')
    setAnalysisTotal(null)
    setActiveCategory('all')
    setStartTime(null)
    setElapsedSec(0)
  }, [safeResearchId])

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
    if (!safeResearchId) {
      setValidationError('Invalid research identifier - unable to subscribe to progress updates.')
      setIsConnecting(false)
      setCurrentStatus('failed')
      return
    }

    setValidationError(null)
    setIsConnecting(true)
    setPollingTimeout(false)

    // Set up polling timeout (configurable)
    if (timeoutTimerRef.current) {
      clearTimeout(timeoutTimerRef.current)
    }
    const TIMEOUT_MS = Number(import.meta.env.VITE_PROGRESS_WS_TIMEOUT_MS || 180000)
    timeoutTimerRef.current = setTimeout(() => {
      setPollingTimeout(true)
      setCurrentStatus('failed')
      if (onCompleteRef.current) {
        onCompleteRef.current()
      }
    }, Number.isFinite(TIMEOUT_MS) ? TIMEOUT_MS : 180000)

    api.connectWebSocket(safeResearchId, (message) => {
      setIsConnecting(false)

      // Clear timeout on any message received
      if (timeoutTimerRef.current) {
        clearTimeout(timeoutTimerRef.current)
        timeoutTimerRef.current = null
      }

      const rawData = message && typeof message.data === 'object' && message.data
        ? (message.data as Record<string, unknown>)
        : {}
      const data = rawData as WebSocketData

      let statusUpdate: WebSocketData | undefined
      const normalizedFromData = normalizeStatus(data.status)
      const safePhase = typeof data.phase === 'string' ? data.phase : undefined

      switch (message.type) {
        case 'research_progress': {
          const safeProgress = toFiniteNumber(data.progress)
          const itemsDone = toNonNegativeInt(data.items_done)
          const itemsTotal = toNonNegativeInt(data.items_total)
          const eta = toNonNegativeInt(data.eta_seconds)
          const messageText = typeof data.message === 'string' && data.message.trim()
            ? stripHtml(data.message)
            : safePhase
              ? `Phase: ${safePhase}`
              : undefined

          statusUpdate = {
            status: normalizedFromData,
            progress: safeProgress !== null ? clampPercentage(safeProgress) : undefined,
            message: messageText,
            items_done: itemsDone ?? undefined,
            items_total: itemsTotal ?? undefined,
            eta_seconds: eta ?? undefined,
            phase: safePhase
          }

          if (itemsDone !== null && itemsTotal !== null && itemsTotal > 0) {
            const boundedDone = Math.min(itemsDone, itemsTotal)
            setDeterminateProgress({ done: boundedDone, total: itemsTotal })
            if (safePhase === 'context_engineering') {
              setCeLayerProgress({
                done: Math.min(boundedDone, CE_LEN),
                total: Math.max(itemsTotal, CE_LEN)
              })
            } else if (safePhase !== 'context_engineering') {
              setCeLayerProgress({ done: 0, total: CE_LEN })
            }
            if (safePhase === 'analysis') {
              setAnalysisTotal(itemsTotal)
            }
          } else {
            setDeterminateProgress(null)
          }

          if (eta !== null) {
            setEtaSeconds(eta)
          }

          if (safePhase) {
            setCurrentPhase(safePhase)
          }

          const sourcesFound = toNonNegativeInt(data.sources_found ?? data.total_sources)
          const sourcesAnalyzed = toNonNegativeInt(data.sources_analyzed)
          const searchesCompleted = toNonNegativeInt(data.searches_completed)
          const totalSearches = toNonNegativeInt(data.total_searches)
          const highQuality = toNonNegativeInt(data.high_quality_sources)
          const duplicates = toNonNegativeInt(data.duplicates_removed)
          const mcpToolsUsed = toNonNegativeInt((rawData as Record<string, unknown>).mcp_tools_used)

          if ([sourcesFound, sourcesAnalyzed, searchesCompleted, totalSearches, highQuality, duplicates, mcpToolsUsed].some(v => v !== null)) {
            setStats(prev => {
              const next = { ...prev, apisQueried: new Set(prev.apisQueried) }
              let changed = false
              if (sourcesFound !== null && sourcesFound !== prev.sourcesFound) {
                next.sourcesFound = sourcesFound
                changed = true
              }
              if (sourcesAnalyzed !== null && sourcesAnalyzed !== prev.sourcesAnalyzed) {
                next.sourcesAnalyzed = sourcesAnalyzed
                changed = true
              }
              if (searchesCompleted !== null && searchesCompleted !== prev.searchesCompleted) {
                next.searchesCompleted = searchesCompleted
                changed = true
              }
              if (totalSearches !== null && totalSearches !== prev.totalSearches) {
                next.totalSearches = totalSearches
                changed = true
              }
              if (highQuality !== null && highQuality !== prev.highQualitySources) {
                next.highQualitySources = highQuality
                changed = true
              }
              if (duplicates !== null && duplicates !== prev.duplicatesRemoved) {
                next.duplicatesRemoved = duplicates
                changed = true
              }
              if (mcpToolsUsed !== null && mcpToolsUsed !== prev.mcpToolsUsed) {
                next.mcpToolsUsed = mcpToolsUsed
                changed = true
              }
              return changed ? next : prev
            })
          }
          break
        }
        case 'system.notification': {
          const server = typeof data.server === 'string' ? stripHtml(data.server) : ''
          const tool = typeof data.tool === 'string' ? stripHtml(data.tool) : ''
          if (server && tool) {
            const phase = data.status ? 'completed' : 'executing'
            statusUpdate = { message: `Tool: MCP ${server}.${tool} ${phase}` }
          } else {
            const messageText = typeof data.message === 'string' ? stripHtml(data.message) : undefined
            statusUpdate = { message: messageText }
          }
          break
        }
        case 'rate_limit.warning': {
          const warningMessage = typeof data.message === 'string' ? stripHtml(data.message) : undefined
          const limitType = typeof data.limit_type === 'string' ? data.limit_type : undefined
          statusUpdate = {
            message: `Warning: Rate limit warning: ${warningMessage || limitType || 'limit approaching'}`
          }
          setRateLimitWarning({
            message: warningMessage,
            limit_type: limitType,
            remaining: toNonNegativeInt(data.remaining) ?? undefined,
            reset_time: typeof data.reset_time === 'string' ? data.reset_time : undefined
          })
          break
        }
        case 'research_phase_change': {
          const oldPhase = typeof data.old_phase === 'string' ? stripHtml(data.old_phase) : 'unknown'
          const newPhase = typeof data.new_phase === 'string' ? stripHtml(data.new_phase) : 'unknown'
          statusUpdate = {
            message: `Phase changed: ${oldPhase} -> ${newPhase}`,
            phase: typeof data.new_phase === 'string' ? data.new_phase : undefined
          }
          if (typeof data.new_phase === 'string') {
            setCurrentPhase(data.new_phase)
          }
          setDeterminateProgress(null)
          setEtaSeconds(null)
          break
        }
        case 'source_found': {
          const sourceInfo = data.source && typeof data.source === 'object'
            ? (data.source as Record<string, unknown>)
            : undefined
          const title = sourceInfo && typeof sourceInfo.title === 'string'
            ? stripHtml(sourceInfo.title)
            : 'New source'
          const totalSources = toNonNegativeInt(data.total_sources)
          statusUpdate = {
            message: `Found source: ${title}${totalSources !== null ? ` (${totalSources} total)` : ''}`
          }
          if (totalSources !== null) {
            setStats(prev => {
              if (prev.sourcesFound === totalSources) return prev
              return { ...prev, apisQueried: new Set(prev.apisQueried), sourcesFound: totalSources }
            })
          }
          if (sourceInfo) {
            const domain = typeof sourceInfo.domain === 'string' ? stripHtml(sourceInfo.domain) : ''
            const snippet = typeof sourceInfo.snippet === 'string' ? stripHtml(sourceInfo.snippet) : undefined
            const credibilityScore = toFiniteNumber(sourceInfo.credibility_score)
            const safeCredibility = credibilityScore !== null
              ? Math.max(0, Math.min(1, credibilityScore))
              : undefined
            setSourcePreviews(prev => {
              const trimmed = MAX_SOURCE_PREVIEWS > 1 ? prev.slice(-(MAX_SOURCE_PREVIEWS - 1)) : []
              return [
                ...trimmed,
                {
                  title,
                  domain,
                  snippet,
                  credibility: safeCredibility
                }
              ]
            })
          }
          break
        }
        case 'source_analyzed': {
          const analyzedCount = toNonNegativeInt(data.analyzed_count)
          const sourceId = typeof data.source_id === 'string' ? stripHtml(data.source_id) : ''
          statusUpdate = {
            message: analyzedCount !== null
              ? `Analyzed source ${sourceId || 'n/a'} (${analyzedCount} analyzed)`
              : `Analyzed source ${sourceId || 'n/a'}`
          }
          if (analyzedCount !== null) {
            setStats(prev => {
              if (prev.sourcesAnalyzed === analyzedCount) return prev
              return { ...prev, apisQueried: new Set(prev.apisQueried), sourcesAnalyzed: analyzedCount }
            })
          }
          break
        }
        case 'research_completed': {
          const duration = toNonNegativeInt((rawData as Record<string, unknown>).duration_seconds)
          statusUpdate = {
            status: 'completed',
            progress: 100,
            message: duration !== null ? `Research completed in ${duration}s` : 'Research completed'
          }
          break
        }
        case 'research_failed': {
          const errorMsg = typeof data.error === 'string' ? stripHtml(data.error) : 'Research failed'
          statusUpdate = {
            status: 'failed',
            message: errorMsg
          }
          // Ensure we trigger completion callback on failure
          if (onCompleteRef.current) {
            setTimeout(() => onCompleteRef.current?.(), 100)
          }
          break
        }
        case 'evidence_builder_skipped': {
          // Handle the case where evidence builder is skipped due to no results
          statusUpdate = {
            status: 'failed',
            message: 'No search results available for evidence building'
          }
          // Trigger completion to transition to results display
          if (onCompleteRef.current) {
            setTimeout(() => onCompleteRef.current?.(), 100)
          }
          break
        }
        case 'research_started': {
          const query = typeof data.query === 'string' ? stripHtml(data.query) : ''
          statusUpdate = {
            status: 'processing',
            message: query ? `Research started for query: ${query}` : 'Research started'
          }
          if (!startTimeRef.current) {
            setStartTime(Date.now())
          }
          break
        }
        case 'search.started': {
          const index = toNonNegativeInt(data.index)
          const total = toNonNegativeInt(data.total)
          const query = typeof data.query === 'string' ? stripHtml(data.query) : ''
          const prefixParts: string[] = ['Searching']
          if (index !== null && total !== null && total > 0) {
            const boundedIndex = Math.min(index + 1, total)
            prefixParts.push(`(${boundedIndex}/${total})`)
          }
          if (query) {
            prefixParts.push(`: ${query}`)
          }
          statusUpdate = { message: prefixParts.join(' ') }

          const rawApi = typeof data.engine === 'string' && data.engine.trim()
            ? data.engine
            : typeof data.api === 'string'
              ? data.api
              : ''
          const apiName = rawApi ? stripHtml(rawApi) : ''

          if (apiName || (total !== null && total > 0)) {
            setStats(prev => {
              let changed = false
              let nextTotal = prev.totalSearches
              let nextApis = prev.apisQueried

              if (apiName) {
                const updatedApis = new Set(prev.apisQueried)
                if (!updatedApis.has(apiName)) {
                  changed = true
                }
                updatedApis.add(apiName)
                if (updatedApis.size > MAX_API_BADGES) {
                  const trimmed = new Set(Array.from(updatedApis).slice(-MAX_API_BADGES))
                  if (!areSetsEqual(trimmed, updatedApis)) {
                    changed = true
                  }
                  nextApis = trimmed
                } else if (changed) {
                  nextApis = updatedApis
                } else {
                  nextApis = updatedApis
                }
              }

              if (total !== null && total > 0 && total !== prev.totalSearches) {
                nextTotal = total
                changed = true
              }

              if (!changed) {
                return prev
              }

              return {
                ...prev,
                totalSearches: nextTotal,
                apisQueried: nextApis instanceof Set ? nextApis : new Set(nextApis)
              }
            })
          }
          break
        }
        case 'search.completed': {
          const results = toNonNegativeInt(data.results_count)
          statusUpdate = {
            message: results !== null ? `Found ${results} results` : 'Search completed'
          }
          // Update searches completed metric. Prefer an explicit count from the
          // backend if provided; otherwise assume each `search.completed` event
          // represents the completion of a single search and increment the
          // existing counter.  We also ensure we never exceed the configured
          // totalSearches so the “X / Y” display remains accurate.

          const completedCount = toNonNegativeInt(data.searches_completed)

          setStats(prev => {
            // Clone mutable members to avoid state mutations.
            const next = { ...prev, apisQueried: new Set(prev.apisQueried) }

            if (completedCount !== null) {
              // If the backend explicitly tells us how many searches have been
              // completed so far, trust that value.
              if (completedCount === prev.searchesCompleted) {
                return prev
              }
              next.searchesCompleted = completedCount
            } else {
              // Otherwise, increment by one for this event.
              const incremented = prev.searchesCompleted + 1
              // Bound by totalSearches when we know that number.
              const bounded = prev.totalSearches > 0 ? Math.min(incremented, prev.totalSearches) : incremented
              if (bounded === prev.searchesCompleted) {
                return prev
              }
              next.searchesCompleted = bounded
            }

            return next
          })
          break
        }
        case 'credibility.check': {
          const domain = typeof data.domain === 'string' ? stripHtml(data.domain) : 'source'
          const score = toFiniteNumber(data.score)
          const pct = score !== null ? `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%` : 'n/a'
          statusUpdate = {
            message: `Checking credibility: ${domain} (${pct})`
          }
          break
        }
        case 'deduplication.progress': {
          const removed = toNonNegativeInt(data.removed)
          statusUpdate = {
            message: removed !== null ? `Removed ${removed} duplicate results` : 'Deduplication in progress'
          }
          const before = toNonNegativeInt(data.before_count)
          const after = toNonNegativeInt(data.after_count)
          const totalRemoved = before !== null && after !== null
            ? Math.max(0, before - after)
            : removed
          if (totalRemoved !== null) {
            setStats(prev => {
              if (prev.duplicatesRemoved === totalRemoved) return prev
              return {
                ...prev,
                apisQueried: new Set(prev.apisQueried),
                duplicatesRemoved: totalRemoved
              }
            })
          }
          break
        }
        default: {
          const safeMessage = typeof data.message === 'string' ? stripHtml(data.message) : undefined
          statusUpdate = {
            status: normalizedFromData,
            progress: toFiniteNumber(data.progress) ?? undefined,
            message: safeMessage
          }
        }
      }

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
          case 'rate_limit.warning':
            return 'high'
          default:
            return 'low'
        }
      }

      const progressValue = statusUpdate?.progress ?? toFiniteNumber(data.progress)
      const normalizedStatus = statusUpdate?.status || normalizedFromData
      const messageText = statusUpdate?.message || (normalizedStatus ? `Status: ${normalizedStatus}` : 'Status update')

      const update: ProgressUpdate = {
        status: normalizedStatus || currentStatusRef.current,
        progress: progressValue !== null ? clampPercentage(progressValue) : undefined,
        message: messageText,
        timestamp: new Date().toISOString(),
        type: message.type,
        priority: toPriority(message.type),
        data: rawData
      }

      setUpdates(prev => [...prev.slice(-(MAX_UPDATES - 1)), update])

      if (normalizedStatus) {
        setCurrentStatus(normalizedStatus)
      }

      if (progressValue !== null) {
        setProgress(clampPercentage(progressValue))
      }

      if (normalizedStatus === 'completed' && onCompleteRef.current) {
        onCompleteRef.current()
      }

      if (normalizedStatus === 'cancelled' && onCancelRef.current) {
        onCancelRef.current()
      }
    })

    return () => {
      api.unsubscribeFromResearch(safeResearchId)
      if (timeoutTimerRef.current) {
        clearTimeout(timeoutTimerRef.current)
        timeoutTimerRef.current = null
      }
    }
  }, [safeResearchId, CE_LEN])

  // Elapsed time ticker
  useEffect(() => {
    if (!startTime) return
    const id = setInterval(() => {
      setElapsedSec(Math.max(0, Math.floor((Date.now() - startTime) / 1000)))
    }, 1000)
    return () => clearInterval(id)
  }, [startTime])

  // (auto-scroll moved into EventLog component)

  const handleCancel = async () => {
    setIsCancelling(true)
    try {
      if (!safeResearchId) {
        setValidationError('Cannot cancel research because the identifier is invalid.')
        return
      }
      await api.cancelResearch(safeResearchId)
      // Status will be updated via WebSocket
    } catch {
      setValidationError('Failed to cancel the research run. Please retry.')
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

  type CategoryKey = 'all' | 'search' | 'sources' | 'analysis' | 'system' | 'errors'

  const canCancel = () => {
    return currentStatus === 'processing' || currentStatus === 'in_progress' || currentStatus === 'pending'
  }

  // Memoized derived displays
  const formatCount = (value: number): string => value.toLocaleString()
  const hasStarted = !validationError && (startTime !== null || updates.length > 0 || currentStatus !== 'pending')
  const boundedSearchesCompleted = useMemo(() => (
    stats.totalSearches > 0 ? Math.min(stats.searchesCompleted, stats.totalSearches) : stats.searchesCompleted
  ), [stats.searchesCompleted, stats.totalSearches])
  const searchesDisplay = useMemo(() => (
    stats.totalSearches > 0 ? `${boundedSearchesCompleted}/${stats.totalSearches}` : (hasStarted ? formatCount(stats.searchesCompleted) : '-')
  ), [boundedSearchesCompleted, stats.totalSearches, stats.searchesCompleted, hasStarted])
  const analyzedDisplay = useMemo(() => (
    analysisTotal && analysisTotal > 0 ? `${Math.min(stats.sourcesAnalyzed, analysisTotal)}/${analysisTotal}` : (hasStarted ? formatCount(stats.sourcesAnalyzed) : '-')
  ), [analysisTotal, stats.sourcesAnalyzed, hasStarted])
  const sourcesFoundDisplay = useMemo(() => (hasStarted ? formatCount(stats.sourcesFound) : '-'), [hasStarted, stats.sourcesFound])
  const highQualityDisplay = useMemo(() => (hasStarted ? formatCount(stats.highQualitySources) : '-'), [hasStarted, stats.highQualitySources])
  const duplicatesDisplay = useMemo(() => (hasStarted ? formatCount(stats.duplicatesRemoved) : '-'), [hasStarted, stats.duplicatesRemoved])
  const mcpToolsDisplay = useMemo(() => (hasStarted ? formatCount(stats.mcpToolsUsed) : '-'), [hasStarted, stats.mcpToolsUsed])
  const elapsedDisplay = useMemo(() => (
    hasStarted ? `${Math.floor(elapsedSec / 60).toString().padStart(2, '0')}:${(elapsedSec % 60).toString().padStart(2, '0')}` : '--:--'
  ), [elapsedSec, hasStarted])
  const qualityRateDisplay = useMemo(() => (
    stats.sourcesFound > 0
      ? `${Math.round((stats.highQualitySources / Math.max(1, stats.sourcesFound)) * 100)}%`
      : hasStarted && stats.highQualitySources > 0
        ? '100%'
        : hasStarted ? '0%' : '-'
  ), [stats.sourcesFound, stats.highQualitySources, hasStarted])

  useEffect(() => {
    if (!updates.length) return
    const latest = updates[updates.length - 1]
    const stamp = new Date(latest.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    const statusText = latest.status ? latest.status.replace(/_/g, ' ') : currentStatus
    const announcement = latest.message ? `${statusText}: ${latest.message}` : statusText
    setLiveAnnouncement(`${stamp} ${announcement}`.trim())
  }, [updates, currentStatus])

  const showSearchMeta = currentPhase === 'search' && stats.totalSearches > 0
  const showAnalysisMeta = currentPhase === 'analysis' && !!analysisTotal && analysisTotal > 0

  return (
    <Card className="mt-6 animate-slide-up">
      <div className="sr-only" aria-live="polite">{liveAnnouncement}</div>
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
      {validationError && (
        <div className="mb-3 p-3 rounded-lg border border-error/40 bg-error/10 text-error text-sm">
          {validationError}
        </div>
      )}
      {pollingTimeout && (
        <div className="mb-3 p-3 rounded-lg border border-warning/40 bg-warning/10 text-warning text-sm">
          <div className="font-medium">Research timeout</div>
          <div className="text-xs">
            The research appears to have stalled. This may be due to insufficient search results or a processing error.
          </div>
        </div>
      )}
      {rateLimitWarning && (
        <div className="mb-3 p-3 rounded-lg border border-error/30 bg-error/10 text-error text-sm">
          <div className="font-medium">Rate limit warning</div>
          <div className="text-xs">
            {rateLimitWarning.message || (rateLimitWarning.limit_type ? `Approaching ${rateLimitWarning.limit_type} limit.` : 'Approaching a rate limit.')}
            {typeof rateLimitWarning.remaining === 'number' ? ` ${rateLimitWarning.remaining} remaining.` : ''}
          </div>
        </div>
      )}

      {/* API status indicators */}
      {stats.apisQueried.size > 0 && (
        <div className="mb-3 space-y-1">
          <div className="flex gap-2 flex-wrap">
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
          <p className="text-xs text-text-muted">{stats.apisQueried.size} API{stats.apisQueried.size === 1 ? '' : 's'} queried</p>
        </div>
      )}

      {(currentStatus === 'processing' || currentStatus === 'in_progress') && (
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
          {(currentPhase || showSearchMeta || showAnalysisMeta) && (
            <div className="text-xs text-text-muted flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
              {currentPhase && (
                <span>Current phase: {currentPhase.replace(/_/g, ' ')}</span>
              )}
              {showSearchMeta && (
                <span>Search progress: {boundedSearchesCompleted} of {stats.totalSearches}</span>
              )}
              {showAnalysisMeta && analysisTotal && (
                <span>Analysis progress: {Math.min(stats.sourcesAnalyzed, analysisTotal)} of {analysisTotal}</span>
              )}
            </div>
          )}
        </div>
      )}

      {hasStarted && (
        <>
          <PhaseTracker currentPhase={currentPhase} currentStatus={currentStatus} ceLayerProgress={ceLayerProgress} />
          <ResearchStats
            sourcesFound={sourcesFoundDisplay}
            searches={searchesDisplay}
            analyzed={analyzedDisplay}
            highQuality={highQualityDisplay}
            qualityRate={qualityRateDisplay}
            duplicates={duplicatesDisplay}
            toolsExecuted={mcpToolsDisplay}
            elapsed={elapsedDisplay}
          />
        </>
      )}

      {/* Category filters and event stream */}
      <EventLog
        updates={updates as unknown as LogUpdate[]}
        isMobile={isMobile}
        showVerbose={showVerbose}
        activeCategory={activeCategory as LogCategory}
        onCategoryChange={(k) => setActiveCategory(k as CategoryKey)}
      />

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
          <Button
            variant="ghost"
            fullWidth
            className="flex items-center justify-between py-2"
            onClick={() => setSourcesCollapsed(v => !v)}
            aria-expanded={!sourcesCollapsed}
          >
            <h4 className="text-sm font-medium text-text flex-1 text-left">Recent Sources</h4>
            <div className="flex items-center gap-2">
              <span className="text-xs text-text-muted">{sourcePreviews.length}</span>
              {sourcesCollapsed ? <FiChevronDown className="h-4 w-4" /> : <FiChevronUp className="h-4 w-4" />}
            </div>
          </Button>
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
