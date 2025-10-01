import React, { useEffect, useMemo, useState, useCallback } from 'react'
import { FiAlertCircle, FiUser, FiClock, FiActivity } from 'react-icons/fi'
import api from '../services/api'
import type { TriageBoardSnapshot, TriageEntry, WebSocketMessage } from '../types/api-types'
import { useWebSocket } from '../hooks/useWebSocket'

const laneOrder = [
  'intake',
  'classification',
  'context',
  'search',
  'analysis',
  'synthesis',
  'review',
  'blocked',
  'done'
] as const

const laneLabels: Record<typeof laneOrder[number], string> = {
  intake: 'Intake',
  classification: 'Classification',
  context: 'Context Engineering',
  search: 'Search',
  analysis: 'Analysis',
  synthesis: 'Synthesis',
  review: 'Review',
  blocked: 'Needs Attention',
  done: 'Completed'
}

const priorityBadge: Record<string, string> = {
  high: 'bg-error/15 text-error border border-error/40',
  medium: 'bg-warning/15 text-warning border border-warning/40',
  low: 'bg-success/15 text-success border border-success/40'
}

function formatRelativeTime(timestamp: string | undefined): string {
  if (!timestamp) return '—'
  try {
    const now = Date.now()
    const ts = new Date(timestamp).getTime()
    if (Number.isNaN(ts)) return '—'
    const diff = Math.max(0, now - ts)
    const minutes = Math.round(diff / 60000)
    if (minutes < 1) return 'just now'
    if (minutes === 1) return '1 min ago'
    if (minutes < 60) return `${minutes} mins ago`
    const hours = Math.round(minutes / 60)
    if (hours === 1) return '1 hr ago'
    if (hours < 24) return `${hours} hrs ago`
    const days = Math.round(hours / 24)
    return `${days}d ago`
  } catch {
    return '—'
  }
}

function safeEntries(board: TriageBoardSnapshot | null | undefined, lane: string): TriageEntry[] {
  if (!board?.lanes) return []
  const entries = board.lanes[lane]
  if (!Array.isArray(entries)) return []
  return entries
}

const EmptyState: React.FC<{ message: string }> = ({ message }) => (
  <div className="flex h-full flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-surface-subtle p-6 text-sm text-text-muted">
    <FiActivity className="h-6 w-6" />
    <span>{message}</span>
  </div>
)

export const TriageBoard: React.FC = () => {
  const [board, setBoard] = useState<TriageBoardSnapshot | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadBoard = useCallback(async () => {
    try {
      const snapshot = await api.getTriageBoard()
      setBoard(snapshot)
      setError(null)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unable to load triage board'
      setError(msg)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadBoard()
    const interval = setInterval(loadBoard, 60_000)
    return () => clearInterval(interval)
  }, [loadBoard])

  const handleSocketMessage = useCallback((message: WebSocketMessage) => {
    if (message.type !== 'triage.board_update') return
    const lanes = message.data?.lanes
    if (!lanes || typeof lanes !== 'object') return

    setBoard((prev) => {
      const updated: TriageBoardSnapshot = {
        updated_at: typeof message.data.updated_at === 'string' ? message.data.updated_at : prev?.updated_at ?? new Date().toISOString(),
        entry_count: typeof message.data.entry_count === 'number' ? message.data.entry_count : Object.values(lanes).reduce((acc, list) => acc + (Array.isArray(list) ? list.length : 0), 0),
        lanes: lanes as Record<string, TriageEntry[]>
      }
      return updated
    })
  }, [])

  useWebSocket({ researchId: 'triage-board', onMessage: handleSocketMessage, enabled: true })

  const laneData = useMemo(() => {
    return laneOrder.map((lane) => ({
      key: lane,
      title: laneLabels[lane],
      items: safeEntries(board, lane)
    }))
  }, [board])

  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-surface-subtle p-6 shadow-sm">
        <div className="flex items-center gap-3 text-text-muted">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary/30 border-t-primary" />
          <span>Loading triage board…</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-xl border border-border bg-surface-subtle p-6 shadow-sm">
        <div className="flex items-center gap-3 text-error">
          <FiAlertCircle className="h-5 w-5" />
          <span>{error}</span>
        </div>
      </div>
    )
  }

  return (
    <section aria-label="Research triage board" className="space-y-3">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold text-text">Research Intake Board</h2>
          <p className="text-sm text-text-muted">Live view of queued and in-flight research requests.</p>
        </div>
        <div className="text-sm text-text-subtle">
          Last update: {formatRelativeTime(board?.updated_at)}
        </div>
      </header>
      <div className="grid grid-cols-1 gap-4 overflow-x-auto lg:grid-cols-3 xl:grid-cols-4">
        {laneData.map((lane) => (
          <div key={lane.key} className="flex min-h-[260px] flex-col rounded-xl border border-border bg-surface shadow-sm">
            <div className="flex items-center justify-between border-b border-border/80 px-4 py-3">
              <div>
                <h3 className="text-sm font-semibold text-text">{lane.title}</h3>
                <p className="text-xs text-text-muted">{lane.items.length} {lane.items.length === 1 ? 'request' : 'requests'}</p>
              </div>
            </div>
            <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-3">
              {lane.items.length === 0 ? (
                <EmptyState message={lane.key === 'done' ? 'No recently completed work' : 'No items in this lane'} />
              ) : (
                lane.items.map((item) => (
                  <article key={item.research_id} className="rounded-lg border border-border bg-surface-subtle p-3 shadow-sm">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold uppercase tracking-wide ${priorityBadge[item.priority] || priorityBadge.low}`}>
                        {item.priority}
                      </span>
                      <span className="text-[11px] uppercase tracking-wide text-text-subtle">score {item.score.toFixed(1)}</span>
                    </div>
                    <p className="text-sm font-medium text-text line-clamp-3" title={item.query}>{item.query}</p>
                    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-text-subtle">
                      <span className="inline-flex items-center gap-1"><FiUser className="h-3.5 w-3.5" />{item.user_role}</span>
                      <span className="inline-flex items-center gap-1"><FiActivity className="h-3.5 w-3.5" />{item.depth}</span>
                      <span className="inline-flex items-center gap-1"><FiClock className="h-3.5 w-3.5" />{formatRelativeTime(item.updated_at)}</span>
                    </div>
                  </article>
                ))
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export default TriageBoard

