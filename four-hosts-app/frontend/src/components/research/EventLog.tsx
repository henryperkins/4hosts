import React, { memo, useMemo, useLayoutEffect, useRef, useState } from 'react'
import { format } from 'date-fns'
import { Button } from '../ui/Button'
import { SwipeableTabs } from '../ui/SwipeableTabs'
import { CollapsibleEvent } from '../ui/CollapsibleEvent'
import {
  FiSearch,
  FiCheckCircle,
  FiDatabase,
  FiMaximize2,
  FiMinimize2,
} from 'react-icons/fi'

export type StatusType = 'pending' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
export type Priority = 'low' | 'medium' | 'high' | 'critical'
export type CategoryKey = 'all' | 'search' | 'sources' | 'analysis' | 'system' | 'errors'

export interface ProgressUpdate {
  status: StatusType
  progress?: number
  message?: string
  timestamp: string
  type?: string
  priority?: Priority
  data?: Record<string, unknown>
}

type EventLogProps = {
  updates: ProgressUpdate[]
  isMobile: boolean
  showVerbose: boolean
  activeCategory: CategoryKey
  onCategoryChange: (key: CategoryKey) => void
}

const cleanMessage = (msg?: string) => (typeof msg === 'string' ? msg.replace(/<[^>]*>/g, '') : '')
const isNoisy = (msg?: string) => {
  const m = (msg || '').toLowerCase()
  return m.includes('heartbeat') || m.includes('still processing')
}

const categorize = (u: ProgressUpdate): CategoryKey => {
  const t = (u.type || '').toLowerCase()
  if (t.includes('search.')) return 'search'
  if (t === 'source_found' || t === 'source_analyzed') return 'sources'
  if (t === 'research_phase_change' && (u.message || '').toLowerCase().includes('analysis')) return 'analysis'
  if (t === 'credibility.check' || t === 'deduplication.progress') return 'analysis'
  if (t === 'system.notification' || t === 'rate_limit.warning' || t === 'connected' || t === 'disconnected') return 'system'
  if (t === 'error' || t === 'research_failed') return 'errors'
  return 'all'
}

const getMessageStyle = (p?: Priority) => {
  switch (p) {
    case 'critical':
      return 'border-l-4 border-error bg-error/5'
    case 'high':
      return 'border-l-4 border-primary bg-primary/5'
    default:
      return ''
  }
}

export const EventLog = memo(function EventLog({ updates, isMobile, showVerbose, activeCategory, onCategoryChange }: EventLogProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  // Allow users to toggle between compact and expanded views for easier scrolling
  const [logExpanded, setLogExpanded] = useState(false)

  // Auto-scroll on new updates
  useLayoutEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [updates])

  const categoryCounts = useMemo((): Record<CategoryKey, number> => {
    const counts: Record<CategoryKey, number> = { all: updates.length, search: 0, sources: 0, analysis: 0, system: 0, errors: 0 }
    for (const u of updates) counts[categorize(u)] += 1
    return counts
  }, [updates])

  const filtered = useMemo(() =>
    updates
      .filter(u => showVerbose || !isNoisy(u.message))
      .filter(u => (activeCategory === 'all' ? true : categorize(u) === activeCategory))
  , [updates, showVerbose, activeCategory])

  const sizeClass = logExpanded ? 'max-h-[80vh]' : 'max-h-72 sm:max-h-[32rem]'

  const renderEvents = () => (
    filtered.map((update, index) => {
      let icon: React.ReactNode = null
      if (update.message?.includes('Searching')) icon = <FiSearch className="h-4 w-4 text-primary animate-pulse" />
      else if (update.message?.includes('credibility')) icon = <FiCheckCircle className="h-4 w-4 text-green-500" />
      else if (update.message?.includes('duplicate')) icon = <FiDatabase className="h-4 w-4 text-yellow-500" />
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
  )

  if (isMobile) {
    return (
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
        onTabChange={(key) => onCategoryChange(key as CategoryKey)}
      >
        <div
          ref={containerRef}
          className={`space-y-2 overflow-y-auto ${sizeClass}`}
          role="log"
          aria-label="Research progress updates"
          aria-live="polite"
        >
          {renderEvents()}
        </div>
      </SwipeableTabs>
    )
  }

  return (
    <>
      <div className="mb-2 flex flex-wrap gap-2 text-xs items-center">
        {([
          { key: 'all', label: `All (${categoryCounts.all})` },
          { key: 'search', label: `Search (${categoryCounts.search})` },
          { key: 'sources', label: `Sources (${categoryCounts.sources})` },
          { key: 'analysis', label: `Analysis (${categoryCounts.analysis})` },
          { key: 'system', label: `System (${categoryCounts.system})` },
          { key: 'errors', label: `Errors (${categoryCounts.errors})`, emphasize: categoryCounts.errors > 0 },
        ] as const).map(tab => (
          <Button
            key={tab.key}
            size="sm"
            variant={activeCategory === tab.key ? 'primary' : 'ghost'}
            className={'emphasize' in tab && tab.emphasize ? 'text-error' : ''}
            onClick={() => onCategoryChange(tab.key as CategoryKey)}
          >
            {tab.label}
          </Button>
        ))}

        {/* Spacer pushes expand toggle to the far right */}
        <span className="flex-1" />

        {/* Expand / collapse log height */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setLogExpanded(!logExpanded)}
          aria-label={logExpanded ? 'Collapse log' : 'Expand log'}
          className="p-1"
        >
          {logExpanded ? (
            <FiMinimize2 className="h-4 w-4" />
          ) : (
            <FiMaximize2 className="h-4 w-4" />
          )}
        </Button>
      </div>
      <div
        ref={containerRef}
        className={`space-y-2 overflow-y-auto ${sizeClass}`}
        role="log"
        aria-label="Research progress updates"
        aria-live="polite"
      >
        {renderEvents()}
      </div>
    </>
  )
})

export default EventLog
