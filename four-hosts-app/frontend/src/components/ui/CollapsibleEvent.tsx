import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiChevronDown, FiChevronUp, FiInfo } from 'react-icons/fi'
import { clsx } from 'clsx'

interface CollapsibleEventProps {
  message: string
  timestamp: string
  details?: Record<string, unknown>
  priority?: 'low' | 'medium' | 'high' | 'critical'
  type?: string
  icon?: React.ReactNode
  className?: string
}

export const CollapsibleEvent: React.FC<CollapsibleEventProps> = ({
  message,
  timestamp,
  details,
  priority,
  type,
  icon,
  className
}) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const hasDetails = details && Object.keys(details).length > 0

  const getPriorityStyle = () => {
    switch (priority) {
      case 'critical':
        return 'border-l-4 border-error bg-error/5'
      case 'high':
        return 'border-l-4 border-primary bg-primary/5'
      case 'medium':
        return 'border-l-2 border-warning bg-warning/5'
      default:
        return ''
    }
  }

  const formatDetails = (data: unknown): string => {
    try {
      if (data && typeof data === 'object') {
        const cleaned = { ...(data as Record<string, unknown>) }
        delete (cleaned as Record<string, unknown>).timestamp
        delete (cleaned as Record<string, unknown>).message
        delete (cleaned as Record<string, unknown>).status
        return JSON.stringify(cleaned, null, 2)
      }
      return String(data)
    } catch {
      return String(data)
    }
  }

  return (
    <div
      className={clsx(
        'text-sm border-b border-border last:border-0 animate-slide-up',
        getPriorityStyle(),
        className
      )}
    >
      <div
        className={clsx(
          'flex items-start gap-3 py-2 px-2',
          hasDetails && 'cursor-pointer hover:bg-surface-subtle/50'
        )}
        onClick={() => hasDetails && setIsExpanded(!isExpanded)}
      >
        <span className="text-text-muted text-xs font-mono">
          {timestamp}
        </span>

        <div className="flex-1 min-w-0">
          <div className="flex items-start gap-2">
            <p className="text-text flex-1">
              {message}
            </p>
            {icon}
          </div>

          {type && (
            <span className="text-xs text-text-muted mt-1 inline-block">
              {type.replace(/_/g, ' ')}
            </span>
          )}
        </div>

        {hasDetails && (
          <button
            className="p-1 hover:bg-surface-muted rounded transition-colors touch-target"
            onClick={(e) => {
              e.stopPropagation()
              setIsExpanded(!isExpanded)
            }}
            aria-label={isExpanded ? 'Collapse details' : 'Expand details'}
          >
            {isExpanded ? (
              <FiChevronUp className="h-4 w-4 text-text-muted" />
            ) : (
              <FiChevronDown className="h-4 w-4 text-text-muted" />
            )}
          </button>
        )}
      </div>

      <AnimatePresence>
        {isExpanded && hasDetails && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-3">
              <div className="flex items-center gap-1 mb-2">
                <FiInfo className="h-3 w-3 text-text-muted" />
                <span className="text-xs font-medium text-text-muted">
                  Event Details
                </span>
              </div>
              <pre className="text-xs bg-surface-subtle p-3 rounded-lg overflow-x-auto text-text-muted">
                {formatDetails(details)}
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
