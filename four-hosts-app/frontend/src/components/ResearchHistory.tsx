import React, { useEffect, useState } from 'react'
import { FiClock, FiSearch, FiTrendingUp, FiEye, FiCalendar, FiChevronRight, FiX } from 'react-icons/fi'
import { format, formatDistanceToNow } from 'date-fns'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'
import type { ResearchHistoryItem } from '../types'
import { paradigmInfo, getParadigmClass, type Paradigm } from '../constants/paradigm'
import { Button } from './ui/Button'
import { Card } from './ui/Card'
import { StatusIcon, type StatusType } from './ui/StatusIcon'
import { ProgressBar } from './ui/ProgressBar'
import { LoadingSpinner } from './ui/LoadingSpinner'

export const ResearchHistory: React.FC = () => {
  const [history, setHistory] = useState<ResearchHistoryItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [hasMore, setHasMore] = useState(true)
  const [offset, setOffset] = useState(0)
  const [hoveredItem, setHoveredItem] = useState<string | null>(null)
  const [cancellingItems, setCancellingItems] = useState<Set<string>>(new Set())
  const navigate = useNavigate()
  const limit = 10

  useEffect(() => {
    const loadHistoryOnMount = async () => {
      try {
        const response = await api.getUserResearchHistory(limit, 0)
        if (response.history.length < limit) {
          setHasMore(false)
        }
        setHistory(response.history)
        setOffset(response.history.length)
      } catch {
        // Failed to load history
      } finally {
        setIsLoading(false)
      }
    }

    loadHistoryOnMount()
  }, [])

  const loadHistory = async () => {
    try {
      const response = await api.getUserResearchHistory(limit, offset)
      if (response.history.length < limit) {
        setHasMore(false)
      }
      setHistory(prev => [...prev, ...response.history])
      setOffset(prev => prev + response.history.length)
    } catch {
      // Failed to load history
    } finally {
      setIsLoading(false)
    }
  }

  const handleViewResult = (researchId: string) => {
    navigate(`/research/${researchId}`)
  }

  const handleCancelResearch = async (researchId: string, event: React.MouseEvent) => {
    event.stopPropagation() // Prevent clicking through to view result

    setCancellingItems(prev => new Set(prev).add(researchId))

    try {
      await api.cancelResearch(researchId)
      
      // Update the local state immediately
      setHistory(prev => prev.map(item => 
        item.research_id === researchId 
          ? { ...item, status: 'cancelled' }
          : item
      ))
    } catch {
      // Failed to cancel research
      // You might want to show a toast error here
    } finally {
      setCancellingItems(prev => {
        const newSet = new Set(prev)
        newSet.delete(researchId)
        return newSet
      })
    }
  }

  const canCancel = (status: string) => {
    return status === 'processing' || status === 'pending' || status === 'in_progress'
  }

  const getStatusType = (status: string): StatusType => {
    switch (status) {
      case 'completed':
        return 'completed'
      case 'failed':
        return 'failed'
      case 'cancelled':
        return 'cancelled'
      case 'processing':
        return 'processing'
      default:
        return 'pending'
    }
  }

  if (isLoading && history.length === 0) {
    return (
      <Card className="p-8">
        <LoadingSpinner 
          size="xl" 
          variant="primary" 
          text="Loading research history..."
          icon="ring"
        />
      </Card>
    )
  }

  if (history.length === 0) {
    return (
      <Card className="p-8 sm:p-12 text-center animate-fade-in">
        <FiSearch className="h-16 w-16 text-text-muted mx-auto mb-4 animate-pulse" />
        <h3 className="text-xl font-semibold text-text mb-2">No research history yet</h3>
        <p className="text-text-muted mb-6">Start exploring and your research queries will appear here</p>
        <Button
          onClick={() => navigate('/')}
          variant="primary"
        >
          Start Researching
        </Button>
      </Card>
    )
  }

  return (
    <Card className="p-4 sm:p-6 animate-fade-in">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-6">
        <h2 className="text-2xl font-bold text-text flex items-center gap-2">
          <FiClock className="h-6 w-6 text-primary" />
          Research History
        </h2>
        <span className="text-sm text-text-muted">
          {history.length} research{history.length !== 1 ? 'es' : ''}
        </span>
      </div>

      <div className="space-y-3">
        {history.map((item, index) => {
          const paradigm = item.paradigm && paradigmInfo[item.paradigm as Paradigm]
          const isHovered = hoveredItem === item.research_id

          return (
            <Card
              key={item.research_id}
              variant="interactive"
              paradigm={item.paradigm as Paradigm}
              className={`relative transition-all duration-300 cursor-pointer transform ${
                isHovered ? 'scale-[1.02] shadow-xl -translate-y-1' : ''
              } animate-slide-up stagger-delay-${index * 50}`}
              onClick={() => handleViewResult(item.research_id)}
              onMouseEnter={() => setHoveredItem(item.research_id)}
              onMouseLeave={() => setHoveredItem(null)}
            >
              <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div className="flex-1">
                  <div className="flex flex-wrap items-center gap-3 mb-2">
                    <StatusIcon 
                      status={getStatusType(item.status)}
                      size="sm"
                    />
                    <h3 className="font-semibold text-text line-clamp-1 text-lg">
                      {item.query}
                    </h3>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 sm:gap-4 text-sm text-text-muted">
                    <span className="flex items-center gap-1">
                      <FiCalendar className="h-3 w-3" />
                      {format(new Date(item.created_at), 'MMM d, yyyy')}
                    </span>

                    <span className="flex items-center gap-1">
                      <FiClock className="h-3 w-3" />
                      {formatDistanceToNow(new Date(item.created_at), { addSuffix: true })}
                    </span>

                    {paradigm && (
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${getParadigmClass(item.paradigm!)} flex items-center gap-1 animate-fade-in`}>
                          {paradigm.icon && (() => {
                            const Icon = paradigm.icon
                            return <Icon className="h-3 w-3" aria-hidden="true" />
                          })()}
                          {paradigm.shortDescription}
                        </span>
                    )}

                    {item.processing_time && (
                      <span className="flex items-center gap-1">
                        <FiTrendingUp className="h-3 w-3" />
                        {item.processing_time}s
                      </span>
                    )}
                    
                    {item.summary?.total_cost !== undefined && (
                      <span className="flex items-center gap-1 text-success">
                        ${item.summary.total_cost.toFixed(4)}
                      </span>
                    )}
                  </div>
                  
                  {item.summary && (
                    <div className="mt-2 space-y-1">
                      {item.summary.answer_preview && (
                        <p className="text-sm text-text-muted line-clamp-2">
                          {item.summary.answer_preview}
                        </p>
                      )}
                      {item.summary.source_count !== undefined && (
                        <p className="text-xs text-text-muted">
                          {item.summary.source_count} sources analyzed
                        </p>
                      )}
                    </div>
                  )}
                </div>

                <div className={`flex items-center gap-2 transition-all duration-300 opacity-100 translate-x-0 ${
                  isHovered ? 'md:opacity-100 md:translate-x-0' : 'md:opacity-0 md:-translate-x-2'
                }`}>
                  {canCancel(item.status) && (
                    <Button
                      variant="danger"
                      size="sm"
                      icon={FiX}
                      loading={cancellingItems.has(item.research_id)}
                      onClick={(e) => handleCancelResearch(item.research_id, e)}
                      className="text-xs mr-2"
                      title="Cancel research"
                    >
                      {cancellingItems.has(item.research_id) ? 'Cancelling...' : 'Cancel'}
                    </Button>
                  )}
                  <FiEye className="h-5 w-5 text-text-muted" />
                  <FiChevronRight className="h-5 w-5 text-text-muted" />
                </div>
              </div>

              {/* Progress bar for processing time */}
              {item.processing_time && (
                <div className="absolute bottom-0 left-0 right-0 rounded-b-xl overflow-hidden">
                  <ProgressBar
                    value={item.processing_time}
                    max={10}
                    size="sm"
                    variant="info"
                    animated={false}
                    className="rounded-none"
                  />
                </div>
              )}
            </Card>
          )
        })}
      </div>

      {hasMore && (
        <div className="mt-6 text-center animate-fade-in">
          <Button
            onClick={loadHistory}
            loading={isLoading}
            disabled={isLoading}
            variant="primary"
            icon={FiClock}
            className="mx-auto"
          >
            {isLoading ? 'Loading...' : 'Load More History'}
          </Button>
        </div>
      )}
    </Card>
  )
}
