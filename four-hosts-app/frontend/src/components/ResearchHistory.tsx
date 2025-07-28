import React, { useEffect, useState } from 'react'
import { Clock, Search, CheckCircle, XCircle, Loader, TrendingUp, Eye, Calendar, ChevronRight, X } from 'lucide-react'
import { format, formatDistanceToNow } from 'date-fns'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'
import type { ResearchHistoryItem } from '../types'
import { paradigmInfo, getParadigmClass, type Paradigm } from '../constants/paradigm'
import { Button } from './ui/Button'

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
        const items = await api.getUserResearchHistory(limit, 0)
        if (items.length < limit) {
          setHasMore(false)
        }
        setHistory(items)
        setOffset(items.length)
      } catch (error) {
        console.error('Failed to load history:', error)
      } finally {
        setIsLoading(false)
      }
    }

    loadHistoryOnMount()
  }, [])

  const loadHistory = async () => {
    try {
      const items = await api.getUserResearchHistory(limit, offset)
      if (items.length < limit) {
        setHasMore(false)
      }
      setHistory(prev => [...prev, ...items])
      setOffset(prev => prev + items.length)
    } catch (error) {
      console.error('Failed to load history:', error)
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
    } catch (error) {
      console.error('Failed to cancel research:', error)
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
    return status === 'processing' || status === 'pending'
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500 animate-scale-in" />
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500 animate-pulse" />
      case 'cancelled':
        return <XCircle className="h-4 w-4 text-orange-500 animate-scale-in" />
      case 'processing':
        return <Loader className="h-4 w-4 text-blue-500 animate-spin" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  if (isLoading && history.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-8 animate-pulse">
        <div className="flex items-center justify-center">
          <div className="relative">
            <Loader className="h-12 w-12 text-blue-600 dark:text-blue-400 animate-spin" />
            <div className="absolute inset-0 h-12 w-12 bg-blue-600 dark:bg-blue-400 rounded-full opacity-20 animate-ping"></div>
          </div>
        </div>
      </div>
    )
  }

  if (history.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-12 text-center animate-fade-in">
        <Search className="h-16 w-16 text-gray-400 dark:text-gray-600 mx-auto mb-4 animate-pulse" />
        <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">No research history yet</h3>
        <p className="text-gray-600 dark:text-gray-400 mb-6">Start exploring and your research queries will appear here</p>
        <button
          onClick={() => navigate('/')}
          className="px-6 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all duration-300 transform hover:scale-105 hover:shadow-lg"
        >
          Start Researching
        </button>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6 animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
          <Clock className="h-6 w-6 text-blue-600 dark:text-blue-400" />
          Research History
        </h2>
        <span className="text-sm text-gray-600 dark:text-gray-400">
          {history.length} research{history.length !== 1 ? 'es' : ''}
        </span>
      </div>

      <div className="space-y-3">
        {history.map((item, index) => {
          const paradigm = item.paradigm && paradigmInfo[item.paradigm as Paradigm]
          const isHovered = hoveredItem === item.research_id

          return (
            <div
              key={item.research_id}
              className={`relative border-2 ${paradigm ? paradigm.borderColor : 'border-gray-200 dark:border-gray-700'} rounded-xl p-4 transition-all duration-300 cursor-pointer transform ${
                isHovered ? 'scale-[1.02] shadow-xl -translate-y-1' : 'hover:shadow-lg'
              } animate-slide-up bg-gradient-to-r from-transparent ${
                paradigm ? `to-${item.paradigm}-50 dark:to-${item.paradigm}-900/10` : 'to-gray-50 dark:to-gray-900/10'
              }`}
              onClick={() => handleViewResult(item.research_id)}
              onMouseEnter={() => setHoveredItem(item.research_id)}
              onMouseLeave={() => setHoveredItem(null)}
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    {getStatusIcon(item.status)}
                    <h3 className="font-semibold text-gray-900 dark:text-gray-100 line-clamp-1 text-lg">
                      {item.query}
                    </h3>
                  </div>

                  <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                    <span className="flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      {format(new Date(item.created_at), 'MMM d, yyyy')}
                    </span>

                    <span className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {formatDistanceToNow(new Date(item.created_at), { addSuffix: true })}
                    </span>

                    {paradigm && (
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getParadigmClass(item.paradigm!)} flex items-center gap-1 animate-fade-in`}>
                        <span>{paradigm.icon}</span>
                        {paradigm.shortDescription}
                      </span>
                    )}

                    {item.processing_time && (
                      <span className="flex items-center gap-1">
                        <TrendingUp className="h-3 w-3" />
                        {item.processing_time}s
                      </span>
                    )}
                  </div>
                </div>

                <div className={`flex items-center gap-2 transition-all duration-300 ${
                  isHovered ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-2'
                }`}>
                  {canCancel(item.status) && (
                    <Button
                      variant="danger"
                      size="sm"
                      icon={X}
                      loading={cancellingItems.has(item.research_id)}
                      onClick={(e) => handleCancelResearch(item.research_id, e)}
                      className="text-xs mr-2"
                      title="Cancel research"
                    >
                      {cancellingItems.has(item.research_id) ? 'Cancelling...' : 'Cancel'}
                    </Button>
                  )}
                  <Eye className="h-5 w-5 text-gray-400 dark:text-gray-500" />
                  <ChevronRight className="h-5 w-5 text-gray-400 dark:text-gray-500" />
                </div>
              </div>

              {/* Progress bar for processing time */}
              {item.processing_time && (
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-200 dark:bg-gray-700 rounded-b-xl overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-1000 ease-out"
                    style={{
                      width: `${Math.min((item.processing_time / 10) * 100, 100)}%`,
                      animationDelay: `${index * 100}ms`
                    }}
                  />
                </div>
              )}
            </div>
          )
        })}
      </div>

      {hasMore && (
        <div className="mt-6 text-center animate-fade-in">
          <button
            onClick={loadHistory}
            disabled={isLoading}
            className="px-6 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all duration-300 transform hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 mx-auto"
          >
            {isLoading ? (
              <>
                <Loader className="h-4 w-4 animate-spin" />
                Loading...
              </>
            ) : (
              <>
                <Clock className="h-4 w-4" />
                Load More History
              </>
            )}
          </button>
        </div>
      )}
    </div>
  )
}
