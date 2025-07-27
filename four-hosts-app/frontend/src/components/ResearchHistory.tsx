import React, { useEffect, useState } from 'react'
import { Clock, Search, CheckCircle, XCircle, Loader } from 'lucide-react'
import { format } from 'date-fns'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'
import type { ResearchHistoryItem } from '../types'

const paradigmColors = {
  dolores: 'bg-red-100 text-red-800',
  teddy: 'bg-blue-100 text-blue-800',
  bernard: 'bg-green-100 text-green-800',
  maeve: 'bg-purple-100 text-purple-800',
}

export const ResearchHistory: React.FC = () => {
  const [history, setHistory] = useState<ResearchHistoryItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [hasMore, setHasMore] = useState(true)
  const [offset, setOffset] = useState(0)
  const navigate = useNavigate()
  const limit = 10

  useEffect(() => {
    loadHistory()
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

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />
      case 'processing':
        return <Loader className="h-4 w-4 text-blue-500 animate-spin" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  if (isLoading && history.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8">
        <div className="flex items-center justify-center">
          <Loader className="h-8 w-8 text-blue-600 animate-spin" />
        </div>
      </div>
    )
  }

  if (history.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8">
        <div className="text-center">
          <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No research history</h3>
          <p className="text-gray-600">Your research queries will appear here</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-bold text-gray-900 mb-6">Research History</h2>
      
      <div className="space-y-4">
        {history.map((item) => (
          <div
            key={item.research_id}
            className="border border-gray-200 rounded-lg p-4 hover:border-gray-300 transition-colors cursor-pointer"
            onClick={() => handleViewResult(item.research_id)}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  {getStatusIcon(item.status)}
                  <h3 className="font-medium text-gray-900 line-clamp-1">{item.query}</h3>
                </div>
                
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {format(new Date(item.created_at), 'MMM d, yyyy HH:mm')}
                  </span>
                  
                  {item.paradigm && (
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      paradigmColors[item.paradigm]
                    }`}>
                      {item.paradigm.charAt(0).toUpperCase() + item.paradigm.slice(1)}
                    </span>
                  )}
                  
                  {item.processing_time && (
                    <span>{item.processing_time}s</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {hasMore && (
        <div className="mt-6 text-center">
          <button
            onClick={loadHistory}
            disabled={isLoading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            {isLoading ? 'Loading...' : 'Load More'}
          </button>
        </div>
      )}
    </div>
  )
}