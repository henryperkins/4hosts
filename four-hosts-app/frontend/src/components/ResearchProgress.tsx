import React, { useEffect, useState } from 'react'
import { Activity, CheckCircle, AlertCircle, Clock } from 'lucide-react'
import { format } from 'date-fns'

interface ResearchProgressProps {
  researchId: string
  onComplete?: () => void
}

interface ProgressUpdate {
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress?: number
  message?: string
  timestamp: string
}

export const ResearchProgress: React.FC<ResearchProgressProps> = ({ researchId, onComplete }) => {
  const [updates, setUpdates] = useState<ProgressUpdate[]>([])
  const [currentStatus, setCurrentStatus] = useState<ProgressUpdate['status']>('pending')
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    // Import api here to use WebSocket
    import('../services/api').then(({ default: api }) => {
      api.connectWebSocket(researchId, (message) => {
        const update: ProgressUpdate = {
          status: message.data.status || currentStatus,
          progress: message.data.progress,
          message: message.data.message,
          timestamp: new Date().toISOString(),
        }

        setUpdates(prev => [...prev, update])
        
        if (message.data.status) {
          setCurrentStatus(message.data.status)
        }
        
        if (message.data.progress !== undefined) {
          setProgress(message.data.progress)
        }

        if (message.data.status === 'completed' && onComplete) {
          onComplete()
        }
      })

      return () => {
        api.unsubscribeFromResearch(researchId)
      }
    })
  }, [researchId, currentStatus, onComplete])

  const getStatusIcon = () => {
    switch (currentStatus) {
      case 'pending':
        return <Clock className="h-5 w-5 text-gray-500 animate-pulse" />
      case 'processing':
        return <Activity className="h-5 w-5 text-blue-500 animate-spin" />
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'failed':
        return <AlertCircle className="h-5 w-5 text-red-500" />
    }
  }

  const getStatusColor = () => {
    switch (currentStatus) {
      case 'pending':
        return 'bg-gray-100 text-gray-800'
      case 'processing':
        return 'bg-blue-100 text-blue-800'
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mt-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Research Progress</h3>
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${getStatusColor()}`}>
          {getStatusIcon()}
          <span className="capitalize">{currentStatus}</span>
        </div>
      </div>

      {progress > 0 && currentStatus === 'processing' && (
        <div className="mb-4">
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      <div className="space-y-2 max-h-64 overflow-y-auto">
        {updates.map((update, index) => (
          <div
            key={index}
            className="flex items-start gap-3 text-sm py-2 border-b border-gray-100 last:border-0"
          >
            <span className="text-gray-500 text-xs whitespace-nowrap">
              {format(new Date(update.timestamp), 'HH:mm:ss')}
            </span>
            <p className="text-gray-700 flex-1">{update.message || `Status: ${update.status}`}</p>
          </div>
        ))}
      </div>

      {updates.length === 0 && (
        <p className="text-center text-gray-500 py-8">
          Waiting for updates...
        </p>
      )}
    </div>
  )
}