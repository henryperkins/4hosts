import React, { useEffect, useState, useRef } from 'react'
import { Activity, CheckCircle, AlertCircle, Clock, Loader2, X, XCircle } from 'lucide-react'
import { format } from 'date-fns'
import { Button } from './ui/Button'

interface ResearchProgressProps {
  researchId: string
  onComplete?: () => void
  onCancel?: () => void
}

interface ProgressUpdate {
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'
  progress?: number
  message?: string
  timestamp: string
}

interface WebSocketData {
  status?: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'
  progress?: number
  message?: string
}

export const ResearchProgress: React.FC<ResearchProgressProps> = ({ researchId, onComplete, onCancel }) => {
  const [updates, setUpdates] = useState<ProgressUpdate[]>([])
  const [currentStatus, setCurrentStatus] = useState<ProgressUpdate['status']>('pending')
  const [progress, setProgress] = useState(0)
  const [isConnecting, setIsConnecting] = useState(true)
  const [isCancelling, setIsCancelling] = useState(false)
  const updatesContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Import api here to use WebSocket
    import('../services/api').then(({ default: api }) => {
      setIsConnecting(true)
      
      api.connectWebSocket(researchId, (message) => {
        setIsConnecting(false)
        const data = message.data as WebSocketData
        const update: ProgressUpdate = {
          status: data.status || currentStatus,
          progress: data.progress,
          message: data.message,
          timestamp: new Date().toISOString(),
        }

        setUpdates(prev => {
          const newUpdates = [...prev, update]
          // Auto-scroll to bottom when new update arrives
          setTimeout(() => {
            if (updatesContainerRef.current) {
              updatesContainerRef.current.scrollTop = updatesContainerRef.current.scrollHeight
            }
          }, 100)
          return newUpdates
        })
        
        if (data.status) {
          setCurrentStatus(data.status)
        }
        
        if (data.progress !== undefined) {
          setProgress(data.progress)
        }

        if (data.status === 'completed' && onComplete) {
          onComplete()
        }
        
        if (data.status === 'cancelled' && onCancel) {
          onCancel()
        }
      })

      return () => {
        api.unsubscribeFromResearch(researchId)
      }
    })
  }, [researchId, currentStatus, onComplete, onCancel])

  const handleCancel = async () => {
    setIsCancelling(true)
    try {
      // Import api here for cancel function
      const { default: api } = await import('../services/api')
      await api.cancelResearch(researchId)
      // Status will be updated via WebSocket
    } catch (error) {
      console.error('Failed to cancel research:', error)
      // You might want to show a toast error here
    } finally {
      setIsCancelling(false)
    }
  }

  const getStatusIcon = () => {
    if (isConnecting) {
      return <Loader2 className="h-5 w-5 text-gray-500 dark:text-gray-400 animate-spin" />
    }
    
    switch (currentStatus) {
      case 'pending':
        return <Clock className="h-5 w-5 text-gray-500 dark:text-gray-400 animate-pulse" />
      case 'processing':
        return <Activity className="h-5 w-5 text-blue-500 dark:text-blue-400 animate-spin" />
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500 dark:text-green-400" />
      case 'failed':
        return <AlertCircle className="h-5 w-5 text-red-500 dark:text-red-400" />
      case 'cancelled':
        return <XCircle className="h-5 w-5 text-orange-500 dark:text-orange-400" />
    }
  }

  const getStatusColor = () => {
    if (isConnecting) {
      return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
    }
    
    switch (currentStatus) {
      case 'pending':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
      case 'processing':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-200'
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-200'
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-200'
      case 'cancelled':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-200'
    }
  }

  const canCancel = () => {
    return currentStatus === 'processing' || currentStatus === 'pending'
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mt-6 transition-colors duration-200 animate-slide-up">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Research Progress</h3>
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
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm transition-all duration-300 ${getStatusColor()}`}>
            {getStatusIcon()}
            <span className="capitalize">
              {isConnecting ? 'Connecting...' : currentStatus}
            </span>
          </div>
        </div>
      </div>

      {progress > 0 && currentStatus === 'processing' && (
        <div className="mb-4">
          <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
            <span>Progress</span>
            <span className="font-medium">{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
            <div
              className="bg-blue-600 dark:bg-blue-500 h-2 rounded-full transition-all duration-500 ease-out relative"
              style={{ width: `${progress}%` }}
            >
              <div className="absolute inset-0 bg-linear-to-r from-transparent to-white/20 animate-shimmer" />
            </div>
          </div>
        </div>
      )}

      <div 
        ref={updatesContainerRef}
        className="space-y-2 max-h-64 overflow-y-auto scroll-smooth scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600 scrollbar-track-transparent"
        role="log"
        aria-label="Research progress updates"
        aria-live="polite"
      >
        {updates.map((update, index) => (
          <div
            key={index}
            className="flex items-start gap-3 text-sm py-2 border-b border-gray-100 dark:border-gray-700 last:border-0 animate-slide-up"
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            <span className="text-gray-500 dark:text-gray-500 text-xs whitespace-nowrap font-mono">
              {format(new Date(update.timestamp), 'HH:mm:ss')}
            </span>
            <p className="text-gray-700 dark:text-gray-300 flex-1">
              {update.message || `Status: ${update.status}`}
            </p>
          </div>
        ))}
      </div>

      {updates.length === 0 && !isConnecting && (
        <div className="text-center py-8 animate-fade-in">
          <Clock className="h-12 w-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
          <p className="text-gray-500 dark:text-gray-400">
            Waiting for updates...
          </p>
        </div>
      )}
      
      {updates.length === 0 && isConnecting && (
        <div className="text-center py-8 animate-fade-in">
          <Loader2 className="h-12 w-12 text-gray-300 dark:text-gray-600 mx-auto mb-3 animate-spin" />
          <p className="text-gray-500 dark:text-gray-400">
            Connecting to research stream...
          </p>
        </div>
      )}
    </div>
  )
}