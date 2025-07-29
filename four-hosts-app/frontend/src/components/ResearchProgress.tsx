import React, { useEffect, useState, useRef } from 'react'
import { Clock, X } from 'lucide-react'
import { format } from 'date-fns'
import { Button } from './ui/Button'
import { Card } from './ui/Card'
import { StatusBadge, type StatusType } from './ui/StatusIcon'
import { ProgressBar } from './ui/ProgressBar'
import { LoadingSpinner } from './ui/LoadingSpinner'

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

  const getStatusType = (): StatusType => {
    if (isConnecting) {
      return 'processing'
    }
    
    switch (currentStatus) {
      case 'pending':
        return 'pending'
      case 'processing':
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

  const canCancel = () => {
    return currentStatus === 'processing' || currentStatus === 'pending'
  }

  return (
    <Card className="p-6 mt-6 animate-slide-up">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text">Research Progress</h3>
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
          <StatusBadge
            status={getStatusType()}
            label={getStatusLabel()}
            variant="subtle"
            size="md"
          />
        </div>
      </div>

      {progress > 0 && currentStatus === 'processing' && (
        <ProgressBar
          value={progress}
          max={100}
          variant="default"
          showLabel
          label="Progress"
          shimmer
          className="mb-4"
        />
      )}

      <div 
        ref={updatesContainerRef}
        className="space-y-2 max-h-64 overflow-y-auto scroll-smooth scrollbar-thin"
        role="log"
        aria-label="Research progress updates"
        aria-live="polite"
      >
        {updates.map((update, index) => (
          <div
            key={index}
            className={`flex items-start gap-3 text-sm py-2 border-b border-border last:border-0 animate-slide-up stagger-delay-${Math.min(index * 50, 300)}`}
          >
            <span className="text-text-muted text-xs whitespace-nowrap font-mono">
              {format(new Date(update.timestamp), 'HH:mm:ss')}
            </span>
            <p className="text-text flex-1">
              {update.message || `Status: ${update.status}`}
            </p>
          </div>
        ))}
      </div>

      {updates.length === 0 && !isConnecting && (
        <div className="text-center py-8 animate-fade-in">
          <Clock className="h-12 w-12 text-text-muted/30 mx-auto mb-3" />
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
    </Card>
  )
}