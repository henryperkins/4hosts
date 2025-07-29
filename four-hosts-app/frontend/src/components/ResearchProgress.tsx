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
  status: 'pending' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
  progress?: number
  message?: string
  timestamp: string
}

interface WebSocketData {
  status?: 'pending' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
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
        const data = message.data as any
        
        // Handle different message types
        let statusUpdate: WebSocketData | undefined
        
        switch (message.type) {
          case 'research_progress':
            statusUpdate = {
              status: data.status,
              progress: data.progress,
              message: data.message || `Phase: ${data.phase || 'processing'}`
            }
            break
          case 'research_phase_change':
            statusUpdate = {
              message: `Phase changed: ${data.old_phase} → ${data.new_phase}`
            }
            break
          case 'source_found':
            statusUpdate = {
              message: `Found source: ${data.source?.title || 'New source'} (${data.total_sources} total)`
            }
            break
          case 'source_analyzed':
            statusUpdate = {
              message: `Analyzed source ${data.source_id} (${data.analyzed_count} analyzed)`
            }
            break
          case 'research_completed':
            statusUpdate = {
              status: 'completed',
              progress: 100,
              message: `Research completed in ${data.duration_seconds}s`
            }
            break
          case 'research_failed':
            statusUpdate = {
              status: 'failed',
              message: `Research failed: ${data.error}`
            }
            break
          case 'research_started':
            statusUpdate = {
              status: 'processing',
              message: `Research started for query: ${data.query}`
            }
            break
          default:
            // Handle legacy message format
            statusUpdate = {
              status: data.status,
              progress: data.progress,
              message: data.message
            }
        }
        
        const update: ProgressUpdate = {
          status: statusUpdate.status || currentStatus,
          progress: statusUpdate.progress,
          message: statusUpdate.message,
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

  const canCancel = () => {
    return currentStatus === 'processing' || currentStatus === 'in_progress' || currentStatus === 'pending'
  }

  return (
    <Card className="mt-6 animate-slide-up">
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

      {progress > 0 && (currentStatus === 'processing' || currentStatus === 'in_progress') && (
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
        className="space-y-2 max-h-64 overflow-y-auto"
        role="log"
        aria-label="Research progress updates"
        aria-live="polite"
      >
        {updates.map((update, index) => (
          <div
            key={index}
            className="flex items-start gap-3 text-sm py-2 border-b border-border last:border-0 animate-slide-up"
          >
            <span className="text-text-muted text-xs font-mono">
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
          <Clock className="h-12 w-12 text-text-muted opacity-30 mx-auto mb-3" />
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