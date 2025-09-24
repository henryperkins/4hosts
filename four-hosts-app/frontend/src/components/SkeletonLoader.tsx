import React from 'react'

interface SkeletonLoaderProps {
  type?: 'text' | 'button' | 'card' | 'form' | 'result'
  count?: number
  className?: string
}

export const SkeletonLoader: React.FC<SkeletonLoaderProps> = ({
  type = 'text',
  count = 1,
  className = ''
}) => {
  const renderSkeleton = () => {
    switch (type) {
      case 'text':
        return (
          <div 
            className={`h-4 bg-surface-muted rounded animate-shimmer ${className}`} 
            role="status"
            aria-label="Loading text content"
          >
            <span className="sr-only">Loading...</span>
          </div>
        )

      case 'button':
        return (
          <div 
            className={`h-10 w-32 bg-surface-muted rounded-lg animate-shimmer ${className}`}
            role="status"
            aria-label="Loading button"
          >
            <span className="sr-only">Loading...</span>
          </div>
        )

      case 'card':
        return (
          <div 
            className={`card animate-pulse ${className}`}
            role="status"
            aria-label="Loading card content"
          >
            <div className="h-6 w-3/4 bg-surface-muted rounded mb-4 animate-shimmer" />
            <div className="space-y-2">
              <div className="h-4 w-full bg-surface-muted rounded animate-shimmer" />
              <div className="h-4 w-5/6 bg-surface-muted rounded animate-shimmer" style={{ animationDelay: '0.1s' }} />
              <div className="h-4 w-4/6 bg-surface-muted rounded animate-shimmer" style={{ animationDelay: '0.2s' }} />
            </div>
            <span className="sr-only">Loading...</span>
          </div>
        )

      case 'form':
        return (
          <div 
            className={`card animate-pulse ${className}`}
            role="status"
            aria-label="Loading form"
          >
            <div className="h-6 w-1/3 bg-surface-muted/80 dark:bg-surface-subtle/70 rounded mb-4 animate-shimmer" />
            <div className="space-y-4">
              <div>
                <div className="h-4 w-1/4 bg-surface-muted/80 dark:bg-surface-subtle/70 rounded mb-2 animate-shimmer" />
                <div className="h-10 w-full bg-surface-muted/80 dark:bg-surface-subtle/70 rounded animate-shimmer" style={{ animationDelay: '0.1s' }} />
              </div>
              <div>
                <div className="h-4 w-1/4 bg-surface-muted/80 dark:bg-surface-subtle/70 rounded mb-2 animate-shimmer" style={{ animationDelay: '0.2s' }} />
                <div className="h-20 w-full bg-surface-muted/80 dark:bg-surface-subtle/70 rounded animate-shimmer" style={{ animationDelay: '0.3s' }} />
              </div>
              <div className="h-10 w-32 bg-surface-muted/80 dark:bg-surface-subtle/70 rounded animate-shimmer" style={{ animationDelay: '0.4s' }} />
            </div>
          </div>
        )

      case 'result':
        return (
          <div
            className={[
              'card animate-pulse bg-surface-subtle dark:bg-surface p-6 shadow-md',
              className
            ].filter(Boolean).join(' ')}
            role="status"
            aria-label="Loading result"
          >
            <div className="flex items-center mb-4">
              <div className="h-10 w-10 bg-surface-muted/80 dark:bg-surface-subtle/70 rounded-full mr-4 animate-shimmer" />
              <div className="h-6 w-1/3 bg-surface-muted/80 dark:bg-surface-subtle/70 rounded animate-shimmer" style={{ animationDelay: '0.1s' }} />
            </div>
            <div className="space-y-2">
              <div className="h-4 w-full bg-surface-muted/80 dark:bg-surface-subtle/70 rounded animate-shimmer" style={{ animationDelay: '0.2s' }} />
              <div className="h-4 w-5/6 bg-surface-muted/80 dark:bg-surface-subtle/70 rounded animate-shimmer" style={{ animationDelay: '0.3s' }} />
              <div className="h-4 w-4/6 bg-surface-muted/80 dark:bg-surface-subtle/70 rounded animate-shimmer" style={{ animationDelay: '0.4s' }} />
            </div>
            <span className="sr-only">Loading...</span>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="space-y-4" aria-live="polite">
      {Array.from({ length: count }).map((_, index) => (
        <div key={index} className="animate-fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
          {renderSkeleton()}
        </div>
      ))}
    </div>
  )
}
