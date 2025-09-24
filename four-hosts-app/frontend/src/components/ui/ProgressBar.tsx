import React from 'react'

interface ProgressBarProps {
  value?: number
  max?: number
  ratio?: number  // Optional ratio prop (0-1)
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
  size?: 'sm' | 'md' | 'lg'
  animated?: boolean
  shimmer?: boolean
  showLabel?: boolean
  label?: string
  className?: string
  'aria-label'?: string
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value = 0,
  max = 100,
  ratio,
  variant = 'default',
  size = 'md',
  animated = true,
  shimmer = false,
  showLabel = false,
  label,
  className = '',
  'aria-label': ariaLabel,
}) => {
  // Use ratio if provided (0-1), otherwise calculate from value/max
  const percentage = ratio !== undefined 
    ? Math.min(Math.max(ratio * 100, 0), 100)
    : Math.min(Math.max((value / max) * 100, 0), 100)
  
  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3'
  }

  const variantClasses = {
    default: 'bg-primary',
    success: 'bg-success',
    warning: 'bg-warning',
    danger: 'bg-error',
    info: 'bg-primary/80'
  }

  const backgroundClasses = {
    default: 'bg-surface-muted',
    success: 'bg-success/15',
    warning: 'bg-warning/15',
    danger: 'bg-error/15',
    info: 'bg-primary/15'
  }

  return (
    <div className={`w-full ${className}`}>
      {(showLabel || label) && (
        <div className="flex justify-between items-center mb-1">
          {label && <span className="text-sm text-text-muted">{label}</span>}
          {showLabel && (
            <span className="text-sm font-medium text-text">
              {percentage.toFixed(0)}%
            </span>
          )}
        </div>
      )}
      
      <div
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={ariaLabel || label || `Progress: ${percentage.toFixed(0)}%`}
        className={`
          relative overflow-hidden rounded-full
          ${sizeClasses[size]}
          ${backgroundClasses[variant]}
        `}
      >
        <div
          className={`
            h-full rounded-full
            ${variantClasses[variant]}
            ${animated ? 'transition-all duration-500 ease-out' : ''}
            ${shimmer ? 'animate-shimmer' : ''}
          `}
          style={{ width: `${percentage}%` }}
        >
          {shimmer && (
            <div className="absolute inset-0 bg-linear-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
          )}
        </div>
      </div>
    </div>
  )
}

// Circular Progress variant
interface CircularProgressProps {
  value: number
  max?: number
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
  strokeWidth?: number
  showLabel?: boolean
  className?: string
}

export const CircularProgress: React.FC<CircularProgressProps> = ({
  value,
  max = 100,
  size = 'md',
  variant = 'default',
  strokeWidth = 4,
  showLabel = true,
  className = ''
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100)
  
  const sizes = {
    sm: 40,
    md: 60,
    lg: 80,
    xl: 100
  }
  
  const variantColors = {
    default: 'stroke-primary',
    success: 'stroke-success',
    warning: 'stroke-warning',
    danger: 'stroke-error',
    info: 'stroke-primary'
  }
  
  const diameter = sizes[size]
  const radius = (diameter - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const strokeDashoffset = circumference - (percentage / 100) * circumference
  
  return (
    <div className={`relative inline-flex items-center justify-center ${className}`}>
      <svg
        width={diameter}
        height={diameter}
        className="transform -rotate-90"
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
      >
        {/* Background circle */}
        <circle
          cx={diameter / 2}
          cy={diameter / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-surface-muted"
        />
        
        {/* Progress circle */}
        <circle
          cx={diameter / 2}
          cy={diameter / 2}
          r={radius}
          fill="none"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className={`${variantColors[variant]} transition-all duration-500 ease-out`}
          style={{
            strokeDashoffset: strokeDashoffset,
          }}
        />
      </svg>
      
      {showLabel && (
        <span className="absolute text-sm font-medium text-text">
          {percentage.toFixed(0)}%
        </span>
      )}
    </div>
  )
}

// Stacked Progress for multiple values
interface StackedProgressProps {
  segments: Array<{
    value: number
    variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
    label?: string
  }>
  max?: number
  size?: 'sm' | 'md' | 'lg'
  showLabels?: boolean
  className?: string
}

export const StackedProgress: React.FC<StackedProgressProps> = ({
  segments,
  max = 100,
  size = 'md',
  showLabels = false,
  className = ''
}) => {
  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3'
  }

  const variantClasses = {
    default: 'bg-primary',
    success: 'bg-success',
    warning: 'bg-warning',
    danger: 'bg-error',
    info: 'bg-primary/80'
  }

  const total = segments.reduce((sum, segment) => sum + segment.value, 0)
  
  return (
    <div className={`w-full ${className}`}>
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={max}
        aria-valuenow={total}
        className={`
          relative overflow-hidden rounded-full bg-surface-muted
          ${sizeClasses[size]} flex
        `}
      >
        {segments.map((segment, index) => {
          const percentage = (segment.value / max) * 100
          return (
            <div
              key={index}
              className={`
                h-full transition-all duration-500 ease-out
                ${variantClasses[segment.variant || 'default']}
                ${index > 0 ? 'border-l border-surface' : ''}
              `}
              style={{ width: `${percentage}%` }}
              title={segment.label}
            />
          )
        })}
      </div>
      
      {showLabels && (
        <div className="flex justify-between mt-2 text-xs text-text-muted">
          {segments.map((segment, index) => (
            <div key={index} className="flex items-center gap-1">
              <div className={`w-3 h-3 rounded ${variantClasses[segment.variant || 'default']}`} />
              <span>{segment.label}: {segment.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
