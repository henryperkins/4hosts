import React from 'react'
import { 
  FiCheckCircle, 
  FiXCircle, 
  FiClock, 
  FiLoader, 
  FiAlertCircle,
  FiCircle,
  FiX
} from 'react-icons/fi'

export type StatusType = 'pending' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'cancelled' | 'warning' | 'info'

interface StatusIconProps {
  status: StatusType
  size?: 'sm' | 'md' | 'lg'
  animated?: boolean
  showLabel?: boolean
  label?: string
  className?: string
}

export const StatusIcon: React.FC<StatusIconProps> = ({
  status,
  size = 'md',
  animated = true,
  showLabel = false,
  label,
  className = ''
}) => {
  const sizeMap = {
    sm: 'h-4 w-4',
    md: 'h-5 w-5',
    lg: 'h-6 w-6'
  }

  const iconSize = sizeMap[size]

  const statusConfig = {
    pending: {
      icon: FiClock,
      color: 'text-text-muted',
      label: label || 'Pending',
      animation: ''
    },
    processing: {
      icon: FiLoader,
      color: 'text-primary',
      label: label || 'Processing',
      animation: animated ? 'animate-spin' : ''
    },
    in_progress: {
      icon: FiLoader,
      color: 'text-primary',
      label: label || 'In Progress',
      animation: animated ? 'animate-spin' : ''
    },
    completed: {
      icon: FiCheckCircle,
      color: 'text-success',
      label: label || 'Completed',
      animation: animated ? 'animate-scale-in' : ''
    },
    failed: {
      icon: FiXCircle,
      color: 'text-error',
      label: label || 'Failed',
      animation: animated ? 'animate-shake' : ''
    },
    cancelled: {
      icon: FiX,
      color: 'text-warning',
      label: label || 'Cancelled',
      animation: animated ? 'animate-scale-in' : ''
    },
    warning: {
      icon: FiAlertCircle,
      color: 'text-warning',
      label: label || 'Warning',
      animation: animated ? 'animate-pulse' : ''
    },
    info: {
      icon: FiCircle,
      color: 'text-primary/80',
      label: label || 'Info',
      animation: ''
    }
  }

  const config = statusConfig[status]
  const Icon = config.icon

  return (
    <div className={`inline-flex items-center gap-2 ${className}`}>
      <Icon 
        className={`
          ${iconSize} 
          ${config.color} 
          ${config.animation}
          transition-all duration-200
        `}
        aria-hidden="true"
      />
      {showLabel && (
        <span className={`text-${size === 'sm' ? 'xs' : size === 'lg' ? 'base' : 'sm'} font-medium ${config.color}`}>
          {config.label}
        </span>
      )}
      <span className="sr-only">{`Status: ${config.label}`}</span>
    </div>
  )
}

// Status Badge component that combines StatusIcon with Badge styling
interface StatusBadgeProps extends StatusIconProps {
  variant?: 'solid' | 'subtle'
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  variant = 'subtle',
  size = 'md',
  animated = true,
  label,
  className = ''
}) => {
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base'
  }

  const statusStyles = {
    pending: {
      solid: 'bg-border text-text',
      subtle: 'bg-surface-subtle text-text-muted border-border'
    },
    processing: {
      solid: 'bg-primary text-white',
      subtle: 'bg-primary/15 text-primary border border-primary/30'
    },
    in_progress: {
      solid: 'bg-primary text-white',
      subtle: 'bg-primary/15 text-primary border border-primary/30'
    },
    completed: {
      solid: 'bg-success text-white',
      subtle: 'bg-success/15 text-success border border-success/30'
    },
    failed: {
      solid: 'bg-error text-white',
      subtle: 'bg-error/15 text-error border border-error/30'
    },
    cancelled: {
      solid: 'bg-warning text-white',
      subtle: 'bg-warning/15 text-warning border border-warning/30'
    },
    warning: {
      solid: 'bg-warning text-white',
      subtle: 'bg-warning/15 text-warning border border-warning/30'
    },
    info: {
      solid: 'bg-primary text-white',
      subtle: 'bg-primary/15 text-primary border border-primary/30'
    }
  }

  const style = statusStyles[status][variant]

  return (
    <span className={`
      inline-flex items-center gap-1.5 
      font-medium rounded-full
      ${variant === 'subtle' ? 'border' : ''}
      ${sizeClasses[size]}
      ${style}
      ${className}
    `}>
      <StatusIcon 
        status={status} 
        size={size} 
        animated={animated}
        showLabel={true}
        label={label}
      />
    </span>
  )
}

// Status Dot for minimal space usage
interface StatusDotProps {
  status: StatusType
  size?: 'sm' | 'md' | 'lg'
  pulse?: boolean
  className?: string
}

export const StatusDot: React.FC<StatusDotProps> = ({
  status,
  size = 'md',
  pulse = false,
  className = ''
}) => {
  const sizeClasses = {
    sm: 'h-2 w-2',
    md: 'h-3 w-3',
    lg: 'h-4 w-4'
  }

  const statusColors = {
    pending: 'bg-border',
    processing: 'bg-primary',
    in_progress: 'bg-primary',
    completed: 'bg-success',
    failed: 'bg-error',
    cancelled: 'bg-warning',
    warning: 'bg-warning',
    info: 'bg-primary/80'
  }

  return (
    <span className={`relative inline-flex ${className}`}>
      {pulse && status === 'processing' && (
        <span className={`
          animate-ping absolute inline-flex h-full w-full rounded-full opacity-75
          ${statusColors[status]}
        `} />
      )}
      <span className={`
        relative inline-flex rounded-full
        ${sizeClasses[size]}
        ${statusColors[status]}
      `}>
        <span className="sr-only">Status: {status}</span>
      </span>
    </span>
  )
}
