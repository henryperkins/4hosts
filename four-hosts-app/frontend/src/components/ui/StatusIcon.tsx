import React from 'react'
import { 
  CheckCircle, 
  XCircle, 
  Clock, 
  Loader, 
  AlertCircle,
  CircleDashed,
  CircleX
} from 'lucide-react'

export type StatusType = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled' | 'warning' | 'info'

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
      icon: Clock,
      color: 'text-gray-500',
      label: label || 'Pending',
      animation: ''
    },
    processing: {
      icon: Loader,
      color: 'text-blue-500',
      label: label || 'Processing',
      animation: animated ? 'animate-spin' : ''
    },
    completed: {
      icon: CheckCircle,
      color: 'text-green-500',
      label: label || 'Completed',
      animation: animated ? 'animate-scale-in' : ''
    },
    failed: {
      icon: XCircle,
      color: 'text-red-500',
      label: label || 'Failed',
      animation: animated ? 'animate-shake' : ''
    },
    cancelled: {
      icon: CircleX,
      color: 'text-orange-500',
      label: label || 'Cancelled',
      animation: animated ? 'animate-scale-in' : ''
    },
    warning: {
      icon: AlertCircle,
      color: 'text-yellow-500',
      label: label || 'Warning',
      animation: animated ? 'animate-pulse' : ''
    },
    info: {
      icon: CircleDashed,
      color: 'text-blue-400',
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
      solid: 'bg-gray-500 text-white',
      subtle: 'bg-gray-100 text-gray-700 border-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:border-gray-700'
    },
    processing: {
      solid: 'bg-blue-500 text-white',
      subtle: 'bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-800'
    },
    completed: {
      solid: 'bg-green-500 text-white',
      subtle: 'bg-green-100 text-green-700 border-green-200 dark:bg-green-900/30 dark:text-green-300 dark:border-green-800'
    },
    failed: {
      solid: 'bg-red-500 text-white',
      subtle: 'bg-red-100 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-800'
    },
    cancelled: {
      solid: 'bg-orange-500 text-white',
      subtle: 'bg-orange-100 text-orange-700 border-orange-200 dark:bg-orange-900/30 dark:text-orange-300 dark:border-orange-800'
    },
    warning: {
      solid: 'bg-yellow-500 text-white',
      subtle: 'bg-yellow-100 text-yellow-700 border-yellow-200 dark:bg-yellow-900/30 dark:text-yellow-300 dark:border-yellow-800'
    },
    info: {
      solid: 'bg-blue-400 text-white',
      subtle: 'bg-blue-50 text-blue-600 border-blue-100 dark:bg-blue-900/20 dark:text-blue-400 dark:border-blue-900'
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
    pending: 'bg-gray-500',
    processing: 'bg-blue-500',
    completed: 'bg-green-500',
    failed: 'bg-red-500',
    cancelled: 'bg-orange-500',
    warning: 'bg-yellow-500',
    info: 'bg-blue-400'
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