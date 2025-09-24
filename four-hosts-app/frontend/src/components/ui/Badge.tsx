import React from 'react'
import type { IconType } from 'react-icons'

interface BadgeProps {
  children: React.ReactNode
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info' | 'paradigm'
  paradigm?: 'dolores' | 'teddy' | 'bernard' | 'maeve'
  size?: 'sm' | 'md' | 'lg'
  icon?: IconType
  onRemove?: () => void
  className?: string
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'default',
  paradigm,
  size = 'md',
  icon: Icon,
  onRemove,
  className = ''
}) => {
  const variantClasses = {
    default: 'badge-default',
    success: 'badge bg-success/20 text-success border border-success/30',
    warning: 'badge bg-warning/20 text-warning border border-warning/30',
    error: 'badge bg-error/20 text-error border border-error/30',
    info: 'badge bg-primary/20 text-primary border border-primary/30',
    paradigm: paradigm ? `badge paradigm-bg-${paradigm} text-paradigm-${paradigm} border border-paradigm-${paradigm}/20` : 'badge-default'
  }

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base'
  }

  return (
    <span
      className={[
        variantClasses[variant],
        sizeClasses[size],
        className
      ].filter(Boolean).join(' ')}
    >
      {Icon && <Icon className={`${size === 'sm' ? 'h-3 w-3' : size === 'lg' ? 'h-5 w-5' : 'h-4 w-4'}`} />}
      {children}
      {onRemove && (
        <button
          onClick={onRemove}
          className="ml-1 -mr-1 hover:bg-surface-muted rounded-full p-0.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1 focus-visible:ring-primary"
          aria-label="Remove"
        >
          <svg
            className={`${size === 'sm' ? 'h-3 w-3' : size === 'lg' ? 'h-4 w-4' : 'h-3.5 w-3.5'}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </span>
  )
}

// Badge Group component for organizing multiple badges
interface BadgeGroupProps {
  children: React.ReactNode
  className?: string
}

export const BadgeGroup: React.FC<BadgeGroupProps> = ({ children, className = '' }) => {
  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      {children}
    </div>
  )
}
