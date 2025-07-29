import React from 'react'
import type { LucideIcon } from 'lucide-react'

interface BadgeProps {
  children: React.ReactNode
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info' | 'paradigm'
  paradigm?: 'dolores' | 'teddy' | 'bernard' | 'maeve'
  size?: 'sm' | 'md' | 'lg'
  icon?: LucideIcon
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
    default: 'bg-surface-muted text-text border border-border',
    success: 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-200 dark:border-green-800',
    warning: 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/30 dark:text-yellow-200 dark:border-yellow-800',
    error: 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/30 dark:text-red-200 dark:border-red-800',
    info: 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/30 dark:text-blue-200 dark:border-blue-800',
    paradigm: paradigm ? `paradigm-bg-${paradigm} text-[--color-paradigm-${paradigm}] border border-[--color-paradigm-${paradigm}]/20` : 'bg-surface-muted text-text'
  }

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base'
  }

  return (
    <span
      className={`
        inline-flex items-center gap-1.5 
        font-medium rounded-full
        transition-all duration-200
        ${variantClasses[variant]}
        ${sizeClasses[size]}
        ${className}
      `}
    >
      {Icon && <Icon className={`${size === 'sm' ? 'h-3 w-3' : size === 'lg' ? 'h-5 w-5' : 'h-4 w-4'}`} />}
      {children}
      {onRemove && (
        <button
          onClick={onRemove}
          className="ml-1 -mr-1 hover:bg-black/10 dark:hover:bg-white/10 rounded-full p-0.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1 focus-visible:ring-blue-500"
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