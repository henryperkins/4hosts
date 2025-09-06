import { forwardRef } from 'react'
import type { HTMLAttributes } from 'react'
import type { IconType } from 'react-icons'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'interactive' | 'paradigm'
  paradigm?: 'dolores' | 'teddy' | 'bernard' | 'maeve'
  elevated?: boolean
}

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ 
    children, 
    variant = 'default',
    paradigm,
    elevated = false,
    className = '',
    ...props 
  }, ref) => {
    const getVariantClass = () => {
      if (variant === 'paradigm' && paradigm) {
        return `paradigm-card paradigm-bg-${paradigm}`
      }
      if (variant === 'interactive' && paradigm) {
        return `card-interactive paradigm-border-${paradigm}`
      }
      return variant === 'interactive' ? 'card-interactive active:scale-[0.98] transition-transform' : 'card'
    }

    const elevationClass = elevated ? 'shadow-xl hover:shadow-2xl' : ''

    return (
      <div
        ref={ref}
        className={`${getVariantClass()} ${elevationClass} ${className}`.trim()}
        {...props}
      >
        {children}
      </div>
    )
  }
)

Card.displayName = 'Card'

interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {
  icon?: IconType
}

export const CardHeader = forwardRef<HTMLDivElement, CardHeaderProps>(
  ({ children, icon: Icon, className = '', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`mb-4 ${className}`.trim()}
        {...props}
      >
        {Icon && (
          <div className="flex items-center gap-3 mb-2">
            <Icon className="h-5 w-5 text-text-muted" />
            {children}
          </div>
        )}
        {!Icon && children}
      </div>
    )
  }
)

CardHeader.displayName = 'CardHeader'

export const CardTitle = forwardRef<HTMLHeadingElement, HTMLAttributes<HTMLHeadingElement>>(
  ({ children, className = '', ...props }, ref) => {
    return (
      <h3
        ref={ref}
        className={`text-lg font-semibold text-text ${className}`.trim()}
        {...props}
      >
        {children}
      </h3>
    )
  }
)

CardTitle.displayName = 'CardTitle'

export const CardContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ children, className = '', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`text-text-muted ${className}`.trim()}
        {...props}
      >
        {children}
      </div>
    )
  }
)

CardContent.displayName = 'CardContent'

export const CardFooter = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ children, className = '', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`mt-4 pt-4 border-t border-border-subtle ${className}`.trim()}
        {...props}
      >
        {children}
      </div>
    )
  }
)

CardFooter.displayName = 'CardFooter'

// Stat Card Component
interface StatCardProps extends HTMLAttributes<HTMLDivElement> {
  label: string
  value: string | number
  icon?: IconType
  trend?: {
    value: number
    label: string
  }
  accentColor?: string
}

export const StatCard = forwardRef<HTMLDivElement, StatCardProps>(
  ({ 
    label, 
    value, 
    icon: Icon, 
    trend,
    accentColor,
    className = '', 
    ...props 
  }, ref) => {
    return (
      <Card
        ref={ref}
        className={className}
        {...props}
      >
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium text-text-muted">{label}</p>
            <p className="text-2xl font-bold text-text mt-1">{value}</p>
            {trend && (
              <p className={`text-sm mt-2 ${
                trend.value >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {trend.value >= 0 ? '+' : ''}{trend.value}% {trend.label}
              </p>
            )}
          </div>
          {Icon && (
            <div 
              className={`p-3 rounded-lg ${
                accentColor || 'bg-surface-muted'
              }`}
            >
              <Icon className="h-6 w-6" />
            </div>
          )}
        </div>
      </Card>
    )
  }
)

StatCard.displayName = 'StatCard'