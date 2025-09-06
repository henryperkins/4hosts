import { forwardRef } from 'react'
import type { ButtonHTMLAttributes } from 'react'
import type { IconType } from 'react-icons'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  icon?: IconType
  iconPosition?: 'left' | 'right'
  loading?: boolean
  fullWidth?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ 
    children, 
    variant = 'primary', 
    size = 'md', 
    icon: Icon,
    iconPosition = 'left',
    loading = false,
    fullWidth = false,
    className = '',
    disabled,
    ...props 
  }, ref) => {
    const variantClasses = {
      primary: 'btn-primary',
      secondary: 'btn-secondary',
      success: 'btn bg-success text-white hover:bg-success/90',
      danger: 'btn bg-error text-white hover:bg-error/90',
      ghost: 'btn-ghost'
    }

    const sizeClasses = {
      sm: 'px-3 py-1.5 text-sm min-h-[36px] md:min-h-[32px]',
      md: 'px-4 py-2 text-base min-h-[44px] md:min-h-[40px]',
      lg: 'px-6 py-3 text-lg min-h-[52px] md:min-h-[48px]'
    }

    const fullWidthClasses = fullWidth ? 'w-full' : ''

    const finalClassName = [
      variantClasses[variant],
      sizeClasses[size],
      fullWidthClasses,
      'touch-target select-none-mobile active:scale-95 transition-transform',
      className
    ].filter(Boolean).join(' ')

    const buttonLabel = loading ? 'Loading' : children

    return (
      <button
        ref={ref}
        className={finalClassName}
        disabled={disabled || loading}
        aria-busy={loading}
        aria-disabled={disabled || loading}
        {...props}
      >
        <span className="relative z-10 flex items-center justify-center space-x-2">
          {loading ? (
            <>
              <span className="spinner" role="status" aria-label="Loading">
                <span className="sr-only">Loading</span>
              </span>
              <span aria-live="polite" aria-atomic="true">
                {typeof buttonLabel === 'string' ? buttonLabel : 'Loading'}
              </span>
            </>
          ) : (
            <>
              {Icon && iconPosition === 'left' && (
                <Icon 
                  className={size === 'sm' ? 'h-4 w-4' : size === 'lg' ? 'h-6 w-6' : 'h-5 w-5'} 
                  aria-hidden="true"
                />
              )}
              <span>{children}</span>
              {Icon && iconPosition === 'right' && (
                <Icon 
                  className={size === 'sm' ? 'h-4 w-4' : size === 'lg' ? 'h-6 w-6' : 'h-5 w-5'}
                  aria-hidden="true"
                />
              )}
            </>
          )}
        </span>
      </button>
    )
  }
)

Button.displayName = 'Button'