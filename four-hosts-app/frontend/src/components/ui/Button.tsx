import React, { forwardRef } from 'react'
import type { ButtonHTMLAttributes } from 'react'
import type { LucideIcon } from 'lucide-react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  icon?: LucideIcon
  iconPosition?: 'left' | 'right'
  loading?: boolean
  fullWidth?: boolean
  ripple?: boolean
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
    ripple = true,
    className = '',
    disabled,
    onClick,
    ...props 
  }, ref) => {
    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      if (ripple && !disabled && !loading) {
        const button = e.currentTarget
        const rect = button.getBoundingClientRect()
        const rippleElement = document.createElement('span')
        const diameter = Math.max(rect.width, rect.height)
        const radius = diameter / 2

        rippleElement.style.width = rippleElement.style.height = `${diameter}px`
        rippleElement.style.left = `${e.clientX - rect.left - radius}px`
        rippleElement.style.top = `${e.clientY - rect.top - radius}px`
        rippleElement.classList.add('ripple')

        button.appendChild(rippleElement)

        setTimeout(() => {
          rippleElement.remove()
        }, 600)
      }

      onClick?.(e)
    }

    const baseClasses = 'group relative overflow-hidden font-medium rounded-md transition-all duration-200 transform active:scale-95 focus-visible:outline-none'
    
    const variantClasses = {
      primary: 'btn-primary',
      secondary: 'btn-secondary',
      success: 'bg-green-600 text-white hover:bg-green-700 hover:shadow-lg focus-visible:ring-2 focus-visible:ring-green-500 focus-visible:ring-offset-2',
      danger: 'bg-red-600 text-white hover:bg-red-700 hover:shadow-lg focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2',
      ghost: 'bg-transparent text-text hover:bg-surface-subtle focus-visible:ring-2 focus-visible:ring-gray-500 focus-visible:ring-offset-2'
    }

    const sizeClasses = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2 text-base',
      lg: 'px-6 py-3 text-lg'
    }

    const disabledClasses = 'opacity-50 cursor-not-allowed hover:shadow-none'
    const fullWidthClasses = fullWidth ? 'w-full' : ''

    const finalClassName = `
      ${baseClasses}
      ${variantClasses[variant]}
      ${sizeClasses[size]}
      ${(disabled || loading) ? disabledClasses : 'hover:-translate-y-0.5'}
      ${fullWidthClasses}
      ${className}
    `.trim()

    return (
      <button
        ref={ref}
        className={finalClassName}
        disabled={disabled || loading}
        onClick={handleClick}
        aria-busy={loading}
        aria-disabled={disabled || loading}
        {...props}
      >
        <span className="flex items-center justify-center space-x-2">
          {loading ? (
            <>
              <span className="loading-spinner" role="progressbar" aria-label="Loading"></span>
              <span aria-live="polite">Loading<span className="loading-dots"></span></span>
            </>
          ) : (
            <>
              {Icon && iconPosition === 'left' && (
                <Icon className={`${size === 'sm' ? 'h-4 w-4' : size === 'lg' ? 'h-6 w-6' : 'h-5 w-5'} transition-transform duration-300 group-hover:scale-110`} />
              )}
              {children}
              {Icon && iconPosition === 'right' && (
                <Icon className={`${size === 'sm' ? 'h-4 w-4' : size === 'lg' ? 'h-6 w-6' : 'h-5 w-5'} transition-transform duration-300 group-hover:scale-110`} />
              )}
            </>
          )}
        </span>
      </button>
    )
  }
)

Button.displayName = 'Button'