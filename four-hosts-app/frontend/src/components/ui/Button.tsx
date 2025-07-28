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

      if (onClick) {
        onClick(e)
      }
    }

    const baseClasses = 'relative overflow-hidden font-medium rounded-md transition-all duration-200 transform active:scale-95'
    
    const variantClasses = {
      primary: 'bg-blue-600 text-white hover:bg-blue-700 hover:shadow-lg focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
      secondary: 'bg-gray-200 text-gray-800 hover:bg-gray-300 hover:shadow-md focus:ring-2 focus:ring-gray-500 focus:ring-offset-2',
      success: 'bg-green-600 text-white hover:bg-green-700 hover:shadow-lg focus:ring-2 focus:ring-green-500 focus:ring-offset-2',
      danger: 'bg-red-600 text-white hover:bg-red-700 hover:shadow-lg focus:ring-2 focus:ring-red-500 focus:ring-offset-2',
      ghost: 'bg-transparent text-gray-700 hover:bg-gray-100 focus:ring-2 focus:ring-gray-500 focus:ring-offset-2'
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
        {...props}
      >
        <span className="flex items-center justify-center space-x-2">
          {loading ? (
            <>
              <span className="loading-spinner"></span>
              <span>Loading<span className="loading-dots"></span></span>
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

// Add ripple effect styles to your CSS
const rippleStyles = `
  .ripple {
    position: absolute;
    border-radius: 50%;
    transform: scale(0);
    animation: ripple-animation 0.6s ease-out;
    background-color: rgba(255, 255, 255, 0.7);
  }

  @keyframes ripple-animation {
    to {
      transform: scale(4);
      opacity: 0;
    }
  }
`

// Inject styles (in a real app, add this to your CSS file)
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement('style')
  styleSheet.textContent = rippleStyles
  document.head.appendChild(styleSheet)
}