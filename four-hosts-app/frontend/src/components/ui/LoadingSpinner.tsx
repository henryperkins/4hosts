import React from 'react'
import { Loader, Loader2 } from 'lucide-react'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'primary' | 'secondary' | 'white' | 'current'
  text?: string
  fullScreen?: boolean
  overlay?: boolean
  icon?: 'default' | 'dots' | 'ring'
  className?: string
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  variant = 'primary',
  text,
  fullScreen = false,
  overlay = false,
  icon = 'default',
  className = ''
}) => {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8',
    xl: 'h-12 w-12'
  }

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
    xl: 'text-xl'
  }

  const variantClasses = {
    primary: 'text-blue-600 dark:text-blue-400',
    secondary: 'text-gray-600 dark:text-gray-400',
    white: 'text-white',
    current: 'text-current'
  }

  const renderSpinner = () => {
    switch (icon) {
      case 'dots':
        return <DotsSpinner size={size} />
      case 'ring':
        return <RingSpinner size={size} />
      default:
        return (
          <Loader2 
            className={`${sizeClasses[size]} animate-spin`}
            aria-hidden="true"
          />
        )
    }
  }

  const spinner = (
    <div 
      className={`inline-flex flex-col items-center justify-center gap-3 ${className}`}
      role="status"
      aria-label={text || 'Loading'}
    >
      <div className={variantClasses[variant]}>
        {renderSpinner()}
      </div>
      {text && (
        <span className={`${textSizeClasses[size]} ${variantClasses[variant]} font-medium`}>
          {text}
          <span className="loading-dots"></span>
        </span>
      )}
      <span className="sr-only">Loading...</span>
    </div>
  )

  if (fullScreen) {
    return (
      <div className={`
        fixed inset-0 z-50 flex items-center justify-center
        ${overlay ? 'bg-black/50 backdrop-blur-sm' : 'bg-surface'}
      `}>
        {spinner}
      </div>
    )
  }

  return spinner
}

// Dots Spinner Component
const DotsSpinner: React.FC<{ size: 'sm' | 'md' | 'lg' | 'xl' }> = ({ size }) => {
  const dotSizes = {
    sm: 'h-1 w-1',
    md: 'h-1.5 w-1.5',
    lg: 'h-2 w-2',
    xl: 'h-3 w-3'
  }

  const gapSizes = {
    sm: 'gap-1',
    md: 'gap-1.5',
    lg: 'gap-2',
    xl: 'gap-3'
  }

  return (
    <div className={`flex ${gapSizes[size]}`}>
      {[0, 1, 2].map((index) => (
        <div
          key={index}
          className={`
            ${dotSizes[size]} 
            rounded-full bg-current 
            animate-pulse
          `}
          style={{
            animationDelay: `${index * 150}ms`,
            animationDuration: '1.4s'
          }}
        />
      ))}
    </div>
  )
}

// Ring Spinner Component
const RingSpinner: React.FC<{ size: 'sm' | 'md' | 'lg' | 'xl' }> = ({ size }) => {
  const sizes = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8',
    xl: 'h-12 w-12'
  }

  const borderSizes = {
    sm: 'border-2',
    md: 'border-2',
    lg: 'border-3',
    xl: 'border-4'
  }

  return (
    <div className={`
      ${sizes[size]} 
      ${borderSizes[size]}
      border-current
      border-t-transparent
      rounded-full
      animate-spin
    `} />
  )
}

// Page Loading Component
interface PageLoadingProps {
  text?: string
}

export const PageLoading: React.FC<PageLoadingProps> = ({ text = 'Loading page...' }) => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface">
      <div className="text-center">
        <LoadingSpinner 
          size="xl" 
          variant="primary" 
          text={text}
          icon="ring"
        />
      </div>
    </div>
  )
}

// Inline Loading Component for buttons and small areas
interface InlineLoadingProps {
  size?: 'sm' | 'md'
  className?: string
}

export const InlineLoading: React.FC<InlineLoadingProps> = ({ 
  size = 'sm', 
  className = '' 
}) => {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <span className="loading-spinner" role="progressbar" aria-label="Loading" />
      <span className="sr-only">Loading...</span>
    </span>
  )
}

// Loading Overlay Component
interface LoadingOverlayProps {
  isLoading: boolean
  text?: string
  children: React.ReactNode
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ 
  isLoading, 
  text, 
  children 
}) => {
  return (
    <div className="relative">
      {children}
      {isLoading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-surface/80 backdrop-blur-sm rounded-lg">
          <LoadingSpinner 
            size="lg" 
            variant="primary" 
            text={text}
          />
        </div>
      )}
    </div>
  )
}