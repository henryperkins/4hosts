import React, { useState, useRef, useEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'

interface TooltipProps {
  children: React.ReactNode
  content: React.ReactNode
  placement?: 'top' | 'bottom' | 'left' | 'right'
  delay?: number
  className?: string
  disabled?: boolean
}

export const Tooltip: React.FC<TooltipProps> = ({
  children,
  content,
  placement = 'top',
  delay = 200,
  className = '',
  disabled = false
}) => {
  const [isVisible, setIsVisible] = useState(false)
  const [position, setPosition] = useState({ top: 0, left: 0 })
  const triggerRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  const calculatePosition = useCallback(() => {
    if (!triggerRef.current || !tooltipRef.current) return

    const triggerRect = triggerRef.current.getBoundingClientRect()
    const tooltipRect = tooltipRef.current.getBoundingClientRect()
    const spacing = 8

    let top = 0
    let left = 0

    switch (placement) {
      case 'top':
        top = triggerRect.top - tooltipRect.height - spacing
        left = triggerRect.left + (triggerRect.width - tooltipRect.width) / 2
        break
      case 'bottom':
        top = triggerRect.bottom + spacing
        left = triggerRect.left + (triggerRect.width - tooltipRect.width) / 2
        break
      case 'left':
        top = triggerRect.top + (triggerRect.height - tooltipRect.height) / 2
        left = triggerRect.left - tooltipRect.width - spacing
        break
      case 'right':
        top = triggerRect.top + (triggerRect.height - tooltipRect.height) / 2
        left = triggerRect.right + spacing
        break
    }

    // Ensure tooltip stays within viewport
    const padding = 8
    const viewportWidth = window.innerWidth
    const viewportHeight = window.innerHeight

    if (left < padding) left = padding
    if (left + tooltipRect.width > viewportWidth - padding) {
      left = viewportWidth - tooltipRect.width - padding
    }
    if (top < padding) top = padding
    if (top + tooltipRect.height > viewportHeight - padding) {
      top = viewportHeight - tooltipRect.height - padding
    }

    setPosition({ top, left })
  }, [placement])

  const showTooltip = () => {
    if (disabled) return
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true)
    }, delay)
  }

  const hideTooltip = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }
    setIsVisible(false)
  }

  useEffect(() => {
    if (isVisible) {
      calculatePosition()
      window.addEventListener('scroll', calculatePosition)
      window.addEventListener('resize', calculatePosition)
      
      return () => {
        window.removeEventListener('scroll', calculatePosition)
        window.removeEventListener('resize', calculatePosition)
      }
    }
  }, [isVisible, calculatePosition])

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  const tooltipContent = isVisible && !disabled && createPortal(
    <div
      ref={tooltipRef}
      role="tooltip"
      className={`
        fixed z-50 px-3 py-2 text-sm text-white bg-gray-900 dark:bg-gray-700 
        rounded-md shadow-lg pointer-events-none
        animate-fade-in transition-opacity duration-200
        ${className}
      `}
      style={{
        top: `${position.top}px`,
        left: `${position.left}px`,
      }}
    >
      {content}
      <div
        className={`
          absolute w-0 h-0 border-transparent
          ${placement === 'top' ? 'bottom-[-8px] left-1/2 -translate-x-1/2 border-t-8 border-l-8 border-r-8 border-t-gray-900 dark:border-t-gray-700' : ''}
          ${placement === 'bottom' ? 'top-[-8px] left-1/2 -translate-x-1/2 border-b-8 border-l-8 border-r-8 border-b-gray-900 dark:border-b-gray-700' : ''}
          ${placement === 'left' ? 'right-[-8px] top-1/2 -translate-y-1/2 border-l-8 border-t-8 border-b-8 border-l-gray-900 dark:border-l-gray-700' : ''}
          ${placement === 'right' ? 'left-[-8px] top-1/2 -translate-y-1/2 border-r-8 border-t-8 border-b-8 border-r-gray-900 dark:border-r-gray-700' : ''}
        `}
      />
    </div>,
    document.body
  )

  return (
    <>
      <div
        ref={triggerRef}
        onMouseEnter={showTooltip}
        onMouseLeave={hideTooltip}
        onFocus={showTooltip}
        onBlur={hideTooltip}
        className="inline-block"
      >
        {children}
      </div>
      {tooltipContent}
    </>
  )
}