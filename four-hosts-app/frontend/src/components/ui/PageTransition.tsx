import React, { useEffect, useState, useRef } from 'react'
import type { ReactNode } from 'react'

interface PageTransitionProps {
  children: ReactNode
  mode?: 'fade' | 'slide' | 'scale'
  duration?: number
  className?: string
}

const TRANSITION_DURATION = 300

export const PageTransition: React.FC<PageTransitionProps> = ({ 
  children, 
  mode = 'fade',
  duration = TRANSITION_DURATION,
  className = ''
}) => {
  const [displayChildren, setDisplayChildren] = useState(children)
  const [transitionStage, setTransitionStage] = useState<'fadeIn' | 'fadeOut'>('fadeIn')
  const [isAnimating, setIsAnimating] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Check if children have actually changed
    const hasChildrenChanged = children !== displayChildren
    
    if (hasChildrenChanged && !isAnimating) {
      setIsAnimating(true)
      setTransitionStage('fadeOut')
    }
  }, [children, displayChildren, isAnimating])

  const handleTransitionEnd = () => {
    if (transitionStage === 'fadeOut') {
      setDisplayChildren(children)
      setTransitionStage('fadeIn')
      // Announce page change to screen readers
      if (containerRef.current) {
        containerRef.current.setAttribute('aria-busy', 'false')
      }
    } else if (transitionStage === 'fadeIn') {
      setIsAnimating(false)
    }
  }

  // Set aria-busy during transitions
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.setAttribute('aria-busy', isAnimating ? 'true' : 'false')
    }
  }, [isAnimating])

  const getTransitionClasses = () => {
    switch (mode) {
      case 'slide':
        return transitionStage === 'fadeIn'
          ? 'opacity-100 translate-y-0'
          : 'opacity-0 -translate-y-4'
      
      case 'scale':
        return transitionStage === 'fadeIn'
          ? 'opacity-100 scale-100'
          : 'opacity-0 scale-95'
      
      case 'fade':
      default:
        return transitionStage === 'fadeIn'
          ? 'opacity-100'
          : 'opacity-0'
    }
  }

  const baseClasses = 'transition-all ease-out'

  return (
    <div
      ref={containerRef}
      className={`${baseClasses} ${getTransitionClasses()} ${className}`}
      style={{ transitionDuration: `${duration}ms` }}
      onTransitionEnd={handleTransitionEnd}
      aria-live="polite"
      aria-atomic="true"
      role="region"
      aria-label="Page content"
    >
      {displayChildren}
    </div>
  )
}