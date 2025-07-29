import React, { useEffect, useRef, useState } from 'react'

interface AnimatedListProps<T> {
  items: T[]
  renderItem: (item: T, index: number) => React.ReactNode
  keyExtractor?: (item: T, index: number) => string | number
  staggerDelay?: number
  animation?: 'fade-in' | 'slide-up' | 'slide-down' | 'scale-in' | 'slide-in-right' | 'slide-in-left'
  className?: string
  itemClassName?: string
  emptyState?: React.ReactNode
  loading?: boolean
  loadingComponent?: React.ReactNode
}

export function AnimatedList<T>({
  items,
  renderItem,
  keyExtractor,
  staggerDelay = 50,
  animation = 'fade-in',
  className = '',
  itemClassName = '',
  emptyState,
  loading = false,
  loadingComponent
}: AnimatedListProps<T>) {
  const [visibleItems, setVisibleItems] = useState<number[]>([])
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Reset visible items when items change
    setVisibleItems([])
    
    // Stagger the appearance of items
    const timeouts: NodeJS.Timeout[] = []
    
    items.forEach((_, index) => {
      const timeout = setTimeout(() => {
        setVisibleItems(prev => [...prev, index])
      }, index * staggerDelay)
      
      timeouts.push(timeout)
    })

    return () => {
      timeouts.forEach(timeout => clearTimeout(timeout))
    }
  }, [items, staggerDelay])

  const animationClasses = {
    'fade-in': 'animate-fade-in',
    'slide-up': 'animate-slide-up',
    'slide-down': 'animate-slide-down',
    'scale-in': 'animate-scale-in',
    'slide-in-right': 'animate-slide-in-right',
    'slide-in-left': 'animate-slide-in-left'
  }

  if (loading && loadingComponent) {
    return <>{loadingComponent}</>
  }

  if (items.length === 0 && emptyState) {
    return <>{emptyState}</>
  }

  return (
    <div ref={containerRef} className={`${className}`}>
      {items.map((item, index) => {
        const key = keyExtractor ? keyExtractor(item, index) : index
        const isVisible = visibleItems.includes(index)
        
        return (
          <div
            key={key}
            className={`
              ${itemClassName}
              ${isVisible ? animationClasses[animation] : 'opacity-0'}
              transition-opacity duration-300
            `}
          >
            {renderItem(item, index)}
          </div>
        )
      })}
    </div>
  )
}

// Virtualized version for large lists
interface VirtualizedAnimatedListProps<T> extends AnimatedListProps<T> {
  itemHeight: number
  containerHeight: number
  overscan?: number
}

export function VirtualizedAnimatedList<T>({
  items,
  renderItem,
  keyExtractor,
  itemHeight,
  containerHeight,
  overscan = 3,
  animation = 'fade-in',
  className = '',
  itemClassName = '',
  emptyState,
  loading = false,
  loadingComponent
}: VirtualizedAnimatedListProps<T>) {
  const [scrollTop, setScrollTop] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)

  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan)
  const endIndex = Math.min(
    items.length - 1,
    Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
  )

  const visibleItems = items.slice(startIndex, endIndex + 1)
  const totalHeight = items.length * itemHeight
  const offsetY = startIndex * itemHeight

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop)
  }

  if (loading && loadingComponent) {
    return <>{loadingComponent}</>
  }

  if (items.length === 0 && emptyState) {
    return <>{emptyState}</>
  }

  const animationClasses = {
    'fade-in': 'animate-fade-in',
    'slide-up': 'animate-slide-up',
    'slide-down': 'animate-slide-down',
    'scale-in': 'animate-scale-in',
    'slide-in-right': 'animate-slide-in-right',
    'slide-in-left': 'animate-slide-in-left'
  }

  return (
    <div
      ref={containerRef}
      className={`overflow-auto ${className}`}
      style={{ height: containerHeight }}
      onScroll={handleScroll}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        <div style={{ transform: `translateY(${offsetY}px)` }}>
          {visibleItems.map((item, index) => {
            const actualIndex = startIndex + index
            const key = keyExtractor ? keyExtractor(item, actualIndex) : actualIndex
            
            return (
              <div
                key={key}
                className={`${itemClassName} ${animationClasses[animation]}`}
                style={{ height: itemHeight }}
              >
                {renderItem(item, actualIndex)}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// List with drag and drop support
interface DraggableAnimatedListProps<T> extends AnimatedListProps<T> {
  onReorder: (items: T[]) => void
}

export function DraggableAnimatedList<T>({
  items,
  renderItem,
  keyExtractor,
  onReorder,
  animation = 'fade-in',
  className = '',
  itemClassName = '',
  emptyState
}: DraggableAnimatedListProps<T>) {
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null)
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null)

  const handleDragStart = (e: React.DragEvent, index: number) => {
    setDraggedIndex(index)
    e.dataTransfer.effectAllowed = 'move'
  }

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault()
    if (dragOverIndex !== index) {
      setDragOverIndex(index)
    }
  }

  const handleDrop = (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault()
    
    if (draggedIndex === null || draggedIndex === dropIndex) {
      return
    }

    const draggedItem = items[draggedIndex]
    const newItems = [...items]
    
    // Remove dragged item
    newItems.splice(draggedIndex, 1)
    
    // Insert at new position
    const adjustedDropIndex = draggedIndex < dropIndex ? dropIndex - 1 : dropIndex
    newItems.splice(adjustedDropIndex, 0, draggedItem)
    
    onReorder(newItems)
    setDraggedIndex(null)
    setDragOverIndex(null)
  }

  const handleDragEnd = () => {
    setDraggedIndex(null)
    setDragOverIndex(null)
  }

  if (items.length === 0 && emptyState) {
    return <>{emptyState}</>
  }

  const animationClasses = {
    'fade-in': 'animate-fade-in',
    'slide-up': 'animate-slide-up',
    'slide-down': 'animate-slide-down',
    'scale-in': 'animate-scale-in',
    'slide-in-right': 'animate-slide-in-right',
    'slide-in-left': 'animate-slide-in-left'
  }

  return (
    <div className={className}>
      {items.map((item, index) => {
        const key = keyExtractor ? keyExtractor(item, index) : index
        const isDragging = draggedIndex === index
        const isDragOver = dragOverIndex === index
        
        return (
          <div
            key={key}
            draggable
            onDragStart={(e) => handleDragStart(e, index)}
            onDragOver={(e) => handleDragOver(e, index)}
            onDrop={(e) => handleDrop(e, index)}
            onDragEnd={handleDragEnd}
            className={`
              ${itemClassName}
              ${animationClasses[animation]}
              ${isDragging ? 'opacity-50' : ''}
              ${isDragOver ? 'border-t-2 border-blue-500' : ''}
              cursor-move transition-all duration-200
            `}
          >
            {renderItem(item, index)}
          </div>
        )
      })}
    </div>
  )
}