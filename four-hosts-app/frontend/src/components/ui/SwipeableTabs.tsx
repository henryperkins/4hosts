import React, { useState } from 'react'
import { motion, AnimatePresence, type PanInfo } from 'framer-motion'
import { clsx } from 'clsx'

interface Tab {
  key: string
  label: string
  badge?: number
  badgeVariant?: 'default' | 'error' | 'warning' | 'success'
}

interface SwipeableTabsProps {
  tabs: Tab[]
  activeTab: string
  onTabChange: (key: string) => void
  children: React.ReactNode
  className?: string
}

export const SwipeableTabs: React.FC<SwipeableTabsProps> = ({
  tabs,
  activeTab,
  onTabChange,
  children,
  className
}) => {
  const activeIndex = tabs.findIndex(t => t.key === activeTab)
  const [isDragging, setIsDragging] = useState(false)

  const handleDragEnd = (_: any, info: PanInfo) => {
    setIsDragging(false)
    const threshold = 50
    const velocity = info.velocity.x
    const offset = info.offset.x

    if (Math.abs(velocity) >= 500 || Math.abs(offset) >= threshold) {
      if (offset > 0 && activeIndex > 0) {
        // Swipe right - go to previous tab
        onTabChange(tabs[activeIndex - 1].key)
      } else if (offset < 0 && activeIndex < tabs.length - 1) {
        // Swipe left - go to next tab
        onTabChange(tabs[activeIndex + 1].key)
      }
    }
  }

  return (
    <div className={clsx('', className)}>
      {/* Tab buttons - scrollable on mobile */}
      <div className="flex gap-1 overflow-x-auto pb-2 mb-3 scrollbar-hide">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => onTabChange(tab.key)}
            className={clsx(
              'px-3 py-1.5 rounded-lg border whitespace-nowrap text-sm transition-colors',
              'flex items-center gap-1.5',
              activeTab === tab.key
                ? 'bg-primary/10 border-primary/30 text-primary'
                : 'bg-surface-subtle border-border text-text-muted hover:bg-surface-muted'
            )}
          >
            {tab.label}
            {tab.badge !== undefined && tab.badge > 0 && (
              <span
                className={clsx(
                  'px-1.5 py-0.5 text-xs rounded-full',
                  tab.badgeVariant === 'error'
                    ? 'bg-error/20 text-error'
                    : tab.badgeVariant === 'warning'
                    ? 'bg-warning/20 text-warning'
                    : tab.badgeVariant === 'success'
                    ? 'bg-success/20 text-success'
                    : 'bg-surface-muted text-text-muted'
                )}
              >
                {tab.badge}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab indicators for mobile */}
      <div className="flex justify-center gap-1 mb-3 sm:hidden">
        {tabs.map((_, idx) => (
          <div
            key={idx}
            className={clsx(
              'h-1.5 rounded-full transition-all duration-300',
              idx === activeIndex
                ? 'w-6 bg-primary'
                : 'w-1.5 bg-surface-muted'
            )}
          />
        ))}
      </div>

      {/* Swipeable content area */}
      <div className="relative overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -300, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            drag="x"
            dragConstraints={{ left: 0, right: 0 }}
            dragElastic={0.2}
            onDragStart={() => setIsDragging(true)}
            onDragEnd={handleDragEnd}
            className={clsx(
              'touch-pan-y',
              isDragging && 'cursor-grabbing'
            )}
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Swipe hint for mobile */}
      <div className="text-center text-xs text-text-muted mt-2 sm:hidden">
        Swipe to navigate tabs
      </div>
    </div>
  )
}

