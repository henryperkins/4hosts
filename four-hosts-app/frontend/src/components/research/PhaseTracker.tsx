import { memo, useMemo } from 'react'
import { FiCpu, FiZap, FiSearch, FiDatabase, FiCheckCircle } from 'react-icons/fi'

type PhaseTrackerProps = {
  currentPhase: string
  currentStatus: 'pending' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'cancelled' | 'warning' | 'info'
  ceLayerProgress: { done: number; total: number }
}

const CE_LAYERS = ['Write', 'Rewrite', 'Select', 'Optimize', 'Compress', 'Isolate'] as const

export const PhaseTracker = memo(function PhaseTracker({ currentPhase, currentStatus, ceLayerProgress }: PhaseTrackerProps) {
  const PHASE_ORDER = useMemo(() => (
    [
      { key: 'classification', label: 'Classification', icon: <FiCpu className="h-4 w-4" /> },
      { key: 'context_engineering', label: 'Context Engineering', icon: <FiZap className="h-4 w-4" /> },
      { key: 'search', label: 'Search & Retrieval', icon: <FiSearch className="h-4 w-4" /> },
      { key: 'analysis', label: 'Analysis', icon: <FiDatabase className="h-4 w-4" /> },
      { key: 'agentic_loop', label: 'Agentic Loop', icon: <FiZap className="h-4 w-4" /> },
      { key: 'synthesis', label: 'Synthesis', icon: <FiCpu className="h-4 w-4" /> },
      { key: 'complete', label: 'Complete', icon: <FiCheckCircle className="h-4 w-4" /> },
    ]
  ), [])

  const phases = useMemo(() => {
    const idx = PHASE_ORDER.findIndex(ph => ph.key === currentPhase)
    return PHASE_ORDER.map((p, i) => ({
      name: p.label,
      icon: p.icon,
      isActive: i === idx,
      isCompleted: i < idx,
    }))
  }, [PHASE_ORDER, currentPhase])

  const showCELayers = currentPhase === 'context_engineering' && (currentStatus === 'processing' || currentStatus === 'in_progress')

  return (
    <div className="overflow-x-auto scrollbar-hide-mobile">
      {showCELayers && (
        <div className="mb-3 flex items-center gap-2">
          {CE_LAYERS.map((label, idx) => {
            const completed = ceLayerProgress.done > idx
            const isActiveLayer = ceLayerProgress.done === idx
            const badgeStyle = completed
              ? 'bg-primary/10 border-primary/30 text-primary'
              : isActiveLayer
                ? 'bg-primary/10 border-primary/30 text-primary'
                : 'bg-surface-subtle border-border text-text-muted'
            return (
              <span key={label} className={`text-[11px] px-2 py-1 rounded-full border ${badgeStyle}`}>
                {label}
              </span>
            )
          })}
        </div>
      )}

      <div className="mb-4 bg-surface-subtle rounded-lg p-4 min-w-[320px]">
        <div className="flex justify-center sm:justify-between items-center gap-2 flex-wrap">
          {phases.map((phase, index) => (
            <div key={phase.name} className="flex items-center flex-1">
              <div className="flex flex-col items-center flex-1">
                <div className={`
                  p-2 rounded-full mb-1 transition-colors
                  ${phase.isCompleted ? 'bg-success/10 text-success' : 
                    phase.isActive ? 'bg-primary/10 text-primary animate-pulse' : 
                    'bg-surface-subtle text-text-muted'}
                `}>
                  {phase.isCompleted ? <FiCheckCircle className="h-4 w-4" /> : phase.icon}
                </div>
                <span className={`text-xs font-medium ${phase.isActive ? 'text-text' : 'text-text-muted'}`}>
                  {phase.name}
                </span>
              </div>
              {index < phases.length - 1 && (
                <div className={`h-0.5 flex-1 mx-2 transition-colors ${phase.isCompleted ? 'bg-success' : 'bg-surface-muted'}`} />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
})

export default PhaseTracker
