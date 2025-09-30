import React from 'react'
import { useResearchDisplay } from './useResearchDisplay'

// Shared component for rendering synergies and conflicts/tensions
const SynthesisGrid: React.FC<{
  synergies?: string[]
  conflicts?: Array<{ description?: string } | string>
  conflictLabel?: 'Conflicts' | 'Tensions'
}> = ({ synergies, conflicts, conflictLabel = 'Conflicts' }) => (
  <div className="grid gap-3 md:grid-cols-2">
    {synergies && synergies.length > 0 ? (
      <div>
        <h4 className="text-sm font-semibold text-text">Synergies</h4>
        <ul className="list-disc pl-5 text-sm text-text-muted space-y-1">
          {synergies.map((item, idx) => (
            <li key={`synergy-${idx}`}>{item}</li>
          ))}
        </ul>
      </div>
    ) : null}
    {conflicts && conflicts.length > 0 ? (
      <div>
        <h4 className="text-sm font-semibold text-text">{conflictLabel}</h4>
        <ul className="list-disc pl-5 text-sm text-text-muted space-y-1">
          {conflicts.map((item, idx) => (
            <li key={`${conflictLabel.toLowerCase()}-${idx}`}>
              {typeof item === 'string' ? item : (item.description || 'Conflict item')}
            </li>
          ))}
        </ul>
      </div>
    ) : null}
  </div>
)

export const MeshAnalysis: React.FC = () => {
  const {
    data: { integratedSynthesis, meshSynthesis },
  } = useResearchDisplay()

  if (!integratedSynthesis && !meshSynthesis) return null

  return (
    <section className="space-y-4">
      {integratedSynthesis ? (
        <div className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-text mb-2">Integrated synthesis</h3>
          {integratedSynthesis.integrated_summary ? (
            <p className="text-sm text-text-muted mb-3 whitespace-pre-line">
              {integratedSynthesis.integrated_summary}
            </p>
          ) : null}
          <SynthesisGrid
            synergies={integratedSynthesis.synergies}
            conflicts={integratedSynthesis.conflicts_identified}
            conflictLabel="Conflicts"
          />
        </div>
      ) : null}

      {meshSynthesis ? (
        <div className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-text mb-2">Mesh network analysis</h3>
          {meshSynthesis.integrated ? (
            <p className="text-sm text-text-muted mb-3 whitespace-pre-line">{meshSynthesis.integrated}</p>
          ) : null}
          <SynthesisGrid
            synergies={meshSynthesis.synergies}
            conflicts={meshSynthesis.tensions}
            conflictLabel="Tensions"
          />
        </div>
      ) : null}
    </section>
  )
}
