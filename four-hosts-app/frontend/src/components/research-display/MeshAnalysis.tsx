import React from 'react'
import { useResearchDisplay } from './useResearchDisplay'

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
          <div className="grid gap-3 md:grid-cols-2">
            {integratedSynthesis.synergies && integratedSynthesis.synergies.length ? (
              <div>
                <h4 className="text-sm font-semibold text-text">Synergies</h4>
                <ul className="list-disc pl-5 text-sm text-text-muted space-y-1">
                  {integratedSynthesis.synergies.map((item, idx) => (
                    <li key={`synergy-${idx}`}>{item}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {integratedSynthesis.conflicts_identified && integratedSynthesis.conflicts_identified.length ? (
              <div>
                <h4 className="text-sm font-semibold text-text">Conflicts</h4>
                <ul className="list-disc pl-5 text-sm text-text-muted space-y-1">
                  {integratedSynthesis.conflicts_identified.map((conflict, idx) => (
                    <li key={`conflict-${idx}`}>{conflict.description}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      {meshSynthesis ? (
        <div className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-text mb-2">Mesh network analysis</h3>
          {meshSynthesis.integrated ? (
            <p className="text-sm text-text-muted mb-3 whitespace-pre-line">{meshSynthesis.integrated}</p>
          ) : null}
          <div className="grid gap-3 md:grid-cols-2">
            {meshSynthesis.synergies && meshSynthesis.synergies.length ? (
              <div>
                <h4 className="text-sm font-semibold text-text">Synergies</h4>
                <ul className="list-disc pl-5 text-sm text-text-muted space-y-1">
                  {meshSynthesis.synergies.map((item: string, idx: number) => (
                    <li key={`mesh-synergy-${idx}`}>{item}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {meshSynthesis.tensions && meshSynthesis.tensions.length ? (
              <div>
                <h4 className="text-sm font-semibold text-text">Tensions</h4>
                <ul className="list-disc pl-5 text-sm text-text-muted space-y-1">
                  {meshSynthesis.tensions.map((item: string, idx: number) => (
                    <li key={`mesh-tension-${idx}`}>{item}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  )
}
