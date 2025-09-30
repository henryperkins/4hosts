import React from 'react'
import { useResearchDisplay } from './useResearchDisplay'

export const AgentTrace: React.FC = () => {
  const {
    data: { metadata },
  } = useResearchDisplay()

  const trace = Array.isArray(metadata?.agent_trace) ? metadata.agent_trace : []
  if (trace.length === 0) return null

  return (
    <section className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-text mb-3">Agent trace</h3>
      <ol className="space-y-2 text-sm text-text-muted">
        {trace.map((entry, idx) => {
          const record = (entry && typeof entry === 'object') ? (entry as Record<string, unknown>) : {}
          const stepRaw = record.step
          const step = typeof stepRaw === 'string' ? stepRaw : `Step ${idx + 1}`
          const coverageRaw = record.coverage
          const coverage = typeof coverageRaw === 'number' ? ` • coverage ${coverageRaw}` : ''
          const proposed = Array.isArray(record.proposed_queries)
            ? (record.proposed_queries as string[])
            : undefined
          const warnings = Array.isArray(record.warnings)
            ? (record.warnings as string[])
            : []
          return (
            <li key={`${step}-${idx}`} className="border border-border rounded-md p-3">
              <div className="font-medium text-text">{step}{coverage}</div>
              {proposed && proposed.length ? (
                <p className="text-xs mt-1">Queries: {proposed.join(', ')}</p>
              ) : null}
              {warnings.length ? (
                <ul className="text-xs text-warning mt-1 list-disc pl-4">
                  {warnings.map((warning: string, warningIdx: number) => (
                    <li key={warningIdx}>{warning}</li>
                  ))}
                </ul>
              ) : null}
            </li>
          )
        })}
      </ol>
    </section>
  )
}
