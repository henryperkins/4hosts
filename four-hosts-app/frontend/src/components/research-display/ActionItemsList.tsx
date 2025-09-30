import React from 'react'
import { useResearchDisplay } from './useResearchDisplay'
import { getPriorityIcon } from '../../utils/research-display'
import { getParadigmDescription, getParadigmClass } from '../../constants/paradigm'
import type { Paradigm } from '../../types'

export const ActionItemsList: React.FC = () => {
  const {
    data: { actionItems, results },
  } = useResearchDisplay()

  if (!actionItems || actionItems.length === 0) {
    return (
      <section className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-text mb-2">Action items</h3>
        <p className="text-sm text-text-muted">No explicit action items were generated for this run.</p>
      </section>
    )
  }

  return (
    <section className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between mb-4">
        <h3 className="text-lg font-semibold text-text">Action items</h3>
        <span className="text-xs text-text-muted">{actionItems.length} total</span>
      </div>
      <ul className="space-y-3">
        {actionItems.map((item, index) => {
          const priority = (item.priority || 'medium').toLowerCase()
          const icon = getPriorityIcon(priority)
          const paradigm = (item.paradigm || results.paradigm_analysis?.primary?.paradigm || 'bernard') as Paradigm
          return (
            <li key={`${item.action || 'action'}-${index}`} className="border border-border rounded-lg p-4 bg-surface-subtle">
              <div className="flex items-start gap-3">
                <div className="mt-1 text-text-muted">{icon}</div>
                <div className="flex-1">
                  <div className="flex flex-wrap gap-2 items-center">
                    <h4 className="text-base font-semibold text-text">{item.action || 'Action'}</h4>
                    <span className={`px-2 py-0.5 text-xs rounded-full border ${getParadigmClass(paradigm)}`}>
                      {getParadigmDescription(paradigm)}
                    </span>
                    <span className="text-xs uppercase tracking-wide text-text-subtle">{priority}</span>
                  </div>
                  {item.description ? (
                    <p className="mt-2 text-sm text-text-muted">{item.description}</p>
                  ) : null}
                  <div className="mt-3 text-xs text-text-subtle flex flex-wrap gap-3">
                    {item.timeframe ? <span>Timeframe: {item.timeframe}</span> : null}
                    {item.owner ? <span>Owner: {item.owner}</span> : null}
                    {item.due_date ? (
                      <span>Due: {new Date(item.due_date).toLocaleDateString()}</span>
                    ) : null}
                  </div>
                </div>
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
