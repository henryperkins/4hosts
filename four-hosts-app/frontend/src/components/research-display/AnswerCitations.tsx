import React, { useState } from 'react'
import { FiExternalLink } from 'react-icons/fi'
import { useResearchDisplay } from './useResearchDisplay'
import { getCredibilityIcon } from '../../utils/research-display'

export const AnswerCitations: React.FC = () => {
  const {
    data: { citations },
  } = useResearchDisplay()
  const [showAll, setShowAll] = useState(false)

  if (!citations || citations.length === 0) return null

  const visible = showAll ? citations : citations.slice(0, 5)

  return (
    <section className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between mb-3">
        <h3 className="text-lg font-semibold text-text">Citations</h3>
        {citations.length > 5 ? (
          <button
            type="button"
            onClick={() => setShowAll(prev => !prev)}
            className="text-sm text-primary hover:opacity-80"
          >
            {showAll ? 'Show fewer' : `Show all ${citations.length}`}
          </button>
        ) : null}
      </div>
      <ul className="space-y-3">
        {visible.map((citation, index) => (
          <li key={`${citation.id || citation.url || 'citation'}-${index}`} className="border border-border rounded-lg p-3 bg-surface-subtle">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <p className="text-sm font-medium text-text">{citation.title || citation.source || citation.url}</p>
                <div className="mt-1 flex flex-wrap gap-2 items-center text-xs text-text-muted">
                  {citation.source ? <span>{citation.source}</span> : null}
                  {typeof citation.credibility_score === 'number' ? (
                    <span className="flex items-center gap-1 text-text">
                      {getCredibilityIcon(citation.credibility_score)} {(citation.credibility_score * 100).toFixed(0)}%
                    </span>
                  ) : null}
                  {citation.paradigm_alignment ? <span>{citation.paradigm_alignment}</span> : null}
                </div>
              </div>
              {citation.url ? (
                <a
                  href={citation.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center text-sm text-primary hover:opacity-80"
                >
                  View source <FiExternalLink className="h-4 w-4 ml-1" />
                </a>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}
