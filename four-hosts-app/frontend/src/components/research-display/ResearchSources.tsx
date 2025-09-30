import React from 'react'
import { FiExternalLink, FiFilter, FiChevronLeft, FiChevronRight } from 'react-icons/fi'
import { useResearchDisplay } from './useResearchDisplay'
import { getCredibilityBand, getCredibilityColor } from '../../utils/credibility'
import { getCredibilityIcon } from '../../utils/research-display'

export const ResearchSources: React.FC = () => {
  const {
    data: { sources },
    filters,
  } = useResearchDisplay()

  const { paginatedSources, selectedCategories, selectedCredBands, categories, toggleCategory, toggleCredBand, page, setPage, totalPages, filteredSources, pageSize, setPageSize } = filters

  if (!sources || sources.length === 0) {
    return null
  }

  return (
    <section className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between mb-4">
        <h3 className="text-lg font-semibold text-text flex items-center gap-2">
          <FiFilter className="h-4 w-4" /> Sources ({filteredSources.length}/{sources.length})
        </h3>
        <div className="flex flex-wrap gap-2 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-text-subtle">Credibility:</span>
            {(['high', 'medium', 'low'] as const).map(band => (
              <button
                key={band}
                type="button"
                onClick={() => toggleCredBand(band)}
                className={`px-2 py-1 rounded-full border transition ${selectedCredBands.has(band) ? 'bg-primary/10 border-primary text-primary' : 'bg-surface-subtle border-border text-text-muted'}`}
              >
                {band}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            <span className="text-text-subtle">Category:</span>
            <button
              type="button"
              onClick={() => toggleCategory('all')}
              className={`px-2 py-1 rounded-full border transition ${selectedCategories.has('all') ? 'bg-primary/10 border-primary text-primary' : 'bg-surface-subtle border-border text-text-muted'}`}
            >
              All
            </button>
            {categories.map(category => (
              <button
                key={category}
                type="button"
                onClick={() => toggleCategory(category)}
                className={`px-2 py-1 rounded-full border transition ${selectedCategories.has(category) ? 'bg-primary/10 border-primary text-primary' : 'bg-surface-subtle border-border text-text-muted'}`}
              >
                {category}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-1">
            <span className="text-text-subtle">Per page</span>
            <select
              value={pageSize}
              onChange={event => setPageSize(Number(event.target.value))}
              className="border border-border rounded px-2 py-1 bg-surface-subtle text-text"
            >
              {[10, 20, 30, 50].map(size => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <ul className="space-y-3">
        {paginatedSources.map((source, idx) => {
          const credibilityScore = typeof source.credibility_score === 'number' ? source.credibility_score : undefined
          const band = credibilityScore !== undefined ? getCredibilityBand(credibilityScore) : null
          const icon = credibilityScore !== undefined ? getCredibilityIcon(credibilityScore) : null
          const colorClass = credibilityScore !== undefined ? getCredibilityColor(credibilityScore) : 'text-text-muted'
          return (
            <li key={`${source.url || source.title || 'source'}-${idx}`} className="border border-border rounded-lg p-4 bg-surface-subtle">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                <div className="flex-1">
                  <h4 className="font-semibold text-text text-sm sm:text-base">
                    {source.title || source.url || 'Untitled source'}
                  </h4>
                  <p className="text-sm text-text-muted mt-1">{source.snippet || 'No snippet available.'}</p>
                  <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-text-subtle">
                    {source.domain ? <span>{source.domain}</span> : null}
                    {source.source_category ? <span className="uppercase tracking-wide">{source.source_category}</span> : null}
                    {source.published_date ? <span>{new Date(source.published_date).toLocaleDateString()}</span> : null}
                    {band ? (
                      <span className={`flex items-center gap-1 font-medium ${colorClass}`}>
                        {icon} {band} credibility
                      </span>
                    ) : null}
                  </div>
                </div>
                {source.url ? (
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center text-sm text-primary hover:opacity-80 whitespace-nowrap"
                  >
                    Open source <FiExternalLink className="h-4 w-4 ml-1" />
                  </a>
                ) : null}
              </div>
            </li>
          )
        })}
      </ul>

      {totalPages > 1 ? (
        <div className="mt-4 flex items-center justify-between text-sm">
          <button
            type="button"
            onClick={() => setPage(Math.max(1, page - 1))}
            disabled={page === 1}
            className="inline-flex items-center gap-1 px-3 py-1.5 border border-border rounded disabled:opacity-50"
          >
            <FiChevronLeft /> Prev
          </button>
          <span className="text-text-muted">
            Page {page} of {totalPages}
          </span>
          <button
            type="button"
            onClick={() => setPage(Math.min(totalPages, page + 1))}
            disabled={page === totalPages}
            className="inline-flex items-center gap-1 px-3 py-1.5 border border-border rounded disabled:opacity-50"
          >
            Next <FiChevronRight />
          </button>
        </div>
      ) : null}
    </section>
  )
}
