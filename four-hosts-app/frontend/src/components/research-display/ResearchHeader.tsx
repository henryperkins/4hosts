import React, { useEffect, useRef, useState } from 'react'
import { FiDownload, FiLoader } from 'react-icons/fi'
import { getParadigmClass, getParadigmDescription } from '../../constants/paradigm'
import { useResearchDisplay } from './useResearchDisplay'

const STATUS_STYLES: Record<string, string> = {
  completed: 'bg-success/10 text-success border-success/30',
  partial: 'bg-warning/10 text-warning border-warning/30',
  failed: 'bg-error/10 text-error border-error/30',
}

export const ResearchHeader: React.FC = () => {
  const {
    data: { results, fetchedAt, degraded, status },
    exportManager,
  } = useResearchDisplay()
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const handleClickOutside = (evt: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(evt.target as Node)) {
        setDropdownOpen(false)
      }
    }
    if (dropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [dropdownOpen])

  const statusKey = (status || 'completed').toLowerCase()
  const statusClass = STATUS_STYLES[statusKey] ?? STATUS_STYLES.completed
  const secondaryParadigm = results.paradigm_analysis?.secondary?.paradigm

  return (
    <div className="bg-surface rounded-lg shadow-lg p-4 sm:p-6 border border-border">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`px-2.5 py-1 text-xs font-medium rounded-full border ${statusClass}`}>
              {statusKey === 'partial' ? 'Partial' : statusKey === 'failed' ? 'Failed' : 'Completed'}
            </span>
            {degraded ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-full border border-warning/40 bg-warning/10 text-warning">
                Degraded
              </span>
            ) : null}
          </div>
          <h2 className="mt-2 text-2xl font-bold text-text">Research results</h2>
          <p className="text-sm text-text-muted mt-1">Query: “{results.query}”</p>
          <p className="text-xs text-text-subtle">Fetched {new Date(fetchedAt).toLocaleString()}</p>
        </div>

        <div className="flex flex-wrap gap-2 sm:justify-end sm:items-center">
          {(() => {
            const primary = results.paradigm_analysis?.primary?.paradigm || 'bernard'
            return (
              <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getParadigmClass(primary)}`}>
                {getParadigmDescription(primary)}
              </span>
            )
          })()}
          {secondaryParadigm ? (
            <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getParadigmClass(secondaryParadigm)}`}>
              + {getParadigmDescription(secondaryParadigm)}
            </span>
          ) : null}

          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setDropdownOpen(open => !open)}
              disabled={exportManager.isExporting}
              className="p-2 text-text-muted hover:text-text hover:bg-surface-subtle rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary"
              aria-label="Export results"
              aria-expanded={dropdownOpen}
              aria-haspopup="true"
            >
              {exportManager.isExporting ? (
                <FiLoader className="h-5 w-5 animate-spin" />
              ) : (
                <FiDownload className="h-5 w-5" />
              )}
            </button>
            <div
              className={`absolute right-0 mt-2 w-48 bg-surface rounded-lg shadow-xl border border-border transition-all duration-200 z-10 ${dropdownOpen ? 'opacity-100 visible translate-y-0' : 'opacity-0 invisible -translate-y-2'}`}
              role="menu"
            >
              <ul className="py-2 text-sm">
                {exportManager.availableFormats.map(format => (
                  <li key={format}>
                    <button
                      type="button"
                      role="menuitem"
                      className="w-full text-left px-3 py-2 hover:bg-surface-subtle transition"
                      onClick={() => {
                        setDropdownOpen(false)
                        exportManager.exportResult(format)
                      }}
                    >
                      {format.toUpperCase()}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
