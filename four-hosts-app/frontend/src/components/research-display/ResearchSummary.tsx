import React from 'react'
import { FiAlertTriangle } from 'react-icons/fi'
import { useResearchDisplay } from './useResearchDisplay'

export const ResearchSummary: React.FC = () => {
  const {
    data: { summary, confidenceInfo, actionableRatio, metadata, evidenceSnapshot, warnings, degraded },
  } = useResearchDisplay()

  const actionablePercent = Number.isFinite(actionableRatio) ? Math.round(actionableRatio * 100) : null
  const averageCredibility = metadata?.credibility_summary?.average_score

  return (
    <div className="space-y-4">
      <section className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-text mb-2">Executive summary</h3>
        <p className="text-text leading-relaxed whitespace-pre-line">{summary || 'No summary available.'}</p>
      </section>

      <section className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <div className="bg-surface-subtle rounded p-4 border border-border">
            <p className="text-xs uppercase text-text-subtle tracking-wide">Confidence</p>
            <p className="text-2xl font-semibold text-text mt-1">{confidenceInfo.percentage.toFixed(0)}%</p>
            <p className="text-sm text-text-muted">{confidenceInfo.band}{confidenceInfo.because ? ` • ${confidenceInfo.because}` : ''}</p>
          </div>
          <div className="bg-surface-subtle rounded p-4 border border-border">
            <p className="text-xs uppercase text-text-subtle tracking-wide">Actionable content</p>
            <p className="text-2xl font-semibold text-text mt-1">{actionablePercent !== null ? `${actionablePercent}%` : '—'}</p>
            <p className="text-sm text-text-muted">Share of the synthesis tagged as actionable</p>
          </div>
          <div className="bg-surface-subtle rounded p-4 border border-border">
            <p className="text-xs uppercase text-text-subtle tracking-wide">Evidence quality</p>
            <p className="text-2xl font-semibold text-text mt-1">{averageCredibility ? averageCredibility.toFixed(2) : '—'}</p>
            <p className="text-sm text-text-muted">Average credibility score across sources</p>
          </div>
          <div className="bg-surface-subtle rounded p-4 border border-border">
            <p className="text-xs uppercase text-text-subtle tracking-wide">Evidence window</p>
            <p className="text-2xl font-semibold text-text mt-1">{evidenceSnapshot.window || 'n/a'}</p>
            <p className="text-sm text-text-muted">{`${evidenceSnapshot.total} sources (${evidenceSnapshot.strong} strong)`}</p>
          </div>
        </div>
        {degraded || (warnings && warnings.length > 0) ? (
          <div className="mt-4 flex gap-2 items-start bg-warning/10 border border-warning/40 rounded-md p-3 text-sm text-warning">
            <FiAlertTriangle className="mt-0.5 h-4 w-4" />
            <div>
              <p className="font-medium">Quality note</p>
              <ul className="list-disc pl-5 space-y-1">
                {degraded ? <li>Synthesis delivered with degraded data quality.</li> : null}
                {(warnings || []).map((warning, idx) => {
                  if (typeof warning === 'string') {
                    return <li key={idx}>{warning}</li>
                  }
                  if (warning && typeof warning === 'object') {
                    const code = 'code' in warning ? String(warning.code) : null
                    const message = 'message' in warning ? String(warning.message) : null
                    return (
                      <li key={idx}>
                        {code ? `${code}: ` : ''}
                        {message ?? JSON.stringify(warning)}
                      </li>
                    )
                  }
                  return <li key={idx}>{String(warning)}</li>
                })}
              </ul>
            </div>
          </div>
        ) : null}
      </section>
    </div>
  )
}
