import React from 'react'
import { useResearchDisplay } from './useResearchDisplay'

export const ResearchMetrics: React.FC = () => {
  const {
    data: { contextLayers, biasCheck, categoryDistribution, biasDistribution, credibilityDistribution, analysisMetrics },
  } = useResearchDisplay()

  const hasDistributions =
    Object.keys(categoryDistribution).length > 0 ||
    Object.keys(biasDistribution).length > 0 ||
    Object.keys(credibilityDistribution).length > 0

  const formatMsToSeconds = (ms?: number) => {
    if (typeof ms !== 'number' || !Number.isFinite(ms)) return null
    const seconds = ms / 1000
    if (seconds >= 10) return `${seconds.toFixed(1)}s`
    return `${seconds.toFixed(2)}s`
  }

  const formatRate = (rate?: number) => {
    if (typeof rate !== 'number' || !Number.isFinite(rate)) return null
    return `${rate.toFixed(2)} /s`
  }

  const analysisHasData = Boolean(analysisMetrics)

  return (
    <section className="space-y-4">
      <div className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-text mb-3">Research metrics</h3>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          <div>
            <h4 className="text-sm font-semibold text-text mb-2">Context engineering</h4>
            {contextLayers ? (
              <ul className="text-sm text-text-muted space-y-1">
                {contextLayers.write_focus ? <li>Write focus: {contextLayers.write_focus}</li> : null}
                {typeof contextLayers.compression_ratio === 'number' ? (
                  <li>Compression ratio: {(contextLayers.compression_ratio * 100).toFixed(1)}%</li>
                ) : null}
                {typeof contextLayers.token_budget === 'number' ? <li>Token budget: {contextLayers.token_budget}</li> : null}
                {contextLayers.isolation_strategy ? <li>Isolation strategy: {contextLayers.isolation_strategy}</li> : null}
                {typeof contextLayers.search_queries_count === 'number' ? (
                  <li>Search queries executed: {contextLayers.search_queries_count}</li>
                ) : null}
              </ul>
            ) : (
              <p className="text-sm text-text-muted">No context-layer telemetry available.</p>
            )}
          </div>
          <div>
            <h4 className="text-sm font-semibold text-text mb-2">Bias & credibility</h4>
            {biasCheck ? (
              <ul className="text-sm text-text-muted space-y-1">
                <li>Balanced sources: {biasCheck.balanced ? 'Yes' : 'Needs balance'}</li>
                <li>Domain diversity: {(biasCheck.domain_diversity * 100).toFixed(1)}%</li>
                {biasCheck.dominant_domain ? (
                  <li>Dominant domain: {biasCheck.dominant_domain} ({Math.round((biasCheck.dominant_share ?? 0) * 100)}%)</li>
                ) : null}
              </ul>
            ) : (
              <p className="text-sm text-text-muted">Bias metrics unavailable.</p>
            )}
          </div>
          <div>
            <h4 className="text-sm font-semibold text-text mb-2">Analysis cadence</h4>
            {analysisHasData && analysisMetrics ? (
              <ul className="text-sm text-text-muted space-y-1">
                {analysisMetrics.sourcesCompleted !== undefined && analysisMetrics.sourcesTotal !== undefined ? (
                  <li>
                    Sources analyzed: {Math.round(analysisMetrics.sourcesCompleted)} / {Math.round(analysisMetrics.sourcesTotal)}
                  </li>
                ) : null}
                {analysisMetrics.durationMs !== undefined ? (
                  <li>Phase duration: {formatMsToSeconds(analysisMetrics.durationMs) ?? '-'}</li>
                ) : null}
                {analysisMetrics.progressUpdates !== undefined ? (
                  <li>Progress updates: {Math.round(analysisMetrics.progressUpdates)}</li>
                ) : null}
                {analysisMetrics.updatesPerSecond !== undefined ? (
                  <li>Update rate: {formatRate(analysisMetrics.updatesPerSecond) ?? '-'}</li>
                ) : null}
                {analysisMetrics.avgUpdateGapMs !== undefined ? (
                  <li>Average gap: {formatMsToSeconds(analysisMetrics.avgUpdateGapMs) ?? '-'}</li>
                ) : null}
                {analysisMetrics.p95UpdateGapMs !== undefined ? (
                  <li>P95 gap: {formatMsToSeconds(analysisMetrics.p95UpdateGapMs) ?? '-'}</li>
                ) : null}
                {analysisMetrics.cancelled ? (
                  <li className="text-warning">Cancelled mid-phase</li>
                ) : null}
              </ul>
            ) : (
              <p className="text-sm text-text-muted">Analysis telemetry unavailable.</p>
            )}
          </div>
        </div>
        {hasDistributions ? (
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <DistributionCard title="Categories" data={categoryDistribution} />
            <DistributionCard title="Bias" data={biasDistribution} />
            <DistributionCard title="Credibility" data={credibilityDistribution} />
          </div>
        ) : null}
      </div>
    </section>
  )
}

const DistributionCard: React.FC<{ title: string; data: Record<string, number> }> = ({ title, data }) => {
  const entries = Object.entries(data || {})
  if (entries.length === 0) return null
  return (
    <div className="bg-surface-subtle border border-border rounded-lg p-4">
      <h5 className="text-sm font-semibold text-text mb-2">{title}</h5>
      <ul className="space-y-1 text-xs text-text-muted">
        {entries.map(([key, value]) => (
          <li key={key} className="flex justify-between">
            <span className="capitalize">{key}</span>
            <span>{value.toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
