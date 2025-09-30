import { useEffect, useState, useCallback } from 'react'
import api from '../../services/api'
import { validateContextMetrics } from '../../utils/validation'
import { Button } from '../ui/Button'

type LayerMetrics = {
  write?: number
  select?: number
  compress?: number
  isolate?: number
}

export function ContextMetricsPanel() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [metrics, setMetrics] = useState<{ total: number; avg: number; layers: LayerMetrics }>({
    total: 0,
    avg: 0,
    layers: {}
  })

  const loadMetrics = useCallback(async (opts: { refresh?: boolean } = {}) => {
    const { refresh = false } = opts
    if (refresh) {
      setRefreshing(true)
    } else {
      setLoading(true)
    }

    try {
      const data = await api.getContextMetrics()
      const parsed = validateContextMetrics(data)
      const cp = parsed.context_pipeline
      setMetrics({
        total: cp.total_processed,
        avg: cp.average_processing_time,
        layers: cp.layer_metrics || {}
      })
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load context metrics')
    } finally {
      if (refresh) {
        setRefreshing(false)
      } else {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    loadMetrics()
  }, [loadMetrics])

  const formatAvg = metrics.avg > 0 && metrics.avg < 0.01
    ? '<0.01s'
    : `${metrics.avg.toFixed(2)}s`

  return (
    <div className="mt-4 p-4 bg-surface-subtle rounded-lg border border-border">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between mb-2">
        <h4 className="text-sm font-semibold text-text text-center sm:text-left">Context Metrics (W‑S‑C‑I)</h4>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => loadMetrics({ refresh: true })}
          disabled={loading || refreshing}
          loading={refreshing}
          className="self-end sm:self-auto text-xs"
        >
          Refresh
        </Button>
      </div>
      {loading ? (
        <p className="text-sm text-text-muted">Loading…</p>
      ) : error ? (
        <p className="text-sm text-error">{error}</p>
      ) : (
        <div className={`grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3 text-sm ${refreshing ? 'opacity-75' : ''}`} aria-live="polite">
          <div className="bg-surface rounded p-3 border border-border text-center sm:text-left">
            <p className="text-text-muted">Processed</p>
            <p className="text-text font-semibold">{metrics.total}</p>
          </div>
          <div className="bg-surface rounded p-3 border border-border text-center sm:text-left">
            <p className="text-text-muted">Avg Time</p>
            <p className="text-text font-semibold">{formatAvg}</p>
          </div>
          <div className="bg-surface rounded p-3 border border-border text-center sm:text-left">
            <p className="text-text-muted">Layer Count</p>
            <p className="text-text font-semibold">W:{metrics.layers.write || 0} S:{metrics.layers.select || 0}</p>
          </div>
          <div className="bg-surface rounded p-3 border border-border text-center sm:text-left">
            <p className="text-text-muted">Pipeline</p>
            <p className="text-text font-semibold">C:{metrics.layers.compress || 0} I:{metrics.layers.isolate || 0}</p>
          </div>
        </div>
      )}
    </div>
  )
}
