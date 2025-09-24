import { useEffect, useState } from 'react'
import api from '../services/api'
import { validateContextMetrics } from '../utils/validation'

type LayerMetrics = {
  write?: number
  select?: number
  compress?: number
  isolate?: number
}

export function ContextMetricsPanel() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState<number>(0)
  const [avg, setAvg] = useState<number>(0)
  const [layers, setLayers] = useState<LayerMetrics>({})

  useEffect(() => {
    const run = async () => {
      try {
        const data = await api.getContextMetrics()
        const parsed = validateContextMetrics(data)
        const cp = parsed.context_pipeline
        setTotal(cp.total_processed)
        setAvg(cp.average_processing_time)
        setLayers(cp.layer_metrics || {})
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load context metrics')
      } finally {
        setLoading(false)
      }
    }
    run()
  }, [])

  return (
    <div className="mt-4 p-4 bg-surface-subtle rounded-lg border border-border">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between mb-2">
        <h4 className="text-sm font-semibold text-text text-center sm:text-left">Context Metrics (W‑S‑C‑I)</h4>
      </div>
      {loading ? (
        <p className="text-sm text-text-muted">Loading...</p>
      ) : error ? (
        <p className="text-sm text-error">{error}</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3 text-sm">
          <div className="bg-surface rounded p-3 border border-border text-center sm:text-left">
            <p className="text-text-muted">Processed</p>
            <p className="text-text font-semibold">{total}</p>
          </div>
          <div className="bg-surface rounded p-3 border border-border text-center sm:text-left">
            <p className="text-text-muted">Avg Time</p>
            <p className="text-text font-semibold">{avg.toFixed(2)}s</p>
          </div>
          <div className="bg-surface rounded p-3 border border-border text-center sm:text-left">
            <p className="text-text-muted">Layer Count</p>
            <p className="text-text font-semibold">W:{layers.write||0} S:{layers.select||0}</p>
          </div>
          <div className="bg-surface rounded p-3 border border-border text-center sm:text-left">
            <p className="text-text-muted">Pipeline</p>
            <p className="text-text font-semibold">C:{layers.compress||0} I:{layers.isolate||0}</p>
          </div>
        </div>
      )}
    </div>
  )
}
