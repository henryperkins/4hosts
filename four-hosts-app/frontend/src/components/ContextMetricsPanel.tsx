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
    <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-900/30 rounded-lg border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">Context Metrics (W‑S‑C‑I)</h4>
      </div>
      {loading ? (
        <p className="text-sm text-gray-600 dark:text-gray-400">Loading...</p>
      ) : error ? (
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div className="bg-white dark:bg-gray-800 rounded p-3 border border-gray-200 dark:border-gray-700">
            <p className="text-gray-600 dark:text-gray-400">Processed</p>
            <p className="text-gray-900 dark:text-gray-100 font-semibold">{total}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded p-3 border border-gray-200 dark:border-gray-700">
            <p className="text-gray-600 dark:text-gray-400">Avg Time</p>
            <p className="text-gray-900 dark:text-gray-100 font-semibold">{avg.toFixed(2)}s</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded p-3 border border-gray-200 dark:border-gray-700">
            <p className="text-gray-600 dark:text-gray-400">Layer Count</p>
            <p className="text-gray-900 dark:text-gray-100 font-semibold">W:{layers.write||0} S:{layers.select||0}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded p-3 border border-gray-200 dark:border-gray-700">
            <p className="text-gray-600 dark:text-gray-400">Pipeline</p>
            <p className="text-gray-900 dark:text-gray-100 font-semibold">C:{layers.compress||0} I:{layers.isolate||0}</p>
          </div>
        </div>
      )}
    </div>
  )
}
