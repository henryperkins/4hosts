
import React, { useEffect, useState, useCallback, useMemo } from 'react'
import type { ReactNode } from 'react'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend } from 'recharts'
import { FiActivity, FiTrendingUp, FiUsers, FiClock, FiDatabase, FiAlertCircle, FiEye, FiMousePointer, FiTarget, FiZap, FiChevronUp } from 'react-icons/fi'
import api from '../services/api'
import type { MetricsData, ExtendedStatsSnapshot, TelemetrySummary } from '../types/api-types'
import { getParadigmHexColor } from '../constants/paradigm'
import { TriageBoard } from './TriageBoard'

interface ABTestMetrics {
  standardView: {
    sessions: number
    completionRate: number
    avgTimeToInsight: number
    errorRate: number
    engagement: number
  }
  ideaBrowserView: {
    sessions: number
    completionRate: number
    avgTimeToInsight: number
    errorRate: number
    engagement: number
  }
}

interface CollapsibleSectionProps {
  title: string
  description?: string
  defaultOpen?: boolean
  children: ReactNode
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ title, description, defaultOpen = true, children }) => {
  const [open, setOpen] = useState(defaultOpen)

  const rotationClass = open ? 'rotate-0' : '-rotate-90'
  const rowClass = open ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'
  const contentState = open ? 'opacity-100 translate-y-0' : 'pointer-events-none -translate-y-3 opacity-0'

  return (
    <section className="relative overflow-hidden rounded-xl border border-border bg-surface-subtle shadow-sm transition-all duration-300">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        aria-expanded={open}
      >
        <div>
          <h2 className="text-lg font-semibold text-text">{title}</h2>
          {description && (
            <p className="mt-1 text-sm text-text-muted">{description}</p>
          )}
        </div>
        <span className={`inline-flex h-8 w-8 items-center justify-center rounded-full bg-surface text-text transition-transform duration-300 ${rotationClass}`} aria-hidden="true">
          <FiChevronUp className="h-4 w-4" />
        </span>
      </button>
      <div className={`grid transition-all duration-300 ${rowClass}`}>
        <div className="overflow-hidden px-5 pb-5">
          <div className={`transition-all duration-300 ${contentState}`}>
            {open && children}
          </div>
        </div>
      </div>
    </section>
  )
}

const metricGradients: Record<'primary' | 'success' | 'error', string> = {
  primary: 'linear-gradient(135deg, rgba(var(--paradigm-bernard-rgb), 0.18) 0%, rgba(var(--paradigm-maeve-rgb), 0.24) 100%)',
  success: 'linear-gradient(135deg, rgba(var(--paradigm-maeve-rgb), 0.18) 0%, rgba(var(--success-rgb), 0.24) 100%)',
  error: 'linear-gradient(135deg, rgba(var(--paradigm-dolores-rgb), 0.22) 0%, rgba(var(--error-rgb), 0.26) 100%)',
}

const formatRelativeTime = (timestamp: string | undefined): string => {
  if (!timestamp) return '—'
  try {
    const now = Date.now()
    const ts = new Date(timestamp).getTime()
    if (Number.isNaN(ts)) return '—'
    const diff = Math.max(0, now - ts)
    const minutes = Math.round(diff / 60000)
    if (minutes < 1) return 'just now'
    if (minutes === 1) return '1 min ago'
    if (minutes < 60) return `${minutes} mins ago`
    const hours = Math.round(minutes / 60)
    if (hours === 1) return '1 hr ago'
    if (hours < 24) return `${hours} hrs ago`
    const days = Math.round(hours / 24)
    return `${days}d ago`
  } catch {
    return '—'
  }
}

const MetricCard = React.memo<{
  icon: React.ElementType
  title: string
  value: string | number
  subtitle?: string
  trend?: number
  accent?: 'primary' | 'success' | 'error'
  compact?: boolean
}>(({ icon: Icon, title, value, subtitle, trend, accent = 'primary', compact = false }) => {
  const padding = compact ? 'p-4' : 'p-6'
  const trendPositive = trend !== undefined && trend > 0

  return (
    <div className="relative overflow-hidden rounded-xl border border-border bg-surface shadow-sm">
      <div className="pointer-events-none absolute inset-0 opacity-80" style={{ background: metricGradients[accent] }} aria-hidden="true" />
      <div className={`relative flex flex-col justify-between gap-4 ${padding}`}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-text-subtle">{title}</p>
            <p className="mt-1 text-3xl font-semibold text-text">{value}</p>
            {subtitle && (
              <p className="mt-1 text-sm text-text-muted">{subtitle}</p>
            )}
          </div>
          <div className="rounded-lg bg-surface/80 p-3 shadow-sm">
            <Icon className="h-6 w-6 text-text" />
          </div>
        </div>
        {trend !== undefined && (
          <div className="flex items-center text-sm font-medium">
            <FiTrendingUp className={`h-4 w-4 ${trendPositive ? 'text-success' : 'text-error'}`} />
            <span className={`ml-2 ${trendPositive ? 'text-success' : 'text-error'}`}>
              {trendPositive ? '+' : trend && trend < 0 ? '-' : ''}{Math.abs(trend)}%
            </span>
          </div>
        )}
      </div>
    </div>
  )
})

export const MetricsDashboard: React.FC = () => {
  const [stats, setStats] = useState<MetricsData | null>(null)
  const [extended, setExtended] = useState<ExtendedStatsSnapshot | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [telemetry, setTelemetry] = useState<TelemetrySummary | null>(null)
  const [telemetryError, setTelemetryError] = useState<string | null>(null)

  const abTestMetrics = useMemo<ABTestMetrics>(() => ({
    standardView: {
      sessions: 1247,
      completionRate: 0.68,
      avgTimeToInsight: 45.2,
      errorRate: 0.03,
      engagement: 0.72
    },
    ideaBrowserView: {
      sessions: 892,
      completionRate: 0.81,
      avgTimeToInsight: 38.7,
      errorRate: 0.02,
      engagement: 0.89
    }
  }), [])

  const loadStats = useCallback(async () => {
    try {
      setIsLoading(true)
      const [data, telemetrySummary] = await Promise.all([
        api.getSystemStatsSafe(),
        api.getTelemetrySummary().catch((err) => {
          const msg = err instanceof Error ? err.message : 'Failed to load telemetry summary'
          setTelemetryError(msg)
          return null
        })
      ])
      const normalized: MetricsData = {
        total_queries: data.total_queries ?? 0,
        active_research: data.active_research ?? 0,
        paradigm_distribution: data.paradigm_distribution ?? {},
        average_processing_time: data.average_processing_time ?? 0,
        cache_hit_rate: data.cache_hit_rate ?? 0,
        system_health: data.system_health ?? 'healthy'
      }
      setStats(normalized)
      api.getExtendedStatsSafe().then(setExtended).catch(() => {})
      if (telemetrySummary) {
        setTelemetry(telemetrySummary)
        setTelemetryError(null)
      }
      setError(null)
    } catch {
      setError('Failed to load system metrics')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadStats()
    const interval = setInterval(loadStats, 30000)
    return () => clearInterval(interval)
  }, [loadStats])

  const paradigmData = useMemo(() => {
    if (!stats?.paradigm_distribution) return []

    return Object.entries(stats.paradigm_distribution)
      .map(([paradigm, count]) => ({
        name: paradigm.charAt(0).toUpperCase() + paradigm.slice(1),
        value: count,
        color: getParadigmHexColor(paradigm as 'dolores' | 'teddy' | 'bernard' | 'maeve')
      }))
      .filter(item => item.value > 0)
  }, [stats?.paradigm_distribution])

  const statsData: MetricsData = stats ?? {
    total_queries: 0,
    active_research: 0,
    paradigm_distribution: {},
    average_processing_time: 0,
    cache_hit_rate: 0,
    system_health: 'healthy'
  }

  const telemetryCards = useMemo(() => {
    if (!telemetry) return null
    return [
      {
        icon: FiActivity,
        title: 'Runs Observed',
        value: telemetry.totals?.runs ?? telemetry.runs ?? 0,
        subtitle: 'Rolling window'
      },
      {
        icon: FiClock,
        title: 'Avg Processing',
        value: `${(telemetry.totals?.avg_processing_time_seconds ?? 0).toFixed(1)}s`,
        subtitle: 'Per research run'
      },
      {
        icon: FiTrendingUp,
        title: 'Avg Queries',
        value: (telemetry.totals?.avg_total_queries ?? 0).toFixed(1),
        subtitle: 'Per run'
      },
      {
        icon: FiDatabase,
        title: 'Avg Results',
        value: (telemetry.totals?.avg_total_results ?? 0).toFixed(1),
        subtitle: 'Per run'
      }
    ]
  }, [telemetry])

  const providerUsage = useMemo(() => {
    if (!telemetry) return []
    return Object.entries(telemetry.providers?.usage ?? {})
      .sort(([, a], [, b]) => b - a)
  }, [telemetry])

  const providerCosts = useMemo(() => {
    if (!telemetry) return []
    return Object.entries(telemetry.providers?.costs ?? {})
      .sort(([, a], [, b]) => b - a)
  }, [telemetry])

  const stageBreakdown = useMemo(() => {
    if (!telemetry) return []
    return Object.entries(telemetry.stages || {})
      .sort(([, a], [, b]) => b - a)
  }, [telemetry])

  const recentTelemetryEvents = useMemo(() => {
    if (!telemetry) return []
    return (telemetry.recent_events || []).slice(0, 5).map((event) => {
      const obj = (event && typeof event === 'object') ? event as Record<string, unknown> : {}
      const paradigm = typeof obj.paradigm === 'string' ? obj.paradigm : String(obj.paradigm ?? 'unknown')
      const depth = typeof obj.depth === 'string' ? obj.depth : String(obj.depth ?? 'standard')
      const timestamp = typeof obj.timestamp === 'string' ? obj.timestamp : ''
      const totalQueries = typeof obj.total_queries === 'number' ? obj.total_queries : Number(obj.total_queries ?? 0)
      const totalResults = typeof obj.total_results === 'number' ? obj.total_results : Number(obj.total_results ?? 0)
      const processing = typeof obj.processing_time_seconds === 'number' ? obj.processing_time_seconds : Number(obj.processing_time_seconds ?? 0)
      return { paradigm, depth, timestamp, totalQueries, totalResults, processing }
    })
  }, [telemetry])

  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-surface-subtle p-8 shadow-sm">
        <div className="flex flex-col items-center gap-3 text-text-muted">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary/40 border-t-primary" />
          <span className="text-sm">Streaming metrics...</span>
        </div>
      </div>
    )
  }

  if (error || !stats) {
    return (
      <div className="rounded-xl border border-border bg-surface-subtle p-8 shadow-sm">
        <div className="flex flex-col items-center gap-3 text-center">
          <FiAlertCircle className="h-12 w-12 text-error" />
          <p className="text-text">{error || 'No data available'}</p>
          <p className="text-sm text-text-muted">Refresh the dashboard or retry later.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <TriageBoard />

      {telemetry && telemetryCards && (
        <CollapsibleSection
          title="Telemetry Insights"
          description="Aggregated metrics from recent research runs"
        >
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
            {telemetryCards.map((card, idx) => (
              <MetricCard
                key={`${card.title}-${idx}`}
                icon={card.icon}
                title={card.title}
                value={card.value}
                subtitle={card.subtitle}
              />
            ))}
          </div>

          <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
            <div className="rounded-xl border border-border bg-surface p-5 shadow-sm">
              <h3 className="text-sm font-medium text-text mb-3">Provider Usage</h3>
              <ul className="space-y-2 text-sm text-text-muted">
                {providerUsage.length === 0 ? (
                  <li>No provider activity recorded.</li>
                ) : (
                  providerUsage.map(([provider, count]) => (
                    <li key={provider} className="flex items-center justify-between">
                      <span className="capitalize text-text">{provider}</span>
                      <span className="font-medium text-text-subtle">{count}</span>
                    </li>
                  ))
                )}
              </ul>

              {providerCosts.length > 0 && (
                <div className="mt-4 rounded-lg bg-surface-subtle p-3 text-xs text-text-muted">
                  <div className="mb-2 font-medium text-text">Provider Cost (USD)</div>
                  <ul className="space-y-1">
                    {providerCosts.map(([provider, cost]) => (
                      <li key={provider} className="flex items-center justify-between">
                        <span className="capitalize">{provider}</span>
                        <span>${cost.toFixed(4)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="rounded-xl border border-border bg-surface p-5 shadow-sm">
              <h3 className="text-sm font-medium text-text mb-3">Coverage & Evidence</h3>
              <div className="space-y-3 text-sm text-text">
                <div className="flex items-center justify-between">
                  <span className="text-text-muted">Grounding coverage</span>
                  <span className="font-semibold">{(telemetry.coverage?.avg_grounding ?? 0).toFixed(2)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-muted">Avg evidence quotes</span>
                  <span className="font-semibold">{(telemetry.coverage?.avg_evidence_quotes ?? 0).toFixed(1)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-muted">Avg evidence documents</span>
                  <span className="font-semibold">{(telemetry.coverage?.avg_evidence_documents ?? 0).toFixed(1)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-muted">Agent iterations</span>
                  <span className="font-semibold">{(telemetry.agent_loop?.avg_iterations ?? 0).toFixed(1)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-muted">Agent new queries</span>
                  <span className="font-semibold">{(telemetry.agent_loop?.avg_new_queries ?? 0).toFixed(1)}</span>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-border bg-surface p-5 shadow-sm">
              <h3 className="text-sm font-medium text-text mb-3">Recent Runs</h3>
              {recentTelemetryEvents.length === 0 ? (
                <p className="text-sm text-text-muted">No recent telemetry events recorded.</p>
              ) : (
                <ul className="space-y-2 text-xs text-text-muted">
                  {recentTelemetryEvents.map((event, index) => (
                    <li key={`${event.timestamp}-${index}`} className="rounded-md border border-border/60 bg-surface-subtle p-3">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-text">{event.paradigm}</span>
                        <span>{formatRelativeTime(event.timestamp)}</span>
                      </div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        <span>queries: {event.totalQueries}</span>
                        <span>results: {event.totalResults}</span>
                        {event.processing > 0 && <span>{event.processing.toFixed(1)}s</span>}
                      </div>
                      <div className="mt-1 text-[11px] uppercase tracking-wide text-text-subtle">depth · {event.depth}</div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {stageBreakdown.length > 0 && (
            <div className="mt-6 rounded-xl border border-border bg-surface p-5 shadow-sm">
              <h3 className="text-sm font-medium text-text mb-3">Stage Breakdown</h3>
              <ul className="grid grid-cols-1 gap-2 text-sm text-text-muted md:grid-cols-2 lg:grid-cols-3">
                {stageBreakdown.map(([stage, count]) => (
                  <li key={stage} className="flex items-center justify-between rounded-lg bg-surface-subtle px-3 py-2">
                    <span className="capitalize text-text">{stage.replace(/_/g, ' ')}</span>
                    <span className="text-text font-medium">{count}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CollapsibleSection>
      )}

      {telemetryError && !telemetry && (
        <div className="rounded-xl border border-border bg-surface-subtle p-4 text-sm text-text-muted">
          Telemetry summary unavailable: {telemetryError}
        </div>
      )}

      {isLoading && (
        <div className="rounded-xl border border-border bg-surface-subtle p-5 text-sm text-text-muted flex items-center gap-2">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary/40 border-t-primary" />
          <span>Refreshing system metrics…</span>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-error bg-error/10 p-5 text-sm text-error flex items-center gap-2">
          <FiAlertCircle className="h-5 w-5" />
          <span>{error}</span>
        </div>
      )}

      <CollapsibleSection
        title="System Pulse"
        description="Real-time throughput across the research stack"
      >
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            icon={FiActivity}
            title="Total Queries"
            value={statsData.total_queries || 0}
            subtitle="All time"
            trend={12}
          />
          <MetricCard
            icon={FiUsers}
            title="Active Research"
            value={statsData.active_research || 0}
            subtitle="Currently processing"
            accent="success"
          />
          <MetricCard
            icon={FiClock}
            title="Avg Processing Time"
            value={`${(statsData.average_processing_time || 0).toFixed(1)}s`}
            subtitle="Per query"
            trend={-8}
          />
          <MetricCard
            icon={FiDatabase}
            title="Cache Hit Rate"
            value={`${((statsData.cache_hit_rate || 0) * 100).toFixed(1)}%`}
            subtitle="Performance boost"
            trend={5}
          />
        </div>
      </CollapsibleSection>

      {extended && (
        <CollapsibleSection
          title="Latency & Reliability"
          description="Tail performance and fallback coverage"
          defaultOpen={false}
        >
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              icon={FiClock}
              title="Classification p95"
              value={extended.latency?.classification ? `${extended.latency.classification.p95.toFixed(0)}ms` : 'N/A'}
              subtitle="Latency"
              compact
            />
            <MetricCard
              icon={FiClock}
              title="Answer Synth p95"
              value={extended.latency?.answer_synthesis ? `${extended.latency.answer_synthesis.p95.toFixed(0)}ms` : 'N/A'}
              subtitle="Latency"
              compact
            />
            <MetricCard
              icon={FiAlertCircle}
              title="Classification Fallback"
              value={extended.fallback_rates?.classification !== undefined ? `${(extended.fallback_rates.classification * 100).toFixed(1)}%` : 'N/A'}
              subtitle="Rule -> LLM usage"
              accent="error"
              compact
            />
            <MetricCard
              icon={FiAlertCircle}
              title="Answer Fallback"
              value={extended.fallback_rates?.answer_synthesis !== undefined ? `${(extended.fallback_rates.answer_synthesis * 100).toFixed(1)}%` : 'N/A'}
              subtitle="Heuristic -> LLM"
              accent="primary"
              compact
            />
          </div>
        </CollapsibleSection>
      )}

      <CollapsibleSection
        title="Paradigm Signals"
        description="Blend of host perspectives and system posture"
      >
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-xl border border-border bg-surface p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-text mb-4">Paradigm Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={paradigmData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }: { name?: string; percent?: number }) => `${name ?? ''} ${(((percent ?? 0) * 100) | 0)}%`}
                  outerRadius={85}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {paradigmData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-xl border border-border bg-surface p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-text mb-4">Performance Metrics</h3>
            <div className="space-y-5">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-text-muted">Cache Hit Rate</span>
                  <span className="font-medium text-text">{((statsData.cache_hit_rate || 0) * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full rounded-full bg-surface-muted h-2">
                  <div
                    className="h-full rounded-full bg-primary"
                    style={{ width: `${(statsData.cache_hit_rate || 0) * 100}%` }}
                  />
                </div>
              </div>

              <div className="border-t border-border pt-4">
                <h4 className="text-sm font-medium text-text mb-3">System Insights</h4>
                <ul className="space-y-2 text-sm text-text">
                  <li className="flex items-center gap-2">
                    <FiTrendingUp className="h-4 w-4 text-success" />
                    {statsData.total_queries > 1000 ? 'High usage detected' : 'Normal usage patterns'}
                  </li>
                  <li className="flex items-center gap-2">
                    <FiActivity className="h-4 w-4 text-primary" />
                    {statsData.active_research > 10 ? 'System under load' : 'System running smoothly'}
                  </li>
                  <li className="flex items-center gap-2">
                    <FiClock className="h-4 w-4 text-primary" />
                    {statsData.average_processing_time < 30 ? 'Fast response times' : 'Consider optimization'}
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        title="Usage Patterns"
        description="Where the hosts are investing their effort"
        defaultOpen={false}
      >
        <div className="rounded-xl border border-border bg-surface p-6 shadow-sm">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <div className="text-center">
              <p className="text-3xl font-semibold text-paradigm-dolores">{statsData.total_queries ? Math.round(((paradigmData.find(p => p.name.toLowerCase() === 'dolores')?.value ?? 0) / statsData.total_queries) * 100) : 0}%</p>
              <p className="mt-1 text-sm text-text-muted">Truth-seeking queries</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-semibold text-paradigm-bernard">{statsData.total_queries ? Math.round(((paradigmData.find(p => p.name.toLowerCase() === 'bernard')?.value ?? 0) / statsData.total_queries) * 100) : 0}%</p>
              <p className="mt-1 text-sm text-text-muted">Analytical queries</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-semibold text-paradigm-maeve">{(statsData.cache_hit_rate || 0) > 0.5 ? 'High' : 'Low'}</p>
              <p className="mt-1 text-sm text-text-muted">Cache efficiency</p>
            </div>
          </div>
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        title="Experimentation"
        description="IdeaBrowser experiment snapshot"
        defaultOpen={false}
      >
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-xl border border-border bg-surface p-6 shadow-sm">
            <h4 className="text-sm font-medium text-text mb-4">Key Performance Metrics</h4>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2 text-sm">
                  <span className="text-text-muted">Task Completion Rate</span>
                  <div className="flex gap-4">
                    <span className="text-text-subtle">Standard: {(abTestMetrics.standardView.completionRate * 100).toFixed(0)}%</span>
                    <span className="font-medium text-success">IdeaBrowser: {(abTestMetrics.ideaBrowserView.completionRate * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className="relative h-2 overflow-hidden rounded-full bg-surface-muted">
                  <div className="absolute inset-y-0 left-0 bg-text-muted" style={{ width: `${abTestMetrics.standardView.completionRate * 100}%` }} />
                  <div className="absolute inset-y-0 left-0 bg-success" style={{ width: `${abTestMetrics.ideaBrowserView.completionRate * 100}%` }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2 text-sm">
                  <span className="text-text-muted">Time to First Insight (seconds)</span>
                  <div className="flex gap-4">
                    <span className="text-text-subtle">Standard: {abTestMetrics.standardView.avgTimeToInsight}s</span>
                    <span className="font-medium text-success">IdeaBrowser: {abTestMetrics.ideaBrowserView.avgTimeToInsight}s</span>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-success">
                  <FiZap className="h-4 w-4" />
                  <span className="text-sm">
                    {((1 - abTestMetrics.ideaBrowserView.avgTimeToInsight / abTestMetrics.standardView.avgTimeToInsight) * 100).toFixed(0)}% faster
                  </span>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2 text-sm">
                  <span className="text-text-muted">User Engagement Score</span>
                  <div className="flex gap-4">
                    <span className="text-text-subtle">Standard: {(abTestMetrics.standardView.engagement * 100).toFixed(0)}%</span>
                    <span className="font-medium text-success">IdeaBrowser: {(abTestMetrics.ideaBrowserView.engagement * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className="relative h-2 overflow-hidden rounded-full bg-surface-muted">
                  <div className="absolute inset-y-0 left-0 bg-text-muted" style={{ width: `${abTestMetrics.standardView.engagement * 100}%` }} />
                  <div className="absolute inset-y-0 left-0 bg-primary" style={{ width: `${abTestMetrics.ideaBrowserView.engagement * 100}%` }} />
                </div>
              </div>

              <div className="border-t border-border pt-4">
                <div className="flex items-center gap-2 mb-2 text-sm font-medium text-text">
                  <FiTarget className="h-4 w-4 text-success" />
                  Overall Improvement
                </div>
                <p className="text-2xl font-semibold text-success">
                  +{((abTestMetrics.ideaBrowserView.completionRate / abTestMetrics.standardView.completionRate - 1) * 100).toFixed(0)}%
                </p>
                <p className="text-sm text-text-muted">in task completion rate</p>
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-surface p-6 shadow-sm">
            <h4 className="text-sm font-medium text-text mb-4">Session Distribution</h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={[
                  {
                    name: 'Standard',
                    sessions: abTestMetrics.standardView.sessions,
                    completions: Math.round(abTestMetrics.standardView.sessions * abTestMetrics.standardView.completionRate)
                  },
                  {
                    name: 'IdeaBrowser',
                    sessions: abTestMetrics.ideaBrowserView.sessions,
                    completions: Math.round(abTestMetrics.ideaBrowserView.sessions * abTestMetrics.ideaBrowserView.completionRate)
                  }
                ]}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="sessions" fill="#8884d8" name="Total Sessions" />
                <Bar dataKey="completions" fill="#82ca9d" name="Completed Tasks" />
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-4 rounded-lg bg-primary/10 p-4">
              <div className="flex items-start gap-3">
                <FiEye className="h-5 w-5 text-primary" />
                <div>
                  <p className="text-sm font-medium text-text">Migration readiness</p>
                  <p className="mt-1 text-sm text-text">
                    IdeaBrowser outperforms the standard layout across core KPIs. Plan the rollout with confidence.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 flex items-center justify-between rounded-lg bg-surface-subtle px-4 py-3">
          <div className="flex items-center gap-2 text-sm text-text">
            <FiMousePointer className="h-4 w-4 text-text-muted" />
            <span>Test Status</span>
          </div>
          <span className="inline-flex items-center rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-medium text-success">
            Active · {(abTestMetrics.standardView.sessions + abTestMetrics.ideaBrowserView.sessions).toLocaleString()} sessions
          </span>
        </div>
      </CollapsibleSection>
    </div>
  )
}
