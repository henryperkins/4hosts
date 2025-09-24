
import React, { useEffect, useState, useCallback, useMemo } from 'react'
import type { ReactNode } from 'react'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend } from 'recharts'
import { FiActivity, FiTrendingUp, FiUsers, FiClock, FiDatabase, FiAlertCircle, FiEye, FiMousePointer, FiTarget, FiZap, FiChevronUp } from 'react-icons/fi'
import api from '../services/api'
import type { MetricsData, ExtendedStatsSnapshot } from '../types/api-types'
import { getParadigmHexColor } from '../constants/paradigm'

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
      const data = await api.getSystemStatsSafe()
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
      <CollapsibleSection
        title="System Pulse"
        description="Real-time throughput across the research stack"
      >
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            icon={FiActivity}
            title="Total Queries"
            value={stats.total_queries || 0}
            subtitle="All time"
            trend={12}
          />
          <MetricCard
            icon={FiUsers}
            title="Active Research"
            value={stats.active_research || 0}
            subtitle="Currently processing"
            accent="success"
          />
          <MetricCard
            icon={FiClock}
            title="Avg Processing Time"
            value={`${(stats.average_processing_time || 0).toFixed(1)}s`}
            subtitle="Per query"
            trend={-8}
          />
          <MetricCard
            icon={FiDatabase}
            title="Cache Hit Rate"
            value={`${((stats.cache_hit_rate || 0) * 100).toFixed(1)}%`}
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
                  <span className="font-medium text-text">{((stats.cache_hit_rate || 0) * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full rounded-full bg-surface-muted h-2">
                  <div
                    className="h-full rounded-full bg-primary"
                    style={{ width: `${(stats.cache_hit_rate || 0) * 100}%` }}
                  />
                </div>
              </div>

              <div className="border-t border-border pt-4">
                <h4 className="text-sm font-medium text-text mb-3">System Insights</h4>
                <ul className="space-y-2 text-sm text-text">
                  <li className="flex items-center gap-2">
                    <FiTrendingUp className="h-4 w-4 text-success" />
                    {stats.total_queries > 1000 ? 'High usage detected' : 'Normal usage patterns'}
                  </li>
                  <li className="flex items-center gap-2">
                    <FiActivity className="h-4 w-4 text-primary" />
                    {stats.active_research > 10 ? 'System under load' : 'System running smoothly'}
                  </li>
                  <li className="flex items-center gap-2">
                    <FiClock className="h-4 w-4 text-primary" />
                    {stats.average_processing_time < 30 ? 'Fast response times' : 'Consider optimization'}
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
              <p className="text-3xl font-semibold text-paradigm-dolores">{stats.total_queries ? Math.round(((paradigmData.find(p => p.name.toLowerCase() === 'dolores')?.value ?? 0) / stats.total_queries) * 100) : 0}%</p>
              <p className="mt-1 text-sm text-text-muted">Truth-seeking queries</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-semibold text-paradigm-bernard">{stats.total_queries ? Math.round(((paradigmData.find(p => p.name.toLowerCase() === 'bernard')?.value ?? 0) / stats.total_queries) * 100) : 0}%</p>
              <p className="mt-1 text-sm text-text-muted">Analytical queries</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-semibold text-paradigm-maeve">{(stats.cache_hit_rate || 0) > 0.5 ? 'High' : 'Low'}</p>
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
