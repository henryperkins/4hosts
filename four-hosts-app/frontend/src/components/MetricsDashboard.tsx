import React, { useEffect, useState } from 'react'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend } from 'recharts'
import { Activity, TrendingUp, Users, Clock, Database, AlertCircle, Eye, MousePointer, Target, Zap } from 'lucide-react'
import api from '../services/api'
import type { SystemStats } from '../services/api'
import { getParadigmHexColor } from '../constants/paradigm'

// Mock A/B test data - in production, this would come from your analytics service
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

export const MetricsDashboard: React.FC = () => {
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [abTestMetrics] = useState<ABTestMetrics>({
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
  })

  useEffect(() => {
    loadStats()
    const interval = setInterval(loadStats, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const loadStats = async () => {
    try {
      const data = await api.getSystemStatsSafe()
      // Normalize to SystemStats-like shape for rendering
      const normalized: SystemStats = {
        total_queries: (data as any).total_queries ?? 0,
        active_research: (data as any).active_research ?? 0,
        paradigm_distribution: (data as any).paradigm_distribution ?? {},
        average_processing_time: (data as any).average_processing_time ?? 0,
        cache_hit_rate: (data as any).cache_hit_rate ?? 0,
        system_health: (data as any).system_status ?? (data as any).system_health ?? 'healthy'
      }
      setStats(normalized)
      setError(null)
    } catch {
      setError('Failed to load system metrics')
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      </div>
    )
  }

  if (error || !stats) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-600">{error || 'No data available'}</p>
        </div>
      </div>
    )
  }

  const paradigmData = Object.entries(stats.paradigm_distribution).map(([paradigm, count]) => ({
    name: paradigm.charAt(0).toUpperCase() + paradigm.slice(1),
    value: count,
    paradigm: paradigm,
  }))

  const healthColor = {
    healthy: 'text-green-600 bg-green-100',
    degraded: 'text-yellow-600 bg-yellow-100',
    critical: 'text-red-600 bg-red-100',
  }[stats.system_health]

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Queries</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total_queries.toLocaleString()}</p>
            </div>
            <Users className="h-8 w-8 text-blue-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active Research</p>
              <p className="text-2xl font-bold text-gray-900">{stats.active_research}</p>
            </div>
            <Activity className="h-8 w-8 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg Processing Time</p>
              <p className="text-2xl font-bold text-gray-900">{stats.average_processing_time.toFixed(1)}s</p>
            </div>
            <Clock className="h-8 w-8 text-purple-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">System Health</p>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${healthColor}`}>
                {stats.system_health.charAt(0).toUpperCase() + stats.system_health.slice(1)}
              </span>
            </div>
            <Database className="h-8 w-8 text-gray-600" />
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Paradigm Distribution */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Paradigm Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={paradigmData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {paradigmData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getParadigmHexColor(entry.paradigm)} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Performance Metrics */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Cache Hit Rate</span>
                <span className="font-medium">{((stats.cache_hit_rate || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ width: `${(stats.cache_hit_rate || 0) * 100}%` }}
                />
              </div>
            </div>

            <div className="pt-4 border-t">
              <h4 className="text-sm font-medium text-gray-700 mb-2">System Insights</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                  {stats.total_queries > 1000 ? 'High usage detected' : 'Normal usage patterns'}
                </li>
                <li className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-blue-500" />
                  {stats.active_research > 10 ? 'System under load' : 'System running smoothly'}
                </li>
                <li className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-purple-500" />
                  {stats.average_processing_time < 30 ? 'Fast response times' : 'Consider optimization'}
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Additional Insights */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Usage Patterns</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <p className="text-3xl font-bold text-blue-600">
              {Math.round(((paradigmData.find(p => p.paradigm === 'dolores')?.value ?? 0) / stats.total_queries) * 100)}%
            </p>
            <p className="text-sm text-gray-600 mt-1">Truth-seeking queries</p>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold text-green-600">
              {Math.round(((paradigmData.find(p => p.paradigm === 'bernard')?.value ?? 0) / stats.total_queries) * 100)}%
            </p>
            <p className="text-sm text-gray-600 mt-1">Analytical queries</p>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold text-purple-600">
              {(stats.cache_hit_rate || 0) > 0.5 ? 'High' : 'Low'}
            </p>
            <p className="text-sm text-gray-600 mt-1">Cache efficiency</p>
          </div>
        </div>
      </div>

      {/* A/B Test Results */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">A/B Test: Standard vs IdeaBrowser View</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Comparison Metrics */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Key Performance Metrics</h4>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Task Completion Rate</span>
                  <div className="flex gap-4 text-sm">
                    <span className="text-gray-500">Standard: {(abTestMetrics.standardView.completionRate * 100).toFixed(0)}%</span>
                    <span className="font-medium text-green-600">IdeaBrowser: {(abTestMetrics.ideaBrowserView.completionRate * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="absolute top-0 left-0 h-full bg-gray-400"
                    style={{ width: `${abTestMetrics.standardView.completionRate * 100}%` }}
                  />
                  <div
                    className="absolute top-0 left-0 h-full bg-green-600"
                    style={{ width: `${abTestMetrics.ideaBrowserView.completionRate * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Time to First Insight (seconds)</span>
                  <div className="flex gap-4 text-sm">
                    <span className="text-gray-500">Standard: {abTestMetrics.standardView.avgTimeToInsight}s</span>
                    <span className="font-medium text-green-600">IdeaBrowser: {abTestMetrics.ideaBrowserView.avgTimeToInsight}s</span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-green-600">
                    {((1 - abTestMetrics.ideaBrowserView.avgTimeToInsight / abTestMetrics.standardView.avgTimeToInsight) * 100).toFixed(0)}% faster
                  </span>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">User Engagement Score</span>
                  <div className="flex gap-4 text-sm">
                    <span className="text-gray-500">Standard: {(abTestMetrics.standardView.engagement * 100).toFixed(0)}%</span>
                    <span className="font-medium text-green-600">IdeaBrowser: {(abTestMetrics.ideaBrowserView.engagement * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="absolute top-0 left-0 h-full bg-gray-400"
                    style={{ width: `${abTestMetrics.standardView.engagement * 100}%` }}
                  />
                  <div
                    className="absolute top-0 left-0 h-full bg-blue-600"
                    style={{ width: `${abTestMetrics.ideaBrowserView.engagement * 100}%` }}
                  />
                </div>
              </div>

              <div className="pt-4 border-t">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="h-4 w-4 text-green-500" />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Overall Improvement</span>
                </div>
                <p className="text-2xl font-bold text-green-600">
                  +{((abTestMetrics.ideaBrowserView.completionRate / abTestMetrics.standardView.completionRate - 1) * 100).toFixed(0)}%
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">in task completion rate</p>
              </div>
            </div>
          </div>

          {/* Session Distribution */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Session Distribution</h4>
            <ResponsiveContainer width="100%" height={200}>
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

            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="flex items-start gap-2">
                <Eye className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-blue-900 dark:text-blue-100">Migration Readiness</p>
                  <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                    Based on current metrics, IdeaBrowser view shows strong improvements across all KPIs.
                    Consider defaulting to IdeaBrowser view in the next phase.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Test Status */}
        <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MousePointer className="h-5 w-5 text-gray-600" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Test Status</span>
            </div>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Active
            </span>
          </div>
          <div className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            <p>Total sessions analyzed: {abTestMetrics.standardView.sessions + abTestMetrics.ideaBrowserView.sessions}</p>
            <p>Test duration: 2 weeks (Phase 2 - User Testing)</p>
          </div>
        </div>
      </div>
    </div>
  )
}
