import React, { useEffect, useState } from 'react'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'
// Additional chart components available if needed:
// import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Legend } from 'recharts'
import { Activity, TrendingUp, Users, Clock, Database, AlertCircle } from 'lucide-react'
import api from '../services/api'
import type { SystemStats } from '../services/api'

const paradigmColors = {
  dolores: '#EF4444',
  teddy: '#3B82F6',
  bernard: '#10B981',
  maeve: '#8B5CF6',
}

export const MetricsDashboard: React.FC = () => {
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadStats()
    const interval = setInterval(loadStats, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const loadStats = async () => {
    try {
      const data = await api.getSystemStats()
      setStats(data)
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
    paradigm: paradigm as keyof typeof paradigmColors,
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
                  <Cell key={`cell-${index}`} fill={paradigmColors[entry.paradigm]} />
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
              {Math.round(paradigmData.find(p => p.paradigm === 'dolores')?.value || 0 / stats.total_queries * 100)}%
            </p>
            <p className="text-sm text-gray-600 mt-1">Truth-seeking queries</p>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold text-green-600">
              {Math.round(paradigmData.find(p => p.paradigm === 'bernard')?.value || 0 / stats.total_queries * 100)}%
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
    </div>
  )
}