import React, { useState, useEffect } from 'react'
import { Search, Settings2, Zap, Database, Brain } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import type { ResearchOptions } from '../services/api'

interface ResearchFormEnhancedProps {
  onSubmit: (query: string, options: ResearchOptions) => void
  isLoading: boolean
}

export const ResearchFormEnhanced: React.FC<ResearchFormEnhancedProps> = ({ onSubmit, isLoading }) => {
  const { user } = useAuth()
  const [query, setQuery] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [options, setOptions] = useState<ResearchOptions>({
    depth: user?.preferences?.default_depth || 'standard',
    include_secondary: true,
    max_sources: 100,
    enable_real_search: user?.preferences?.enable_real_search || false,
    enable_ai_classification: user?.preferences?.enable_ai_classification || false,
  })

  useEffect(() => {
    // Update options when user preferences change
    if (user?.preferences) {
      setOptions(prev => ({
        ...prev,
        depth: user.preferences?.default_depth || prev.depth,
        enable_real_search: user.preferences?.enable_real_search || false,
        enable_ai_classification: user.preferences?.enable_ai_classification || false,
      }))
    }
  }, [user])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSubmit(query, options)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-md p-6">
      <div className="space-y-4">
        <div>
          <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
            Research Query
          </label>
          <div className="relative">
            <input
              id="query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What would you like to research?"
              className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading}
            />
            <Search className="absolute right-3 top-3.5 h-5 w-5 text-gray-400" />
          </div>
        </div>

        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
          >
            <Settings2 className="h-4 w-4" />
            Advanced Options
          </button>

          <div className="flex items-center gap-4 text-sm">
            {options.enable_real_search && (
              <span className="flex items-center gap-1 text-green-600">
                <Database className="h-4 w-4" />
                Real Search
              </span>
            )}
            {options.enable_ai_classification && (
              <span className="flex items-center gap-1 text-purple-600">
                <Brain className="h-4 w-4" />
                AI Classification
              </span>
            )}
          </div>
        </div>

        {showAdvanced && (
          <div className="border-t pt-4 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Research Depth
              </label>
              <div className="grid grid-cols-3 gap-2">
                {(['quick', 'standard', 'deep'] as const).map((depth) => (
                  <button
                    key={depth}
                    type="button"
                    onClick={() => setOptions({ ...options, depth })}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      options.depth === depth
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    <Zap className={`h-4 w-4 inline mr-1 ${depth === 'quick' ? '' : 'opacity-50'}`} />
                    {depth.charAt(0).toUpperCase() + depth.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={options.include_secondary || false}
                  onChange={(e) => setOptions({ ...options, include_secondary: e.target.checked })}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-700">Include secondary paradigms</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={options.enable_real_search || false}
                  onChange={(e) => setOptions({ ...options, enable_real_search: e.target.checked })}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-700">Enable real search APIs</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={options.enable_ai_classification || false}
                  onChange={(e) => setOptions({ ...options, enable_ai_classification: e.target.checked })}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-700">Enable AI classification</span>
              </label>

              <div>
                <label className="block text-sm text-gray-700 mb-1">Max sources</label>
                <input
                  type="number"
                  min="10"
                  max="200"
                  value={options.max_sources || 100}
                  onChange={(e) => setOptions({ ...options, max_sources: parseInt(e.target.value) })}
                  className="w-full px-3 py-1 text-sm border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading || !query.trim()}
          className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? 'Researching...' : 'Start Research'}
        </button>
      </div>
    </form>
  )
}