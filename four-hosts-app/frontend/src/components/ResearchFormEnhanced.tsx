import React, { useState, useEffect } from 'react'
import { Settings2, Loader2, Sparkles, Users, AlertCircle } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import type { ResearchOptions } from '../types'

interface ResearchFormEnhancedProps {
  onSubmit: (query: string, options: ResearchOptions) => void
  isLoading: boolean
}

export const ResearchFormEnhanced: React.FC<ResearchFormEnhancedProps> = ({ onSubmit, isLoading }) => {
  const { user } = useAuth()
  const [query, setQuery] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [error, setError] = useState('')
  const [paradigm, setParadigm] = useState('auto')
  const [depth, setDepth] = useState('standard')
  
  const [options, setOptions] = useState<ResearchOptions>({
    depth: user?.preferences?.default_depth || 'standard',
    include_secondary: true,
    max_sources: 50,
    enable_real_search: user?.preferences?.enable_real_search !== false,
    language: 'en',
    region: 'us',
    enable_ai_classification: true
  })

  useEffect(() => {
    // Update options when user preferences change
    if (user?.preferences) {
      setOptions(prev => ({
        ...prev,
        depth: user.preferences?.default_depth || prev.depth,
        enable_real_search: user.preferences?.enable_real_search !== false,
      }))
    }
  }, [user])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSubmit(query, { ...options, depth: depth as 'quick' | 'standard' | 'deep' })
    }
  }

  const paradigmOptions = [
    { value: 'auto', label: 'Auto', icon: '🔮', description: 'Let AI choose', color: 'text-purple-600 dark:text-purple-400' },
    { value: 'dolores', label: 'Dolores', icon: '🛡️', description: 'Truth & Justice', color: 'text-red-600 dark:text-red-400' },
    { value: 'bernard', label: 'Bernard', icon: '🧠', description: 'Analysis & Logic', color: 'text-blue-600 dark:text-blue-400' },
    { value: 'teddy', label: 'Teddy', icon: '❤️', description: 'Care & Support', color: 'text-orange-600 dark:text-orange-400' },
    { value: 'maeve', label: 'Maeve', icon: '📈', description: 'Strategy & Power', color: 'text-green-600 dark:text-green-400' }
  ]

  return (
    <form onSubmit={handleSubmit} className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-8 animate-scale-in transition-all duration-300 hover:shadow-3xl">
      <div className="space-y-6">
        {/* Query Input */}
        <div className="animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
          <label htmlFor="query" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Research Query
          </label>
          <div className="relative">
            <textarea
              id="query"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
                setError('')
              }}
              placeholder="What would you like to research today?"
              className={`w-full px-4 py-3 border-2 rounded-xl transition-all duration-200 resize-none
                ${error
                  ? 'border-red-500 dark:border-red-400 focus:ring-red-500 dark:focus:ring-red-400'
                  : 'border-gray-300 dark:border-gray-600 focus:border-blue-500 dark:focus:border-blue-400'
                }
                bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100
                placeholder-gray-400 dark:placeholder-gray-500
                focus:ring-2 focus:ring-offset-2 dark:focus:ring-offset-gray-800 focus:outline-none
                hover:border-gray-400 dark:hover:border-gray-500`}
              rows={3}
              disabled={isLoading}
            />
            {error && (
              <div className="absolute -bottom-6 left-0 flex items-center text-red-600 dark:text-red-400 text-sm animate-shake">
                <AlertCircle className="h-4 w-4 mr-1" />
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Paradigm Selection */}
        <div className="animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Paradigm Selection
          </label>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {paradigmOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setParadigm(option.value)}
                disabled={isLoading}
                className={`relative p-3 rounded-xl border-2 transition-all duration-200 transform hover:scale-105 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-offset-2 dark:focus:ring-offset-gray-800
                  ${paradigm === option.value
                    ? `border-blue-500 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/30 shadow-lg scale-105 ${
                        option.value !== 'auto' ? `ring-2 ring-blue-500/50` : ''
                      }`
                    : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 bg-white dark:bg-gray-800'
                  }
                  ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                  group`}
              >
                <div className="text-center">
                  <div className={`text-2xl mb-1 transition-transform duration-200 group-hover:scale-110 ${
                    paradigm === option.value ? 'animate-bounce-in' : ''
                  }`}>
                    {option.icon}
                  </div>
                  <div className={`font-medium text-sm ${option.color || 'text-gray-900 dark:text-gray-100'}`}>
                    {option.label}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    {option.description}
                  </div>
                </div>
                {paradigm === option.value && (
                  <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-transparent to-blue-500/10 dark:to-blue-400/10 pointer-events-none animate-pulse" />
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Depth Selection */}
        <div className="animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Research Depth
          </label>
          <div className="flex gap-3">
            {[
              { value: 'quick', label: 'Quick', description: 'Fast overview' },
              { value: 'standard', label: 'Standard', description: 'Balanced analysis' },
              { value: 'deep', label: 'Deep', description: 'Comprehensive research' },
            ].map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setDepth(option.value)}
                disabled={isLoading}
                className={`flex-1 p-3 rounded-xl border-2 transition-all duration-200 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-offset-2 dark:focus:ring-offset-gray-800
                  ${depth === option.value
                    ? 'border-blue-500 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/30 shadow-lg scale-105'
                    : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 bg-white dark:bg-gray-800'
                  }
                  ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
              >
                <div className="font-medium text-gray-900 dark:text-gray-100">{option.label}</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">{option.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Advanced Options Toggle */}
        <div className="animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors duration-200"
          >
            <Settings2 className={`h-4 w-4 transition-transform duration-200 ${showAdvanced ? 'rotate-90' : ''}`} />
            {showAdvanced ? 'Hide' : 'Show'} Advanced Options
          </button>
        </div>

        {/* Advanced Options */}
        {showAdvanced && (
          <div className="animate-fade-in-up space-y-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="flex items-center cursor-pointer group">
                  <input
                    type="checkbox"
                    checked={options.enable_ai_classification}
                    onChange={(e) => setOptions({ ...options, enable_ai_classification: e.target.checked })}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded dark:bg-gray-700 transition-transform duration-200 group-hover:scale-110"
                  />
                  <span className="ml-2 text-sm text-gray-700 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-gray-100 transition-colors duration-200">
                    Enable AI classification
                  </span>
                </label>
              </div>

              <div>
                <label className="block text-sm text-gray-700 dark:text-gray-300 mb-1">
                  Max sources
                </label>
                <input
                  type="number"
                  min="10"
                  max="200"
                  value={options.max_sources || 100}
                  onChange={(e) => setOptions({ ...options, max_sources: parseInt(e.target.value) })}
                  className="w-full px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent dark:bg-gray-700 dark:text-white transition-colors duration-200"
                />
              </div>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="w-full bg-gradient-to-r from-blue-500 to-blue-600 dark:from-blue-600 dark:to-blue-700 text-white py-3 px-4 rounded-xl font-medium hover:from-blue-600 hover:to-blue-700 dark:hover:from-blue-700 dark:hover:to-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:ring-offset-2 dark:focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-[1.02] hover:shadow-lg group relative overflow-hidden"
          >
            <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -skew-x-12 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-700"></span>
            {isLoading ? (
              <span className="flex items-center justify-center relative">
                <Loader2 className="h-5 w-5 animate-spin mr-2" />
                <span className="animate-pulse">Researching...</span>
              </span>
            ) : (
              <span className="flex items-center justify-center relative">
                <Users className="h-5 w-5 mr-2 opacity-0 group-hover:opacity-100 transition-all duration-300 transform group-hover:rotate-12" />
                <span>Ask the Four Hosts</span>
                <Sparkles className="h-5 w-5 ml-2 opacity-0 group-hover:opacity-100 transition-all duration-300 transform group-hover:-rotate-12" />
              </span>
            )}
          </button>
        </div>
      </div>
    </form>
  )
}
