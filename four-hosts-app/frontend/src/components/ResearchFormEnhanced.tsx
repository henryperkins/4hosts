import React, { useState, useEffect } from 'react'
import { Settings2, Users } from 'lucide-react'
import { useAuth } from '../hooks/useAuth'
import type { ResearchOptions } from '../types'
import { Button } from './ui/Button'
import { InputField } from './ui/InputField'

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
  const [depth, setDepth] = useState<'quick' | 'standard' | 'deep'>(user?.preferences?.default_depth ?? 'standard')

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
    setError('')
    
    const trimmedQuery = query.trim()
    if (!trimmedQuery) {
      setError('Please enter a research query')
      return
    }
    
    if (trimmedQuery.length < 3) {
      setError('Query must be at least 3 characters long')
      return
    }
    
    onSubmit(trimmedQuery, { ...options, depth })
  }

  const paradigmOptions = [
    { value: 'auto', label: 'Auto', icon: '🔮', description: 'Let AI choose', colorClass: 'text-purple-600 dark:text-purple-400' },
    { value: 'dolores', label: 'Dolores', icon: '🛡️', description: 'Truth & Justice', colorClass: 'text-paradigm-dolores' },
    { value: 'bernard', label: 'Bernard', icon: '🧠', description: 'Analysis & Logic', colorClass: 'text-paradigm-bernard' },
    { value: 'teddy', label: 'Teddy', icon: '❤️', description: 'Care & Support', colorClass: 'text-paradigm-teddy' },
    { value: 'maeve', label: 'Maeve', icon: '📈', description: 'Strategy & Power', colorClass: 'text-paradigm-maeve' }
  ]

  return (
    <form onSubmit={handleSubmit} className="card-hover animate-fade-in">
      <div className="space-y-6">
        {/* Query Input */}
        <div className="animate-slide-up">
          <label htmlFor="query" className="block text-sm font-medium text-text mb-2">
            Research Query
          </label>
          <InputField
            id="query"
            textarea
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setError('')
            }}
            placeholder="What would you like to research today?"
            rows={3}
            disabled={isLoading}
            status={error ? 'error' : undefined}
            errorMessage={error}
            className="border-2"
            required
            minLength={3}
            aria-describedby={error ? "query-error" : undefined}
          />
        </div>

        {/* Paradigm Selection */}
        <div className="animate-slide-up" style={{ animationDelay: '100ms' }}>
          <div className="block text-sm font-medium text-text mb-3" role="group" aria-label="Paradigm Selection">
            Paradigm Selection
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {paradigmOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setParadigm(option.value)}
                disabled={isLoading}
                className={`relative p-3 rounded-lg border transition-colors
                  ${paradigm === option.value
                    ? 'border-primary bg-primary/10 shadow-md'
                    : 'border-border hover:border-text-subtle bg-surface'
                  }
                  ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:shadow-lg'}
                  group`}
                aria-pressed={paradigm === option.value}
                aria-label={`${option.label}: ${option.description}`}
              >
                <div className="text-center">
                  <div className={`text-2xl mb-1 ${
                    paradigm === option.value ? 'animate-fade-in' : ''
                  }`}>
                    {option.icon}
                  </div>
                  <div className={`font-medium text-sm ${option.colorClass || 'text-text'}`}>
                    {option.label}
                  </div>
                  <div className="text-xs text-text-muted mt-1">
                    {option.description}
                  </div>
                </div>
                {paradigm === option.value && (
                  <div className="absolute inset-0 rounded-lg bg-primary/5 pointer-events-none" />
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Depth Selection */}
        <div className="animate-slide-up" style={{ animationDelay: '200ms' }}>
          <label className="block text-sm font-medium text-text mb-3">
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
                onClick={() => setDepth(option.value as 'quick' | 'standard' | 'deep')}
                disabled={isLoading}
                className={`flex-1 p-3 rounded-lg border transition-colors
                  ${depth === option.value
                    ? 'border-primary bg-primary/10 shadow-md'
                    : 'border-border hover:border-text-subtle bg-surface'
                  }
                  ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:shadow-lg'}`}
              >
                <div className="font-medium text-text">{option.label}</div>
                <div className="text-xs text-text-muted mt-1">{option.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Advanced Options Toggle */}
        <div className="animate-slide-up" style={{ animationDelay: '300ms' }}>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-primary hover:text-primary/80 transition-colors"
          >
            <Settings2 className={`h-4 w-4 transition-transform ${showAdvanced ? 'rotate-90' : ''}`} />
            {showAdvanced ? 'Hide' : 'Show'} Advanced Options
          </button>
        </div>

        {/* Advanced Options */}
        {showAdvanced && (
          <div className="animate-slide-up space-y-4 p-4 bg-surface-subtle rounded-lg">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="flex items-center cursor-pointer group">
                  <input
                    type="checkbox"
                    checked={options.enable_ai_classification}
                    onChange={(e) => setOptions({ ...options, enable_ai_classification: e.target.checked })}
                    className="h-4 w-4 text-primary focus:ring-primary border-border rounded"
                  />
                  <span className="ml-2 text-sm text-text">
                    Enable AI classification
                  </span>
                </label>
              </div>

              <div>
                <label className="block text-sm text-text mb-1">
                  Max sources
                </label>
                <InputField
                  type="number"
                  min="10"
                  max="200"
                  value={options.max_sources || 100}
                  onChange={(e) => setOptions({ ...options, max_sources: parseInt(e.target.value) })}
                  className="input text-sm"
                />
              </div>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="animate-slide-up" style={{ animationDelay: '400ms' }}>
          <Button
            type="submit"
            disabled={!query.trim()}
            loading={isLoading}
            fullWidth
            icon={Users}
            className="btn-primary"
          >
            {isLoading ? 'Researching...' : 'Ask the Four Hosts'}
          </Button>
        </div>
      </div>
    </form>
  )
}
