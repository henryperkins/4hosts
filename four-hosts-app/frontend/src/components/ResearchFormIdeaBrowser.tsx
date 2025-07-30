import React, { useState, useEffect } from 'react'
import { Settings2, Users, BarChart3, Zap, Target, Brain } from 'lucide-react'
import { useAuth } from '../hooks/useAuth'
import type { ResearchOptions, Paradigm } from '../types'
import { Button } from './ui/Button'
import { InputField } from './ui/InputField'
import { Badge } from './ui/Badge'
import { cn } from '../utils/cn'

interface ResearchFormIdeaBrowserProps {
  onSubmit: (query: string, options: ResearchOptions) => void
  isLoading: boolean
}

interface ResearchDimensions {
  urgency: number
  complexity: number
  businessImpact: number
  paradigmConfidence: number
}

interface ParadigmMetrics {
  score: number
  trending: string
  activeResearchers: string
  recentSuccessRate: number
}

const paradigmMetrics: Record<Paradigm | 'auto', ParadigmMetrics> = {
  auto: { score: 9.5, trending: '+15%', activeResearchers: '10.4K', recentSuccessRate: 94 },
  dolores: { score: 8.5, trending: '+12%', activeResearchers: '2.3K', recentSuccessRate: 87 },
  bernard: { score: 9.2, trending: '+5%', activeResearchers: '4.1K', recentSuccessRate: 92 },
  teddy: { score: 7.8, trending: '+18%', activeResearchers: '1.8K', recentSuccessRate: 85 },
  maeve: { score: 8.9, trending: '+23%', activeResearchers: '3.2K', recentSuccessRate: 90 }
}

const getMetricColor = (score: number): string => {
  if (score >= 8) return 'text-green-600 dark:text-green-400'
  if (score >= 6) return 'text-blue-600 dark:text-blue-400'
  if (score >= 4) return 'text-amber-600 dark:text-amber-400'
  return 'text-red-600 dark:text-red-400'
}

const getTrendColor = (trend: string): string => {
  return trend.startsWith('+') ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
}

export const ResearchFormIdeaBrowser: React.FC<ResearchFormIdeaBrowserProps> = ({ onSubmit, isLoading }) => {
  const { user } = useAuth()
  const [query, setQuery] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [error, setError] = useState('')
  const [paradigm, setParadigm] = useState<Paradigm | 'auto'>('auto')
  const [depth, setDepth] = useState<'quick' | 'standard' | 'deep' | 'deep_research'>(user?.preferences?.default_depth ?? 'standard')
  
  const [dimensions, setDimensions] = useState<ResearchDimensions>({
    urgency: 5,
    complexity: 5,
    businessImpact: 5,
    paradigmConfidence: 75
  })

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

    const finalOptions = {
      ...options,
      depth,
      paradigm_override: paradigm !== 'auto' ? paradigm : undefined
    }

    onSubmit(trimmedQuery, finalOptions)
  }

  const paradigmOptions = [
    { value: 'auto', label: 'Auto', icon: '🔮', description: 'AI-Optimized', colorClass: 'text-purple-600 dark:text-purple-400' },
    { value: 'dolores', label: 'Dolores', icon: '🛡️', description: 'Truth & Justice', colorClass: 'text-paradigm-dolores' },
    { value: 'bernard', label: 'Bernard', icon: '🧠', description: 'Analysis & Logic', colorClass: 'text-paradigm-bernard' },
    { value: 'teddy', label: 'Teddy', icon: '❤️', description: 'Care & Support', colorClass: 'text-paradigm-teddy' },
    { value: 'maeve', label: 'Maeve', icon: '📈', description: 'Strategy & Power', colorClass: 'text-paradigm-maeve' }
  ] as const

  // Calculate estimated research quality based on dimensions
  const calculateQualityScore = () => {
    const urgencyWeight = dimensions.urgency * 0.2
    const complexityWeight = dimensions.complexity * 0.3
    const impactWeight = dimensions.businessImpact * 0.3
    const confidenceWeight = (dimensions.paradigmConfidence / 100) * 10 * 0.2
    return (urgencyWeight + complexityWeight + impactWeight + confidenceWeight).toFixed(1)
  }

  // Calculate recommended depth based on dimensions
  const getRecommendedDepth = () => {
    const score = parseFloat(calculateQualityScore())
    if (score >= 8) return 'deep_research'
    if (score >= 6) return 'deep'
    if (score >= 4) return 'standard'
    return 'quick'
  }

  // Calculate estimated cost based on depth and options
  const getEstimatedCost = () => {
    const baseCosts = { quick: 0.02, standard: 0.05, deep: 0.12, deep_research: 0.25 }
    const baseCost = baseCosts[depth]
    const sourcesMultiplier = (options.max_sources || 50) / 50
    return (baseCost * sourcesMultiplier).toFixed(3)
  }

  return (
    <form onSubmit={handleSubmit} className="card-hover animate-fade-in">
      <div className="space-y-6">
        {/* Metrics Dashboard */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-4 animate-slide-up">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Research Quality Metrics
            </h3>
            <Badge variant="default" size="sm">
              Score: {calculateQualityScore()}/10
            </Badge>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <label className="text-xs text-gray-600 dark:text-gray-400">Urgency</label>
              <input
                type="range"
                min="0"
                max="10"
                value={dimensions.urgency}
                onChange={(e) => setDimensions({ ...dimensions, urgency: parseInt(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
              />
              <div className="flex justify-between text-xs mt-1">
                <span className={getMetricColor(dimensions.urgency)}>{dimensions.urgency}</span>
                <span className="text-gray-500">Impact</span>
              </div>
            </div>
            
            <div>
              <label className="text-xs text-gray-600 dark:text-gray-400">Complexity</label>
              <input
                type="range"
                min="0"
                max="10"
                value={dimensions.complexity}
                onChange={(e) => setDimensions({ ...dimensions, complexity: parseInt(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
              />
              <div className="flex justify-between text-xs mt-1">
                <span className={getMetricColor(dimensions.complexity)}>{dimensions.complexity}</span>
                <span className="text-gray-500">Depth</span>
              </div>
            </div>
            
            <div>
              <label className="text-xs text-gray-600 dark:text-gray-400">Business Impact</label>
              <input
                type="range"
                min="0"
                max="10"
                value={dimensions.businessImpact}
                onChange={(e) => setDimensions({ ...dimensions, businessImpact: parseInt(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
              />
              <div className="flex justify-between text-xs mt-1">
                <span className={getMetricColor(dimensions.businessImpact)}>{dimensions.businessImpact}</span>
                <span className="text-gray-500">ROI</span>
              </div>
            </div>
            
            <div>
              <label className="text-xs text-gray-600 dark:text-gray-400">Confidence</label>
              <input
                type="range"
                min="0"
                max="100"
                value={dimensions.paradigmConfidence}
                onChange={(e) => setDimensions({ ...dimensions, paradigmConfidence: parseInt(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
              />
              <div className="flex justify-between text-xs mt-1">
                <span className={getMetricColor(dimensions.paradigmConfidence / 10)}>{dimensions.paradigmConfidence}%</span>
                <span className="text-gray-500">Clarity</span>
              </div>
            </div>
          </div>
          
          {/* Recommendations based on metrics */}
          <div className="mt-3 p-2 bg-white/50 dark:bg-gray-800/50 rounded text-xs">
            <div className="flex items-center gap-2">
              <Brain className="h-3 w-3 text-blue-600 dark:text-blue-400" />
              <span className="text-gray-700 dark:text-gray-300">
                Recommended depth: <strong className="text-blue-600 dark:text-blue-400">{getRecommendedDepth()}</strong>
                {' '}• Estimated cost: <strong>${getEstimatedCost()}</strong>
              </span>
            </div>
          </div>
        </div>

        {/* Query Input */}
        <div className="animate-slide-up" style={{ animationDelay: '100ms' }}>
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

        {/* Enhanced Paradigm Selection with Metrics */}
        <div className="animate-slide-up" style={{ animationDelay: '200ms' }}>
          <div className="block text-sm font-medium text-text mb-3" role="group" aria-label="Paradigm Selection">
            Paradigm Selection
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {paradigmOptions.map((option) => {
              const metrics = paradigmMetrics[option.value]
              return (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setParadigm(option.value)}
                  disabled={isLoading}
                  className={cn(
                    "relative p-3 rounded-lg border transition-all duration-200 overflow-hidden",
                    paradigm === option.value
                      ? 'border-primary bg-primary/10 shadow-md scale-105'
                      : 'border-border hover:border-text-subtle bg-surface hover:shadow-lg',
                    isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
                    "group"
                  )}
                  aria-pressed={paradigm === option.value}
                  aria-label={`${option.label}: ${option.description}`}
                >
                  {/* Metrics overlay */}
                  <div className="absolute top-1 right-1 flex flex-col items-end gap-0.5">
                    <Badge variant="default" size="sm" className={cn("text-xs", getMetricColor(metrics.score))}>
                      {metrics.score}
                    </Badge>
                    <span className={cn("text-xs font-semibold", getTrendColor(metrics.trending))}>
                      {metrics.trending}
                    </span>
                  </div>
                  
                  <div className="text-center">
                    <div className={cn(
                      "text-2xl mb-1 transition-transform",
                      paradigm === option.value ? 'animate-fade-in scale-110' : ''
                    )}>
                      {option.icon}
                    </div>
                    <div className={cn("font-medium text-sm", option.colorClass || 'text-text')}>
                      {option.label}
                    </div>
                    <div className="text-xs text-text-muted mt-1">
                      {option.description}
                    </div>
                    
                    {/* Active researchers and success rate */}
                    <div className="mt-2 space-y-1">
                      <div className="flex items-center justify-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                        <Users className="h-3 w-3" />
                        <span>{metrics.activeResearchers}</span>
                      </div>
                      <div className="flex items-center justify-center gap-1 text-xs">
                        <Target className="h-3 w-3" />
                        <span className={getMetricColor(metrics.recentSuccessRate / 10)}>
                          {metrics.recentSuccessRate}% success
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  {paradigm === option.value && (
                    <div className="absolute inset-0 rounded-lg bg-primary/5 pointer-events-none" />
                  )}
                </button>
              )
            })}
          </div>
        </div>

        {/* Enhanced Depth Selection with Cost Calculator */}
        <div className="animate-slide-up" style={{ animationDelay: '300ms' }}>
          <label className="block text-sm font-medium text-text mb-3">
            Research Depth
          </label>
          <div className="flex gap-3">
            {[
              { value: 'quick', label: 'Quick', description: 'Fast overview', time: '~30s', cost: '$0.02' },
              { value: 'standard', label: 'Standard', description: 'Balanced analysis', time: '~2min', cost: '$0.05' },
              { value: 'deep', label: 'Deep', description: 'Comprehensive research', time: '~5min', cost: '$0.12' },
              { value: 'deep_research', label: 'Deep AI', description: 'o3 model analysis', badge: 'PRO', time: '~10min', cost: '$0.25' },
            ].map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setDepth(option.value as 'quick' | 'standard' | 'deep' | 'deep_research')}
                disabled={isLoading || (option.value === 'deep_research' && user?.role === 'free')}
                className={cn(
                  "flex-1 p-3 rounded-lg border transition-all duration-200",
                  depth === option.value
                    ? 'border-primary bg-primary/10 shadow-md'
                    : 'border-border hover:border-text-subtle bg-surface',
                  isLoading || (option.value === 'deep_research' && user?.role === 'free') 
                    ? 'opacity-50 cursor-not-allowed' 
                    : 'cursor-pointer hover:shadow-lg',
                  depth === option.value && option.value === getRecommendedDepth() && 'ring-2 ring-green-500 ring-offset-2'
                )}
              >
                <div className="relative">
                  <div className="font-medium text-text">{option.label}</div>
                  <div className="text-xs text-text-muted mt-1">{option.description}</div>
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-xs text-gray-500">{option.time}</span>
                    <span className="text-xs font-semibold text-green-600 dark:text-green-400">{option.cost}</span>
                  </div>
                  {option.badge && (
                    <span className="absolute -top-2 -right-2 px-1.5 py-0.5 text-xs font-bold bg-gradient-to-r from-orange-500 to-yellow-500 text-white rounded">
                      {option.badge}
                    </span>
                  )}
                  {depth === option.value && option.value === getRecommendedDepth() && (
                    <Zap className="absolute -top-2 -left-2 h-4 w-4 text-green-500 animate-pulse" />
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Advanced Options Toggle */}
        <div className="animate-slide-up" style={{ animationDelay: '400ms' }}>
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
                  Max sources ({options.max_sources})
                </label>
                <input
                  type="range"
                  min="10"
                  max="200"
                  value={options.max_sources || 100}
                  onChange={(e) => setOptions({ ...options, max_sources: parseInt(e.target.value) })}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
              </div>
            </div>
          </div>
        )}

        {/* Submit Button with Metrics Summary */}
        <div className="animate-slide-up" style={{ animationDelay: '500ms' }}>
          <div className="mb-2 flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
            <span>Quality Score: <strong className={getMetricColor(parseFloat(calculateQualityScore()))}>{calculateQualityScore()}/10</strong></span>
            <span>Est. Time: <strong>~{depth === 'quick' ? '30s' : depth === 'standard' ? '2min' : depth === 'deep' ? '5min' : '10min'}</strong></span>
            <span>Est. Cost: <strong className="text-green-600 dark:text-green-400">${getEstimatedCost()}</strong></span>
          </div>
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