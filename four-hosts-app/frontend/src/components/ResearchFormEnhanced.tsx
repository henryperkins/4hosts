import React, { useReducer, useEffect, useCallback, useMemo } from 'react'
import { FiSettings, FiUsers } from 'react-icons/fi'
import { useAuth } from '../hooks/useAuth'
import type { ResearchOptions, Paradigm } from '../types'
import { Button } from './ui/Button'
import { InputField } from './ui/InputField'

interface ResearchFormState {
  query: string
  showAdvanced: boolean
  error: string
  paradigm: string
  depth: 'quick' | 'standard' | 'deep' | 'deep_research'
  options: ResearchOptions
  comprehensive: boolean
}

type FormAction =
  | { type: 'SET_QUERY'; payload: string }
  | { type: 'SET_ERROR'; payload: string }
  | { type: 'TOGGLE_ADVANCED' }
  | { type: 'SET_PARADIGM'; payload: string }
  | { type: 'SET_DEPTH'; payload: ResearchFormState['depth'] }
  | { type: 'UPDATE_OPTIONS'; payload: Partial<ResearchOptions> }
  | { type: 'RESET_FORM' }
  | { type: 'INIT_FROM_PREFERENCES'; payload: { depth: ResearchFormState['depth']; enable_real_search: boolean } }
  | { type: 'SET_COMPREHENSIVE'; payload: boolean }

function formReducer(state: ResearchFormState, action: FormAction): ResearchFormState {
  switch (action.type) {
    case 'SET_QUERY':
      return { ...state, query: action.payload, error: '' }
    case 'SET_ERROR':
      return { ...state, error: action.payload }
    case 'TOGGLE_ADVANCED':
      return { ...state, showAdvanced: !state.showAdvanced }
    case 'SET_PARADIGM':
      return { ...state, paradigm: action.payload }
    case 'SET_DEPTH':
      return { ...state, depth: action.payload, options: { ...state.options, depth: action.payload } }
    case 'UPDATE_OPTIONS':
      return { ...state, options: { ...state.options, ...action.payload } }
    case 'RESET_FORM':
      return { ...state, query: '', error: '', paradigm: 'auto' }
    case 'SET_COMPREHENSIVE':
      return { ...state, comprehensive: action.payload }
    case 'INIT_FROM_PREFERENCES':
      return {
        ...state,
        depth: action.payload.depth,
        options: {
          ...state.options,
          depth: action.payload.depth,
          enable_real_search: action.payload.enable_real_search
        }
      }
    default:
      return state
  }
}

interface ResearchFormEnhancedProps {
  onSubmit: (query: string, options: ResearchOptions) => void
  isLoading: boolean
  onQueryChange?: (query: string) => void
}

// Commented out - not currently used
// const ParadigmOption = React.memo<{
//   value: string
//   label: string
//   icon: string
//   description: string
//   colorClass: string
//   isSelected: boolean
//   onSelect: (value: string) => void
// }>(({ value, label, icon, description, colorClass, isSelected, onSelect }) => (
//   <button
//     type="button"
//     onClick={() => onSelect(value)}
//     className={`group w-full text-left px-4 py-3 rounded-lg border transition-all ${
//       isSelected 
//         ? 'border-primary bg-primary/10 shadow-sm' 
//         : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
//     }`}
//   >
//     <div className="flex items-center gap-3">
//       <span className="text-2xl">{icon}</span>
//       <div className="flex-1">
//         <div className={`font-medium ${colorClass}`}>{label}</div>
//         <div className="text-xs text-gray-500 dark:text-gray-400">{description}</div>
//       </div>
//     </div>
//   </button>
// ))

export const ResearchFormEnhanced: React.FC<ResearchFormEnhancedProps> = ({ onSubmit, isLoading, onQueryChange }) => {
  const { user } = useAuth()
  
  const initialState: ResearchFormState = {
    query: '',
    showAdvanced: false,
    error: '',
    paradigm: 'auto',
    depth: 'standard',
    options: {
      depth: 'standard',
      include_secondary: true,
      max_sources: 50,
      enable_real_search: true,
      language: 'en',
      region: 'us',
      enable_ai_classification: true
    },
    comprehensive: false
  }

  const [state, dispatch] = useReducer(formReducer, initialState)

  // Initialize from user preferences
  useEffect(() => {
    if (user?.preferences) {
      dispatch({
        type: 'INIT_FROM_PREFERENCES',
        payload: {
          depth: user.preferences.default_depth || 'standard',
          enable_real_search: user.preferences.enable_real_search !== false
        }
      })
    }
  }, [user])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    
    const trimmedQuery = state.query.trim()
    if (!trimmedQuery) {
      dispatch({ type: 'SET_ERROR', payload: 'Please enter a research query' })
      return
    }

    if (trimmedQuery.length < 10) {
      dispatch({ type: 'SET_ERROR', payload: 'Query must be at least 10 characters long' })
      return
    }

    // Include paradigm override if a specific paradigm is selected
    const optionsToSend: ResearchOptions = {
      ...state.options,
      paradigm_override: state.paradigm !== 'auto' ? (state.paradigm as Paradigm) : undefined,
    }

    onSubmit(trimmedQuery, optionsToSend)
  }, [state.query, state.options, state.paradigm, onSubmit])

  const paradigmOptions = useMemo(() => [
    { value: 'auto', label: 'Auto', icon: '🔮', description: 'Let AI choose', colorClass: 'text-purple-600 dark:text-purple-400' },
    { value: 'dolores', label: 'Dolores', icon: '🛡️', description: 'Truth & Justice', colorClass: 'text-paradigm-dolores' },
    { value: 'bernard', label: 'Bernard', icon: '🧠', description: 'Analysis & Logic', colorClass: 'text-paradigm-bernard' },
    { value: 'teddy', label: 'Teddy', icon: '❤️', description: 'Care & Support', colorClass: 'text-paradigm-teddy' },
    { value: 'maeve', label: 'Maeve', icon: '📈', description: 'Strategy & Power', colorClass: 'text-paradigm-maeve' }
  ], [])

  // Currently not used - for future deep research feature
  // const _depthOptions = useMemo(() => [
  //   { value: 'quick', label: 'Quick', description: 'Fast overview (5-10 sources)' },
  //   { value: 'standard', label: 'Standard', description: 'Balanced research (10-25 sources)' },
  //   { value: 'deep', label: 'Deep', description: 'Comprehensive analysis (25-50 sources)' },
  //   { value: 'deep_research', label: 'Deep Research', description: 'Exhaustive investigation (50+ sources)', pro: true }
  // ], [])

  const handleQueryChange = useCallback((e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const value = e.target.value
    dispatch({ type: 'SET_QUERY', payload: value })
    // Notify parent for live classification preview (debounced upstream)
    if (onQueryChange) onQueryChange(value)
  }, [onQueryChange])

  const handleParadigmSelect = useCallback((value: string) => {
    dispatch({ type: 'SET_PARADIGM', payload: value })
    const override = value !== 'auto' ? (value as Paradigm) : undefined
    dispatch({ type: 'UPDATE_OPTIONS', payload: { paradigm_override: override } })
  }, [])

  // Currently not used - for future depth selection
  // const _handleDepthChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
  //   dispatch({ type: 'SET_DEPTH', payload: e.target.value as ResearchFormState['depth'] })
  // }, [])

  const toggleAdvanced = useCallback(() => {
    dispatch({ type: 'TOGGLE_ADVANCED' })
  }, [])

  // Comprehensive Mode toggle — boosts coverage and analysis depth where allowed
  const canUseDeep = true // Deep research is now available to all users
  const handleComprehensiveToggle = useCallback(() => {
    const next = !state.comprehensive
    // When enabling, raise sensible limits; when disabling, restore defaults
    if (next) {
      const boosted: Partial<ResearchOptions> = {
        enable_real_search: true,
        enable_ai_classification: true,
        // Keep under server-side higher-cost threshold while still broad
        max_sources: 100,
        search_context_size: 'large'
      }
      dispatch({ type: 'UPDATE_OPTIONS', payload: boosted })
      if (canUseDeep) {
        dispatch({ type: 'SET_DEPTH', payload: 'deep' })
      }
    } else {
      dispatch({ type: 'UPDATE_OPTIONS', payload: { max_sources: 50, search_context_size: 'medium' } })
      dispatch({ type: 'SET_DEPTH', payload: state.depth === 'deep' ? 'standard' : state.depth })
    }
    dispatch({ type: 'SET_COMPREHENSIVE', payload: next })
  }, [state.comprehensive, canUseDeep, state.depth])

  // Currently not used - for future access control
  // const _canAccessDeepResearch = user?.role && ['pro', 'enterprise', 'admin'].includes(user.role)

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
            value={state.query}
            onChange={handleQueryChange}
            placeholder="What would you like to research today?"
            rows={3}
            disabled={isLoading}
            status={state.error ? 'error' : undefined}
            errorMessage={state.error}
            className="border-2"
            required
            minLength={10}
            aria-describedby={state.error ? "query-error" : undefined}
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
                onClick={() => handleParadigmSelect(option.value)}
                disabled={isLoading}
                className={`relative p-3 rounded-lg border transition-colors
                  ${state.paradigm === option.value
                    ? 'border-primary bg-primary/10 shadow-md'
                    : 'border-border hover:border-text-subtle bg-surface'
                  }
                  ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:shadow-lg'}
                  group`}
                aria-pressed={state.paradigm === option.value}
                aria-label={`${option.label}: ${option.description}`}
              >
                <div className="text-center">
                  <div className={`text-2xl mb-1 ${
                    state.paradigm === option.value ? 'animate-fade-in' : ''
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
                {state.paradigm === option.value && (
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
            {(() => {
              // Deep research is now available to all users (variable unused; keep comment for clarity)
              type DepthValue = ResearchFormState['depth']
              interface DepthOption { value: DepthValue; label: string; description: string; badge?: string }
              const depthOptions: DepthOption[] = [
                { value: 'quick', label: 'Quick', description: 'Fast overview' },
                { value: 'standard', label: 'Standard', description: 'Balanced analysis' },
                { value: 'deep', label: 'Deep', description: 'Comprehensive research' },
                { value: 'deep_research', label: 'Deep AI', description: 'Advanced analysis' },
              ]
              return depthOptions.map((option) => {
                const disabled = isLoading
                const isSelected = state.depth === option.value
                return (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => {
                      console.log('Depth button clicked:', option.value)
                      dispatch({ type: 'SET_DEPTH', payload: option.value })
                    }}
                    disabled={disabled}
                    className={`flex-1 p-3 rounded-lg border-2 transition-all duration-200
                  ${isSelected
                    ? 'border-primary bg-primary/20 shadow-lg scale-105 ring-2 ring-primary/50'
                    : 'border-border hover:border-primary/50 bg-surface hover:bg-surface-hover'
                  }
                  ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:shadow-md'}`}
                  >
                    <div className="relative">
                      <div className={`font-medium ${isSelected ? 'text-primary' : 'text-text'}`}>
                        {option.label}
                        {isSelected && <span className="ml-2">✓</span>}
                      </div>
                      <div className={`text-xs mt-1 ${isSelected ? 'text-primary/80' : 'text-text-muted'}`}>
                        {option.description}
                      </div>
                      {option.badge && (
                        <span className="absolute -top-2 -right-2 px-1.5 py-0.5 text-xs font-bold bg-gradient-to-r from-orange-500 to-yellow-500 text-white rounded">
                          {option.badge}
                        </span>
                      )}
                    </div>
                  </button>
                )
              })
            })()}
          </div>
        </div>

        {/* Advanced Options Toggle */}
        <div className="animate-slide-up" style={{ animationDelay: '300ms' }}>
          <button
            type="button"
            onClick={toggleAdvanced}
            className="flex items-center gap-2 text-sm text-primary hover:text-primary/80 transition-colors"
          >
            <FiSettings className={`h-4 w-4 transition-transform ${state.showAdvanced ? 'rotate-90' : ''}`} />
            {state.showAdvanced ? 'Hide' : 'Show'} Advanced Options
          </button>
        </div>

        {/* Advanced Options */}
        {state.showAdvanced && (
          <div className="animate-slide-up space-y-4 p-4 bg-surface-subtle rounded-lg">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="flex items-center cursor-pointer group">
                  <input
                    type="checkbox"
                    checked={state.options.enable_ai_classification}
                    onChange={(e) => dispatch({ type: 'UPDATE_OPTIONS', payload: { enable_ai_classification: e.target.checked } })}
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
                  value={state.options.max_sources || 100}
                  onChange={(e) => dispatch({ type: 'UPDATE_OPTIONS', payload: { max_sources: parseInt(e.target.value) } })}
                  className="input text-sm"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="flex items-center cursor-pointer group">
                  <input
                    type="checkbox"
                    checked={state.comprehensive}
                    onChange={handleComprehensiveToggle}
                    className="h-4 w-4 text-primary focus:ring-primary border-border rounded"
                  />
                  <span className="ml-2 text-sm text-text">
                    Comprehensive Mode
                    <span className="ml-2 text-xs text-text-muted">broader coverage, larger context</span>
                  </span>
                </label>
              </div>
              <div>
                <label className="block text-sm text-text mb-1">Search context size</label>
                <select
                  className="input text-sm"
                  value={state.options.search_context_size || 'medium'}
                  onChange={(e) => dispatch({ type: 'UPDATE_OPTIONS', payload: { search_context_size: (e.target.value as 'small' | 'medium' | 'large') } })}
                >
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="animate-slide-up" style={{ animationDelay: '400ms' }}>
          <Button
            type="submit"
            disabled={!state.query.trim()}
            loading={isLoading}
            fullWidth
            icon={FiUsers}
            className="btn-primary"
          >
            {isLoading ? 'Researching...' : 'Ask the Four Hosts'}
          </Button>
        </div>
      </div>
    </form>
  )
}
