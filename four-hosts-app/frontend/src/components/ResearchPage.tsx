import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle } from 'lucide-react'
import { ToggleSwitch } from './ui/ToggleSwitch'
import { Alert } from './ui/Alert'
import { ResearchFormEnhanced } from './ResearchFormEnhanced'
import { ResearchFormIdeaBrowser } from './ResearchFormIdeaBrowser'
import { ResearchProgress } from './ResearchProgress'
import { ResearchProgressIdeaBrowser } from './ResearchProgressIdeaBrowser'
import { ResultsDisplayEnhanced } from './ResultsDisplayEnhanced'
import { ResultsDisplayIdeaBrowser } from './ResultsDisplayIdeaBrowser'
import ParadigmDisplay from './ParadigmDisplay'
import api from '../services/api'
import type { ResearchResult, ParadigmClassification, ResearchOptions } from '../types'

export const ResearchPage = () => {
  const navigate = useNavigate()
  const [isLoading, setIsLoading] = useState(false)
  const [paradigmClassification, setParadigmClassification] = useState<ParadigmClassification | null>(null)
  const [results, setResults] = useState<ResearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [currentResearchId, setCurrentResearchId] = useState<string | null>(null)
  const [showProgress, setShowProgress] = useState(false)
  const [useIdeaBrowser, setUseIdeaBrowser] = useState(() => {
    try {
      const saved = localStorage.getItem('useIdeaBrowser')
      return saved ? JSON.parse(saved) : false
    } catch {
      return false
    }
  })
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [])

  const handleSubmit = async (query: string, options: ResearchOptions) => {
    setIsLoading(true)
    setError(null)
    setResults(null)
    setShowProgress(true)

    try {
      // Submit research query
      let data
      if (options.depth === 'deep_research') {
        // Use deep research endpoint for o3-deep-research model
        // TODO: Add UI for search context size and user location configuration
        data = await api.submitDeepResearch(
          query, 
          options.paradigm_override || undefined,
          'medium', // Default search context size
          undefined // User location - could detect from browser
        )
      } else {
        // Use standard research endpoint
        data = await api.submitResearch(query, options)
      }
      setCurrentResearchId(data.research_id)

      // Poll for results
      let retries = 0
      const maxRetries = 60

      pollIntervalRef.current = setInterval(async () => {
        try {
          const resultsData = await api.getResearchResults(data.research_id)

          // Handle staged response (status present but not final)
          const stagedStatuses = ['queued','processing','in_progress']
          if ((resultsData as any)?.status && stagedStatuses.includes((resultsData as any).status)) {
            if (retries >= maxRetries) {
              setError('Research timeout - please try again')
              setIsLoading(false)
              setShowProgress(false)
              clearInterval(pollIntervalRef.current!)
            }
            return
          }

          // Handle terminal staged failures
          if ((resultsData as any)?.status === 'failed' || (resultsData as any)?.status === 'cancelled') {
            setError(`Research ${(resultsData as any).status}: ${(resultsData as any).message || 'Please try again'}`)
            setIsLoading(false)
            setShowProgress(false)
            clearInterval(pollIntervalRef.current!)
            return
          }

          // Final result path
          setResults(resultsData)

          // Extract paradigm classification from results
          if ((resultsData as any).paradigm_analysis && (resultsData as any).paradigm_analysis.primary) {
            const primary = (resultsData as any).paradigm_analysis.primary
            const secondary = (resultsData as any).paradigm_analysis.secondary
            const distribution: Record<string, number> = { [primary.paradigm]: primary.confidence }
            
            if (secondary) {
              distribution[secondary.paradigm] = secondary.confidence
            }

            const allParadigms = ['dolores', 'teddy', 'bernard', 'maeve']
            allParadigms.forEach(p => {
              if (!distribution[p]) distribution[p] = 0
            })

            setParadigmClassification({
              primary: primary.paradigm,
              secondary: secondary?.paradigm || null,
              distribution,
              confidence: primary.confidence,
              explanation: {
                [primary.paradigm]: primary.approach
              }
            })
          }
          setIsLoading(false)
          setShowProgress(false)
          clearInterval(pollIntervalRef.current!)
        } catch {
          if (retries >= maxRetries) {
            setError('Research timeout - please try again')
            setIsLoading(false)
            setShowProgress(false)
            clearInterval(pollIntervalRef.current!)
          }
        }
        retries++
      }, 2000)

    } catch (err) {
      if (err instanceof Error && err.message.includes('No refresh token available')) {
        // Authentication error - redirect to login
        navigate('/login')
      } else {
        setError(err instanceof Error ? err.message : 'An error occurred')
        setIsLoading(false)
        setShowProgress(false)
      }
    }
  }

  return (
    <div id="main-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fade-in">
      <div className="mb-8 text-center animate-slide-down">
        <h1 className="text-responsive-3xl sm:text-3xl font-bold text-text mb-2">
          Discover Insights Through Four Perspectives
        </h1>
        <p className="text-responsive-base sm:text-base text-text-muted">
          Let our AI hosts guide your research with their unique paradigms
        </p>
      </div>

      {/* Toggle for IdeaBrowser mode */}
      <div className="mb-4 flex items-center justify-center gap-3">
        <label className="text-sm text-text-muted">Standard View</label>
        <ToggleSwitch
          checked={useIdeaBrowser}
          onChange={(checked) => {
            setUseIdeaBrowser(checked)
            localStorage.setItem('useIdeaBrowser', JSON.stringify(checked))
          }}
          size="sm"
        />
        <label className="text-sm text-text-muted">IdeaBrowser View</label>
      </div>

      {useIdeaBrowser ? (
        <ResearchFormIdeaBrowser onSubmit={handleSubmit} isLoading={isLoading} />
      ) : (
        <ResearchFormEnhanced onSubmit={handleSubmit} isLoading={isLoading} />
      )}

      {error && (
        <Alert variant="error" title="Research Error" className="mt-4">
          {error}
        </Alert>
      )}

      {paradigmClassification && (
        <div className="animate-scale-in">
          <ParadigmDisplay classification={paradigmClassification} />
        </div>
      )}

      {showProgress && currentResearchId && (
        <div className="animate-slide-up">
          {useIdeaBrowser ? (
            <ResearchProgressIdeaBrowser
              researchId={currentResearchId}
              onComplete={() => setShowProgress(false)}
              onCancel={() => {
                setShowProgress(false)
                setCurrentResearchId(null)
              }}
            />
          ) : (
            <ResearchProgress
              researchId={currentResearchId}
              onComplete={() => setShowProgress(false)}
              onCancel={() => {
                setShowProgress(false)
                setCurrentResearchId(null)
              }}
            />
          )}
        </div>
      )}

      {results && !showProgress && (
        <div className="animate-fade-in">
          {useIdeaBrowser ? (
            <ResultsDisplayIdeaBrowser results={results} />
          ) : (
            <ResultsDisplayEnhanced results={results} />
          )}
        </div>
      )}
    </div>
  )
}