import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
// AlertCircle not currently used
import { Alert } from './ui/Alert'
import { ResearchFormEnhanced } from './ResearchFormEnhanced'
import { ResearchProgress } from './ResearchProgress'
import { ResultsDisplayEnhanced } from './ResultsDisplayEnhanced'
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
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [])

  // Type helpers for staged/terminal statuses from results endpoint
  type StagedStatus = 'queued' | 'processing' | 'in_progress' | 'failed' | 'cancelled'
  const getStatus = (obj: unknown): StagedStatus | string | undefined => {
    if (obj && typeof obj === 'object') {
      const rec = obj as Record<string, unknown>
      return typeof rec.status === 'string' ? rec.status : undefined
    }
    return undefined
  }
  const getMessage = (obj: unknown): string | undefined => {
    if (obj && typeof obj === 'object') {
      const rec = obj as Record<string, unknown>
      return typeof rec.message === 'string' ? rec.message : undefined
    }
    return undefined
  }

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
          // Increment retries for each poll tick
          retries++

          const resultsData = await api.getResearchResults(data.research_id)

          // Handle staged response (status present but not final)
          const stagedStatuses: StagedStatus[] = ['queued','processing','in_progress']
          const status = getStatus(resultsData)
          if (status && stagedStatuses.includes(status as StagedStatus)) {
            if (retries >= maxRetries) {
              setError('Research timeout - please try again')
              setIsLoading(false)
              setShowProgress(false)
              clearInterval(pollIntervalRef.current!)
            }
            return
          }

          // Handle terminal staged failures
          if (status === 'failed' || status === 'cancelled') {
            const message = getMessage(resultsData) || 'Please try again'
            setError(`Research ${status}: ${message}`)
            setIsLoading(false)
            setShowProgress(false)
            clearInterval(pollIntervalRef.current!)
            return
          }

          // Final result path
          setResults(resultsData)

          // Extract paradigm classification from results
          if (resultsData.paradigm_analysis && resultsData.paradigm_analysis.primary) {
            const primary = resultsData.paradigm_analysis.primary
            const secondary = resultsData.paradigm_analysis.secondary
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

      <ResearchFormEnhanced onSubmit={handleSubmit} isLoading={isLoading} />

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
          <ResearchProgress
            researchId={currentResearchId}
            onComplete={() => setShowProgress(false)}
            onCancel={() => {
              setShowProgress(false)
              setCurrentResearchId(null)
            }}
          />
        </div>
      )}

      {results && !showProgress && (
        <div className="animate-fade-in">
          <ResultsDisplayEnhanced results={results} />
        </div>
      )}
    </div>
  )
}
