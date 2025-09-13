import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
// AlertCircle not currently used
import { Alert } from './ui/Alert'
import { ResearchFormEnhanced } from './ResearchFormEnhanced'
import { ResearchProgress } from './ResearchProgress'
import { ResultsDisplayEnhanced } from './ResultsDisplayEnhanced'
import ParadigmDisplay from './ParadigmDisplay'
import { ClassificationFeedback } from './feedback/ClassificationFeedback'
import api from '../services/api'
import type { ResearchResult, ParadigmClassification, ResearchOptions } from '../types'

export const ResearchPage = () => {
  const navigate = useNavigate()
  const [isLoading, setIsLoading] = useState(false)
  const [paradigmClassification, setParadigmClassification] = useState<ParadigmClassification | null>(null)
  const [liveQuery, setLiveQuery] = useState<string>('')
  const [liveClassification, setLiveClassification] = useState<ParadigmClassification | null>(null)
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

  // Live classification preview with debounce
  useEffect(() => {
    // Reset preview when query cleared or while a research is running
    if (!liveQuery || isLoading) {
      setLiveClassification(null)
      return
    }

    const trimmed = liveQuery.trim()
    if (trimmed.length < 10) {
      setLiveClassification(null)
      return
    }

    const handle = setTimeout(async () => {
      try {
        const data = await api.classifyQuery(trimmed)
        // Normalize to ParadigmClassification shape
        if (data && data.primary) {
          setLiveClassification({
            primary: data.primary,
            secondary: data.secondary ?? null,
            distribution: data.distribution || {},
            confidence: typeof data.confidence === 'number' ? data.confidence : 0,
            explanation: data.explanation || {},
            signals: (data as any).signals || undefined
          })
        }
      } catch {
        // Silent fail for preview
      }
    }, 400)

    return () => clearTimeout(handle)
  }, [liveQuery, isLoading])

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
      const data = await api.submitResearch(query, options)
      setCurrentResearchId(data.research_id)

      // Poll for results (extend to avoid premature timeout)
      let retries = 0
      const maxRetries = 600 // 20 minutes at 2s interval

      pollIntervalRef.current = setInterval(async () => {
        try {
          // Increment retries for each poll tick
          retries++

          const resultsData = await api.getResearchResults(data.research_id)

          // Handle staged response (status present but not final)
          const stagedStatuses: StagedStatus[] = ['queued','processing','in_progress']
          const status = getStatus(resultsData)
          if (status && stagedStatuses.includes(status as StagedStatus)) {
            // Still processing; keep polling (WebSocket shows progress). Do not auto-cancel.
            if (retries >= maxRetries) {
              // Convert to a soft timeout: keep progress visible, stop polling, let WS finish.
              setError('Research is taking longer than usual but is still running. You can keep this page open or come back later in History.')
              setIsLoading(false)
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
            setError('Research is taking longer than usual but is still running. You can keep this page open or come back later in History.')
            setIsLoading(false)
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

      <ResearchFormEnhanced onSubmit={handleSubmit} isLoading={isLoading} onQueryChange={setLiveQuery} />

      {error && (
        <Alert variant="error" title="Research Error" className="mt-4">
          {error}
        </Alert>
      )}

      {/* Live preview before submission */}
      {!results && !showProgress && liveClassification && (
        <div className="animate-scale-in">
          <div className="mb-2 text-xs text-text-muted">Live Paradigm Preview</div>
          <ParadigmDisplay classification={liveClassification} />
          {/* Optional classification feedback before run (research_id omitted) */}
          <div className="mt-2">
            <ClassificationFeedback
              researchId={null}
              query={liveQuery}
              classification={liveClassification}
            />
          </div>
        </div>
      )}

      {paradigmClassification && (
        <div className="animate-scale-in">
          <ParadigmDisplay classification={paradigmClassification} />
          {/* Classification feedback tied to this research run when available */}
          <div className="mt-2">
            <ClassificationFeedback
              researchId={results?.research_id || null}
              query={results?.query || liveQuery}
              classification={paradigmClassification}
            />
          </div>
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
