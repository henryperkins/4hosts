import { useState, useEffect, useRef, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
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

type StagedStatus = 'queued' | 'processing' | 'in_progress' | 'failed' | 'cancelled'

const stagedStatuses: StagedStatus[] = ['queued', 'processing', 'in_progress']

const getStatus = (obj: unknown): StagedStatus | 'completed' | undefined => {
  if (obj && typeof obj === 'object') {
    const rec = obj as Record<string, unknown>
    if (typeof rec.status === 'string') {
      return rec.status as StagedStatus | 'completed'
    }
    if ('answer' in rec && 'paradigm_analysis' in rec) {
      return 'completed'
    }
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
  const [stopPolling, setStopPolling] = useState(false)
  const pollStartRef = useRef<number | null>(null)
  const [classificationState, setClassificationState] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle')

  // Live classification preview with debounce
  useEffect(() => {
    // Reset preview when query cleared or while a research is running
    if (!liveQuery || isLoading) {
      setLiveClassification(null)
      setClassificationState('idle')
      return
    }

    const trimmed = liveQuery.trim()
    if (trimmed.length < 10) {
      setLiveClassification(null)
      setClassificationState('idle')
      return
    }

    let cancelled = false
    setClassificationState('loading')

    const handle = setTimeout(async () => {
      try {
        const data = await api.classifyQuery(trimmed)
        if (data && data.primary) {
          setLiveClassification({
            primary: data.primary,
            secondary: data.secondary ?? null,
            distribution: data.distribution || {},
            confidence: typeof data.confidence === 'number' ? data.confidence : 0,
            explanation: data.explanation || {}
          })
          if (!cancelled) {
            setClassificationState('ready')
          }
        }
      } catch {
        if (!cancelled) {
          setClassificationState('error')
        }
      }
    }, 400)

    return () => {
      cancelled = true
      clearTimeout(handle)
    }
  }, [liveQuery, isLoading])

  const handleSubmit = async (query: string, options: ResearchOptions) => {
    setIsLoading(true)
    setError(null)
    setResults(null)
    setShowProgress(true)
    setStopPolling(false)
    setParadigmClassification(null)
    setClassificationState('idle')

    try {
      // Submit research query
      const data = await api.submitResearch(query, options)
      setCurrentResearchId(data.research_id)
      pollStartRef.current = Date.now()

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

  // React Query polling for results
  const { data: polledResults, error: queryError } = useQuery({
    queryKey: ['research-results', currentResearchId],
    queryFn: () => api.getResearchResults(currentResearchId as string),
    // Keep polling enabled until we have final results OR explicitly stopped
    enabled: !!currentResearchId && !stopPolling,
    // Keep polling until final status
    refetchInterval: (query) => {
      const d: unknown = query.state.data as unknown
      const s = getStatus(d)
      // Keep polling if still in progress
      if (s && stagedStatuses.includes(s as StagedStatus)) return 2000
      // Stop polling once we have a final status
      return false
    },
    staleTime: 1000,
  })

  // Handle query errors separately (React Query v5 doesn't have onError)
  useEffect(() => {
    if (queryError) {
      const message = queryError instanceof Error ? queryError.message : 'Unable to check research status.'
      setError(message)
      setIsLoading(false)
      setShowProgress(false)
      setStopPolling(true)
    }
  }, [queryError])

  // Apply polled results
  useEffect(() => {
    if (!polledResults || !currentResearchId) return

    const status = getStatus(polledResults)

    // Skip if still in progress
    if (status && stagedStatuses.includes(status as StagedStatus)) return

    // Handle error states
    if (status === 'failed' || status === 'cancelled') {
      const message = getMessage(polledResults) || 'Please try again'
      setError(`Research ${status}: ${message}`)
      setIsLoading(false)
      setShowProgress(false)
      setStopPolling(true)
      setCurrentResearchId(null)
      return
    }

    // Final result path - either explicit 'completed' or full results object
    if (status === 'completed' || (polledResults && typeof polledResults === 'object' &&
        'answer' in polledResults && 'paradigm_analysis' in polledResults)) {

      // Set results first
      setResults(polledResults as ResearchResult)

      // Extract paradigm classification from results
      const r = polledResults as ResearchResult
      if (r.paradigm_analysis && r.paradigm_analysis.primary) {
        const primary = r.paradigm_analysis.primary
        const secondary = r.paradigm_analysis.secondary

        const distribution: Record<string, number> = {}
        distribution[primary.paradigm] = typeof primary.confidence === 'number' ? primary.confidence : 0
        if (secondary?.paradigm) {
          distribution[secondary.paradigm] = typeof secondary.confidence === 'number' ? secondary.confidence : 0
        }
        const allParadigms = ['dolores', 'teddy', 'bernard', 'maeve']
        allParadigms.forEach(p => { if (!(p in distribution)) distribution[p] = 0 })
        const explanation: Record<string, string> = { [primary.paradigm]: primary.approach ?? '' }
        if (secondary?.paradigm) explanation[secondary.paradigm] = secondary.approach ?? ''

        setParadigmClassification({
          primary: primary.paradigm,
          secondary: secondary?.paradigm || null,
          distribution,
          confidence: typeof primary.confidence === 'number' ? primary.confidence : 0,
          explanation
        })
      }

      // Clean up states
      setIsLoading(false)
      setShowProgress(false)
      setStopPolling(true)
      setCurrentResearchId(null)
    }
  }, [polledResults, currentResearchId])

  // Soft timeout for polling (default 20 minutes)
  useEffect(() => {
    if (!showProgress || !currentResearchId) return
    const TIMEOUT_MS = Number(import.meta.env.VITE_RESULTS_POLL_TIMEOUT_MS || 20 * 60 * 1000)
    const id = setInterval(() => {
      const start = pollStartRef.current
      if (!start) return
      const elapsed = Date.now() - start
      if (elapsed >= TIMEOUT_MS) {
        setError('We paused live updates after a long run. Your research continues in the background and will appear in History.')
        setIsLoading(false)
        setStopPolling(true)
        clearInterval(id)
      }
    }, 2000)
    return () => clearInterval(id)
  }, [showProgress, currentResearchId])

  const classificationBlock = useMemo(() => {
    if (results && !showProgress && paradigmClassification) {
      return {
        heading: 'Paradigm Overview',
        classification: paradigmClassification,
        researchId: results.research_id || null,
        query: results.query || liveQuery,
      }
    }

    if (!results && !showProgress && liveClassification) {
      return {
        heading: 'Live Paradigm Preview',
        classification: liveClassification,
        researchId: null,
        query: liveQuery,
      }
    }

    return null
  }, [results, showProgress, paradigmClassification, liveClassification, liveQuery])

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

      <ResearchFormEnhanced
        onSubmit={handleSubmit}
        isLoading={isLoading}
        onQueryChange={setLiveQuery}
        classificationState={classificationState}
      />

      {error && (
        <Alert variant="error" title="Research Error" className="mt-4">
          {error}
        </Alert>
      )}

      {classificationBlock && (
        <div className="animate-scale-in">
          <div className="mb-2 text-xs text-text-muted">{classificationBlock.heading}</div>
          <ParadigmDisplay classification={classificationBlock.classification} />
          <div className="mt-2">
            <ClassificationFeedback
              researchId={classificationBlock.researchId}
              query={classificationBlock.query}
              classification={classificationBlock.classification}
            />
          </div>
        </div>
      )}

      {showProgress && currentResearchId && (
        <div className="animate-slide-up">
          <ResearchProgress
            researchId={currentResearchId}
            onComplete={() => {
              setShowProgress(false)
              // Don't need to set isLoading false here - the polling will handle it
            }}
            onCancel={() => {
              setShowProgress(false)
              setCurrentResearchId(null)
              setIsLoading(false)
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
