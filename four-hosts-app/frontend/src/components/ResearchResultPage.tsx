import { useState, useEffect, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { FiAlertCircle } from 'react-icons/fi'
import { SkeletonLoader } from './SkeletonLoader'
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from './ui/Card'
import { Button } from './ui/Button'
// Alert component not currently used
import { ResultsDisplayEnhanced } from './ResultsDisplayEnhanced'
import api from '../services/api'
import type { ResearchResult } from '../types'

export const ResearchResultPage = () => {
  const { id } = useParams<{ id: string }>()
  const [results, setResults] = useState<ResearchResult | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  useEffect(() => {
    if (!id) return

    const fetchOnce = async () => {
      try {
        const data = await api.getResearchResults(id)

        // Handle terminal failure / cancellation
        if (data.status === 'failed' || data.status === 'cancelled') {
          setError(data.message || `Research ${data.status}`)
          setIsLoading(false)
          if (pollRef.current) clearInterval(pollRef.current)
          return
        }

        // If not yet completed, keep polling
        if (data.status && data.status !== 'completed') {
          setIsLoading(true)
          return
        }

        // Completed result
        setResults(data)
        setIsLoading(false)
        if (pollRef.current) clearInterval(pollRef.current)
      } catch {
        setError('Failed to load research results')
        setIsLoading(false)
        if (pollRef.current) clearInterval(pollRef.current)
      }
    }

    // Initial fetch immediately
    fetchOnce()

    // Poll every 2s until completion
    pollRef.current = setInterval(fetchOnce, 2000)

  // We intentionally do not include params.id to avoid re-fetching on same id
  }, [id])

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        <SkeletonLoader type='card' count={2} className="bg-surface-subtle" />
        <SkeletonLoader type='text' count={3} />
      </div>
    )
  }

  if (error || !results) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Card className="text-center">
          <CardHeader>
            <div className="flex justify-center mb-4">
              <FiAlertCircle className="h-16 w-16 text-error" />
            </div>
            <CardTitle>Research Unavailable</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-text-muted mb-6">
              {error || 'Results not found'}
            </p>
          </CardContent>
          <CardFooter>
            <div className="flex gap-4 justify-center">
              <Button
                variant="secondary"
                onClick={() => window.history.back()}
              >
                Go Back
              </Button>
              <Link to="/history">
                <Button variant="primary">
                  View History
                </Button>
              </Link>
            </div>
          </CardFooter>
        </Card>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <ResultsDisplayEnhanced results={results} />
    </div>
  )
}
