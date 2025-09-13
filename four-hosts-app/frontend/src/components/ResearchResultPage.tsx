import { useState, useEffect, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { FiAlertCircle } from 'react-icons/fi'
import { LoadingSpinner } from './ui/LoadingSpinner'
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
        if ((data as any).status === 'failed' || (data as any).status === 'cancelled') {
          setError((data as any).message || `Research ${(data as any).status}`)
          setIsLoading(false)
          if (pollRef.current) clearInterval(pollRef.current)
          return
        }

        // If not yet completed, keep polling
        if ((data as any).status && (data as any).status !== 'completed') {
          setIsLoading(true)
          return
        }

        // Completed result
        setResults(data)
        setIsLoading(false)
        if (pollRef.current) clearInterval(pollRef.current)
      } catch (e) {
        setError('Failed to load research results')
        setIsLoading(false)
        if (pollRef.current) clearInterval(pollRef.current)
      }
    }

    // Initial fetch immediately
    fetchOnce()

    // Poll every 2s until completion
    pollRef.current = setInterval(fetchOnce, 2000)

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id])

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col items-center justify-center py-12">
          <LoadingSpinner size="xl" variant="primary" text="Loading research results..." />
        </div>
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
