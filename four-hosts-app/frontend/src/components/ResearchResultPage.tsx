import { useState, useEffect, useRef } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { FiAlertCircle } from 'react-icons/fi'
import { SkeletonLoader } from './SkeletonLoader'
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from './ui/Card'
import { Button } from './ui/Button'
import { ResearchDisplayContainer } from './research-display/ResearchDisplayContainer'
import api from '../services/api'
import type { ResearchResult } from '../types'

type ApiError = Error & { status?: number }

export const ResearchResultPage = () => {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [results, setResults] = useState<ResearchResult | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [errorCode, setErrorCode] = useState<number | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  useEffect(() => {
    if (!id) {
      setError('Missing research identifier')
      setIsLoading(false)
      return
    }

    let cancelled = false

    const fetchOnce = async () => {
      try {
        const data = await api.getResearchResults(id)
        if (cancelled) return

        // Handle terminal failure / cancellation
        if (data.status === 'failed' || data.status === 'cancelled') {
          setError(data.message || `Research ${data.status}`)
          setErrorCode(null)
          setIsLoading(false)
          if (pollRef.current) clearInterval(pollRef.current)
          return
        }

        // If not yet completed, keep polling
        if (data.status && data.status !== 'completed') {
          setIsLoading(true)
          setError(null)
          setErrorCode(null)
          return
        }

        setResults(data)
        setError(null)
        setErrorCode(null)
        setIsLoading(false)
        if (pollRef.current) clearInterval(pollRef.current)
      } catch (err) {
        if (cancelled) return
        const apiError = err as ApiError
        const message = apiError?.message || 'Failed to load research results'
        const status = typeof apiError?.status === 'number' ? apiError.status : null
        setError(message)
        setErrorCode(status)
        setIsLoading(false)
        if (pollRef.current) clearInterval(pollRef.current)
        if (status === 401) {
          navigate('/login')
        }
      }
    }

    fetchOnce()
    pollRef.current = setInterval(fetchOnce, 2000)

    return () => {
      cancelled = true
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [id, navigate])

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        <SkeletonLoader type='card' count={2} className="bg-surface-subtle" />
        <SkeletonLoader type='text' count={3} />
      </div>
    )
  }

  if (error || !results) {
    const heading = errorCode === 404 ? 'Research Not Found' : 'Research Unavailable'
    const description = error || (errorCode === 404
      ? 'We couldn’t find this research request. It may have expired or been removed.'
      : 'We hit a problem retrieving this research. Please try again or return to your history.')
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Card className="text-center">
          <CardHeader>
            <div className="flex justify-center mb-4">
              <FiAlertCircle className="h-16 w-16 text-error" />
            </div>
            <CardTitle>{heading}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-text-muted mb-6">
              {description}
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
              <Link to="/">
                <Button variant="ghost">
                  Back to Research
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
      <ResearchDisplayContainer results={results} />
    </div>
  )
}
