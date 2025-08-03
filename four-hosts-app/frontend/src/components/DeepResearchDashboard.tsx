import React, { useState, useEffect } from 'react'
import { Zap, Clock, Brain, TrendingUp, RefreshCw, Play, AlertCircle } from 'lucide-react'
import { Card, CardHeader, CardContent } from './ui/Card'
import { Button } from './ui/Button'
import { Badge } from './ui/Badge'
import { StatusIcon } from './ui/StatusIcon'
import { LoadingSpinner } from './ui/LoadingSpinner'
import { ResearchProgress } from './ResearchProgress'
import api from '../services/api'
import { useAuth } from '../hooks/useAuth'
import type { ResearchOptions } from '../types'
import toast from 'react-hot-toast'

interface DeepResearchItem {
  research_id: string
  query: string
  status: string
  created_at: string
  paradigm?: string
  has_results: boolean
}

interface DeepResearchDashboardProps {
  onSubmitResearch?: (query: string, options: ResearchOptions) => void
}

export const DeepResearchDashboard: React.FC<DeepResearchDashboardProps> = ({ onSubmitResearch }) => {
  const { user } = useAuth()
  const [deepResearches, setDeepResearches] = useState<DeepResearchItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedResearch, setSelectedResearch] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [query, setQuery] = useState('')
  const [contextSize, setContextSize] = useState<'small' | 'medium' | 'large'>('medium')
  const [userLocation, setUserLocation] = useState('')

  // Check if user has access to deep research
  const hasDeepResearchAccess = user?.role && ['pro', 'enterprise', 'admin'].includes(user.role)

  useEffect(() => {
    if (hasDeepResearchAccess) {
      loadDeepResearches()
    }
  }, [hasDeepResearchAccess])

  const loadDeepResearches = async () => {
    try {
      setIsLoading(true)
      const data = await api.getDeepResearchStatus()
      setDeepResearches((data as { deep_research_queries?: DeepResearchItem[] }).deep_research_queries || [])
    } catch {
      // Failed to load deep research status
      toast.error('Failed to load deep research data')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmitDeepResearch = async () => {
    if (!query.trim()) {
      toast.error('Please enter a research query')
      return
    }

    setIsSubmitting(true)
    try {
      const locationData = userLocation.trim() ? { country: userLocation } : undefined
      const result = await api.submitDeepResearch(query, undefined, contextSize, locationData)

      toast.success('Deep research started!')
      setSelectedResearch(result.research_id)
      setQuery('')

      // Refresh the list
      await loadDeepResearches()
    } catch (error) {
      toast.error((error as Error).message || 'Failed to start deep research')
      // Deep research submission error
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleResumeResearch = async (researchId: string) => {
    try {
      await api.resumeDeepResearch(researchId)
      toast.success('Deep research resumed')
      setSelectedResearch(researchId)
      await loadDeepResearches()
    } catch (error) {
      toast.error((error as Error).message || 'Failed to resume deep research')
      // Resume error
    }
  }

  const getStatusVariant = (status: string): 'default' | 'success' | 'error' | 'warning' | 'info' => {
    switch (status) {
      case 'completed': return 'success'
      case 'failed': return 'error'
      case 'cancelled': return 'warning'
      case 'processing':
      case 'in_progress': return 'info'
      default: return 'default'
    }
  }

  if (!hasDeepResearchAccess) {
    return (
      <Card className="p-8 text-center">
        <Zap className="h-16 w-16 text-orange-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold mb-2">Deep Research with o3 Model</h2>
        <p className="text-text-muted mb-6">
          Access advanced AI research capabilities with OpenAI's most powerful reasoning model.
        </p>
        <p className="text-sm text-text-muted mb-4">
          Deep Research is available for PRO subscribers and above.
        </p>
        <Button variant="primary" className="bg-gradient-to-r from-orange-500 to-yellow-500">
          Upgrade to PRO
        </Button>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* New Deep Research Form */}
      <Card>
        <CardHeader>
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Zap className="h-5 w-5 text-orange-500" />
            Start Deep Research
          </h2>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-text mb-2">
                Research Query
              </label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full p-3 border border-border rounded-lg resize-none focus:ring-2 focus:ring-primary focus:border-primary"
                rows={3}
                placeholder="Enter your complex research question for deep analysis..."
                disabled={isSubmitting}
              />
              <p className="text-xs text-text-muted mt-1">
                Deep research is best for complex, multi-faceted questions that benefit from extended reasoning.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-text mb-2">
                  Search Context Size
                </label>
                <select
                  value={contextSize}
                  onChange={(e) => setContextSize(e.target.value as 'small' | 'medium' | 'large')}
                  className="w-full p-2 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary"
                  disabled={isSubmitting}
                >
                  <option value="small">Small (Fast, focused)</option>
                  <option value="medium">Medium (Balanced)</option>
                  <option value="large">Large (Comprehensive)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-text mb-2">
                  Location Context (Optional)
                </label>
                <input
                  type="text"
                  value={userLocation}
                  onChange={(e) => setUserLocation(e.target.value)}
                  className="w-full p-2 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary"
                  placeholder="e.g., United States, Europe"
                  disabled={isSubmitting}
                />
              </div>
            </div>

            <Button
              onClick={handleSubmitDeepResearch}
              disabled={!query.trim()}
              loading={isSubmitting}
              icon={Brain}
              className="bg-gradient-to-r from-orange-500 to-yellow-500 hover:from-orange-600 hover:to-yellow-600"
            >
              {isSubmitting ? 'Starting Deep Research...' : 'Start Deep Research'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Active Research Progress */}
      {selectedResearch && (
        <ResearchProgress
          researchId={selectedResearch}
          onComplete={() => {
            setSelectedResearch(null)
            loadDeepResearches()
          }}
          onCancel={() => {
            setSelectedResearch(null)
            loadDeepResearches()
          }}
        />
      )}

      {/* Deep Research History */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Deep Research History
            </h2>
            <Button
              variant="secondary"
              size="sm"
              icon={RefreshCw}
              onClick={loadDeepResearches}
              loading={isLoading}
            >
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-center py-8">
              <LoadingSpinner size="lg" text="Loading deep research history..." />
            </div>
          ) : deepResearches.length === 0 ? (
            <div className="text-center py-8">
              <Brain className="h-12 w-12 text-text-muted mx-auto mb-3 opacity-50" />
              <p className="text-text-muted">No deep research queries yet</p>
              <p className="text-sm text-text-muted mt-1">
                Start your first deep research above to see it here
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {deepResearches.map((research) => (
                <div
                  key={research.research_id}
                  className="border border-border rounded-lg p-4 hover:bg-surface-subtle transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <StatusIcon
                          status={research.status as 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'}
                          size="sm"
                        />
                        <h3 className="font-medium text-text line-clamp-1">
                          {research.query}
                        </h3>
                        <Badge variant={getStatusVariant(research.status)}>
                          {research.status}
                        </Badge>
                      </div>

                      <div className="flex items-center gap-4 text-sm text-text-muted">
                        <span>
                          {new Date(research.created_at).toLocaleString()}
                        </span>
                        {research.paradigm && (
                          <span className="capitalize">
                            {research.paradigm} paradigm
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {research.status === 'failed' && (
                        <Button
                          variant="secondary"
                          size="sm"
                          icon={Play}
                          onClick={() => handleResumeResearch(research.research_id)}
                        >
                          Resume
                        </Button>
                      )}

                      {(research.status === 'processing' || research.status === 'in_progress') && (
                        <Button
                          variant="secondary"
                          size="sm"
                          icon={TrendingUp}
                          onClick={() => setSelectedResearch(research.research_id)}
                        >
                          View Progress
                        </Button>
                      )}

                      {research.status === 'completed' && research.has_results && (
                        <Button
                          variant="primary"
                          size="sm"
                          onClick={() => {
                            // Navigate to results page or trigger onSubmitResearch
                            if (onSubmitResearch) {
                              onSubmitResearch(research.query, {
                                depth: 'deep_research',
                                enable_real_search: true
                              })
                            }
                          }}
                        >
                          View Results
                        </Button>
                      )}
                    </div>
                  </div>

                  {research.status === 'failed' && (
                    <div className="mt-3 p-2 bg-error/10 border border-error/20 rounded flex items-center gap-2">
                      <AlertCircle className="h-4 w-4 text-error" />
                      <span className="text-sm text-error">
                        Research failed. You can resume from where it left off.
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Deep Research Info */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <Zap className="h-5 w-5 text-orange-500 shrink-0 mt-0.5" />
            <div>
              <h3 className="font-medium text-text mb-1">About Deep Research</h3>
              <p className="text-sm text-text-muted">
                Deep Research uses OpenAI's o3 model for extended reasoning on complex topics.
                It can take 5-15 minutes but provides more thorough analysis than standard research.
                Best for multi-faceted questions requiring deep understanding.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
