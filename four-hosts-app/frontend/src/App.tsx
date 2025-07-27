import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, Navigate, useParams } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { Home, History, User, BarChart3 } from 'lucide-react'

// Context providers
import { AuthProvider, useAuth } from './contexts/AuthContext'

// Components
import { LoginForm } from './components/auth/LoginForm'
import { RegisterForm } from './components/auth/RegisterForm'
import { ProtectedRoute } from './components/auth/ProtectedRoute'
import { ResearchFormEnhanced } from './components/ResearchFormEnhanced'
import { ResearchProgress } from './components/ResearchProgress'
import { ResultsDisplayEnhanced } from './components/ResultsDisplayEnhanced'
import ParadigmDisplay from './components/ParadigmDisplay'
import { UserProfile } from './components/UserProfile'
import { ResearchHistory } from './components/ResearchHistory'
import { MetricsDashboard } from './components/MetricsDashboard'

// Services and types
import api from './services/api'
import type { ResearchResult, ParadigmClassification } from './types'

// Navigation component
const Navigation = () => {
  const { isAuthenticated, user } = useAuth()

  if (!isAuthenticated) return null

  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center gap-8">
            <Link to="/" className="text-xl font-bold text-gray-900">
              Four Hosts Research
            </Link>
            <div className="flex items-center gap-6">
              <Link
                to="/"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <Home className="h-4 w-4" />
                <span>Research</span>
              </Link>
              <Link
                to="/history"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <History className="h-4 w-4" />
                <span>History</span>
              </Link>
              <Link
                to="/metrics"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <BarChart3 className="h-4 w-4" />
                <span>Metrics</span>
              </Link>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600">
              Welcome, {user?.username}
            </span>
            <Link
              to="/profile"
              className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <User className="h-4 w-4" />
              <span>Profile</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}

// Main research page component
const ResearchPage = () => {
  const [isLoading, setIsLoading] = useState(false)
  const [paradigmClassification, setParadigmClassification] = useState<ParadigmClassification | null>(null)
  const [results, setResults] = useState<ResearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [currentResearchId, setCurrentResearchId] = useState<string | null>(null)
  const [showProgress, setShowProgress] = useState(false)

  const handleSubmit = async (query: string, options: any) => {
    setIsLoading(true)
    setError(null)
    setResults(null)
    setShowProgress(true)
    
    try {
      // Submit research query
      const data = await api.submitResearch(query, options)
      setParadigmClassification(data.paradigm_classification)
      setCurrentResearchId(data.research_id)
      
      // Poll for results
      let retries = 0
      const maxRetries = 60
      
      const pollInterval = setInterval(async () => {
        try {
          const resultsData = await api.getResearchResults(data.research_id)
          setResults(resultsData)
          setIsLoading(false)
          setShowProgress(false)
          clearInterval(pollInterval)
        } catch (err) {
          // Continue polling if not ready
          if (retries >= maxRetries) {
            setError('Research timeout - please try again')
            setIsLoading(false)
            setShowProgress(false)
            clearInterval(pollInterval)
          }
        }
        retries++
      }, 2000)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      setIsLoading(false)
      setShowProgress(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <ResearchFormEnhanced onSubmit={handleSubmit} isLoading={isLoading} />
      
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">{error}</p>
        </div>
      )}
      
      {paradigmClassification && (
        <ParadigmDisplay classification={paradigmClassification} />
      )}
      
      {showProgress && currentResearchId && (
        <ResearchProgress 
          researchId={currentResearchId} 
          onComplete={() => setShowProgress(false)}
        />
      )}
      
      {results && !showProgress && (
        <ResultsDisplayEnhanced results={results} />
      )}
    </div>
  )
}

// Research result page (for viewing historical results)
const ResearchResultPage = () => {
  const { id } = useParams<{ id: string }>()
  const [results, setResults] = useState<ResearchResult | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadResults = async () => {
      if (!id) return
      
      try {
        const data = await api.getResearchResults(id)
        setResults(data)
      } catch (err) {
        setError('Failed to load research results')
      } finally {
        setIsLoading(false)
      }
    }

    loadResults()
  }, [id])

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      </div>
    )
  }

  if (error || !results) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error || 'Results not found'}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <ResultsDisplayEnhanced results={results} />
    </div>
  )
}

// Main App component
function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Toaster position="top-right" />
          <Navigation />
          
          <Routes>
            {/* Public routes */}
            <Route path="/login" element={<LoginForm />} />
            <Route path="/register" element={<RegisterForm />} />
            
            {/* Protected routes */}
            <Route path="/" element={
              <ProtectedRoute>
                <ResearchPage />
              </ProtectedRoute>
            } />
            
            <Route path="/research/:id" element={
              <ProtectedRoute>
                <ResearchResultPage />
              </ProtectedRoute>
            } />
            
            <Route path="/history" element={
              <ProtectedRoute>
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                  <ResearchHistory />
                </div>
              </ProtectedRoute>
            } />
            
            <Route path="/profile" element={
              <ProtectedRoute>
                <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                  <UserProfile />
                </div>
              </ProtectedRoute>
            } />
            
            <Route path="/metrics" element={
              <ProtectedRoute>
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                  <h1 className="text-2xl font-bold text-gray-900 mb-6">System Metrics</h1>
                  <MetricsDashboard />
                </div>
              </ProtectedRoute>
            } />
            
            {/* Default redirect */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </Router>
    </AuthProvider>
  )
}

export default App