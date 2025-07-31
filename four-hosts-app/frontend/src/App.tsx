import { useState, useEffect, useRef } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, Navigate, useParams, useLocation, useNavigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { Home, History, User, BarChart3, Menu, X, AlertCircle, Loader2 } from 'lucide-react'
import { PageTransition } from './components/ui/PageTransition'
import { ToggleSwitch } from './components/ui/ToggleSwitch'

// Context providers
import { AuthProvider } from './contexts/AuthContext'
import { useAuth } from './hooks/useAuth'
import { ThemeContext, useTheme } from './contexts/ThemeContext'

// Components
import { LoginForm } from './components/auth/LoginForm'
import { RegisterForm } from './components/auth/RegisterForm'
import { ProtectedRoute } from './components/auth/ProtectedRoute'
import { ResearchFormEnhanced } from './components/ResearchFormEnhanced'
import { ResearchFormIdeaBrowser } from './components/ResearchFormIdeaBrowser'
import { ResearchProgress } from './components/ResearchProgress'
import { ResearchProgressIdeaBrowser } from './components/ResearchProgressIdeaBrowser'
import { ResultsDisplayEnhanced } from './components/ResultsDisplayEnhanced'
import { ResultsDisplayIdeaBrowser } from './components/ResultsDisplayIdeaBrowser'
import ParadigmDisplay from './components/ParadigmDisplay'
import { UserProfile } from './components/UserProfile'
import { ResearchHistory } from './components/ResearchHistory'
import { MetricsDashboard } from './components/MetricsDashboard'

// Services and types
import api from './services/api'
import type { ResearchResult, ParadigmClassification, ResearchOptions } from './types'

// Navigation component
const Navigation = () => {
  const { isAuthenticated, user, logout } = useAuth()
  const location = useLocation()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const { darkMode, toggleDarkMode } = useTheme()

  if (!isAuthenticated) return null

  const isActive = (path: string) => {
    return location.pathname === path
  }

  const closeMobileMenu = () => {
    setMobileMenuOpen(false)
  }

  // Paradigm-themed navigation items
  const navItems = [
    { path: '/', icon: Home, label: 'Research', paradigm: 'dolores' },
    { path: '/history', icon: History, label: 'History', paradigm: 'bernard' },
    { path: '/metrics', icon: BarChart3, label: 'Metrics', paradigm: 'teddy' },
  ]

  const getParadigmHoverClass = (paradigm?: string) => {
    if (!paradigm) return ''
    const paradigmHoverColors: Record<string, string> = {
      dolores: 'hover:text-paradigm-dolores',
      bernard: 'hover:text-paradigm-bernard',
      teddy: 'hover:text-paradigm-teddy',
      maeve: 'hover:text-paradigm-maeve'
    }
    return paradigmHoverColors[paradigm] || ''
  }

  return (
    <>
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-primary text-white px-4 py-2 rounded-md z-50">
        Skip to main content
      </a>
      <nav className="bg-surface shadow-lg border-b border-border animate-slide-down transition-all duration-300 backdrop-blur-sm bg-opacity-95 dark:bg-opacity-95">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center gap-8">
            <Link
              to="/"
              className="text-xl font-bold text-text hover:text-blue-600 dark:hover:text-blue-400 transition-all duration-300 flex items-center gap-2 group touch-target"
              onClick={closeMobileMenu}
            >
              <span className="text-2xl group-hover:rotate-12 transition-transform duration-300 inline-block" role="img" aria-label="Theater masks">🎭</span>
              <span className="hidden sm:inline gradient-brand text-responsive-xl">Four Hosts Research</span>
              <span className="sm:hidden gradient-brand text-responsive-lg">4H Research</span>
            </Link>
            <div className="hidden md:flex items-center gap-2">
              {navItems.map(({ path, icon: Icon, label, paradigm }) => {
                return (
                  <Link
                    key={path}
                    to={path}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 hover-lift focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 touch-target ${
                      isActive(path)
                        ? 'bg-linear-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 text-blue-700 dark:text-blue-300 shadow-lg scale-105'
                        : `text-text-muted hover:bg-surface-subtle ${getParadigmHoverClass(paradigm)} active:scale-95`
                    }`}
                    aria-current={isActive(path) ? 'page' : undefined}
                  >
                    <Icon className={`h-4 w-4 transition-all duration-300 ${isActive(path) ? 'animate-pulse' : 'group-hover:rotate-12'}`} />
                    <span>{label}</span>
                  </Link>
                )
              })}
            </div>
          </div>

          <div className="flex items-center gap-2 md:gap-4">
            <span className="hidden md:block text-sm text-text-muted animate-fade-in">
              Welcome, <span className="font-medium gradient-accent">{user?.username}</span>
            </span>

            {/* Dark mode toggle */}
            <ToggleSwitch
              checked={darkMode}
              onChange={toggleDarkMode}
              aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              size="sm"
              className="hidden md:inline-flex"
            />

            <Link
              to="/profile"
              className={`hidden md:flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 hover-lift focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                isActive('/profile')
                  ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 shadow-md'
                  : 'text-text-muted hover:text-text hover:bg-surface-subtle'
              }`}
              aria-current={isActive('/profile') ? 'page' : undefined}
            >
              <User className="h-4 w-4 transition-transform duration-200 hover:scale-110" />
              <span>Profile</span>
            </Link>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-text-muted hover:text-text hover:bg-surface-subtle transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 touch-target active:scale-95"
              aria-label="Toggle menu"
              aria-expanded={mobileMenuOpen}
            >
              {mobileMenuOpen ? (
                <X className="h-5 w-5" />
              ) : (
                <Menu className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile menu with smooth transitions */}
        <div className={`md:hidden transition-all duration-300 ease-in-out overflow-hidden ${
          mobileMenuOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
        }`}>
          <div className="py-2 space-y-1">
            {navItems.map(({ path, label }) => (
              <Link
                key={path}
                to={path}
                onClick={closeMobileMenu}
                className={`block px-4 py-2 transition-all duration-300 transform hover:translate-x-2 ${
                  isActive(path)
                  ? 'bg-linear-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 text-blue-700 dark:text-blue-300'
                    : 'text-text-muted hover:bg-surface-subtle'
                }`}
              >
                {label}
              </Link>
            ))}
            <div className="px-4 py-2 border-t border-border mt-2 pt-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-muted">Dark Mode</span>
                <ToggleSwitch
                  checked={darkMode}
                  onChange={toggleDarkMode}
                  size="sm"
                />
              </div>
            </div>
            <button
              onClick={() => {
                logout()
                closeMobileMenu()
              }}
              className="block w-full text-left px-4 py-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors duration-200"
            >
              Logout
            </button>
          </div>
        </div>
      </div>
    </nav>
    </>
  )
}

// Main research page component with enhanced animations
const ResearchPage = () => {
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
          
          // Check if research is still in progress
          if (resultsData.status && resultsData.status !== 'completed' && resultsData.status !== 'failed' && resultsData.status !== 'cancelled') {
            // Still processing, continue polling
            if (retries >= maxRetries) {
              setError('Research timeout - please try again')
              setIsLoading(false)
              setShowProgress(false)
              clearInterval(pollIntervalRef.current!)
            }
          } else if (resultsData.status === 'failed' || resultsData.status === 'cancelled') {
            // Research failed or was cancelled
            setError(`Research ${resultsData.status}: ${resultsData.message || 'Please try again'}`)
            setIsLoading(false)
            setShowProgress(false)
            clearInterval(pollIntervalRef.current!)
          } else {
            // Research completed successfully
            setResults(resultsData)
            // Extract paradigm classification from results
            if (resultsData.paradigm_analysis && resultsData.paradigm_analysis.primary) {
              // Build distribution from primary and secondary paradigms
              const distribution: Record<string, number> = {
                [resultsData.paradigm_analysis.primary.paradigm]: resultsData.paradigm_analysis.primary.confidence
              }
              
              if (resultsData.paradigm_analysis.secondary) {
                distribution[resultsData.paradigm_analysis.secondary.paradigm] = resultsData.paradigm_analysis.secondary.confidence
              }
              
              // Fill in other paradigms with 0 if not present
              const allParadigms = ['dolores', 'teddy', 'bernard', 'maeve']
              allParadigms.forEach(p => {
                if (!distribution[p]) {
                  distribution[p] = 0
                }
              })
              
              setParadigmClassification({
                primary: resultsData.paradigm_analysis.primary.paradigm,
                secondary: resultsData.paradigm_analysis.secondary?.paradigm || null,
                distribution,
                confidence: resultsData.paradigm_analysis.primary.confidence,
                explanation: {
                  [resultsData.paradigm_analysis.primary.paradigm]: resultsData.paradigm_analysis.primary.approach
                }
              })
            }
            setIsLoading(false)
            setShowProgress(false)
            clearInterval(pollIntervalRef.current!)
          }
        } catch {
          // API error - continue polling if not at max retries
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
      setError(err instanceof Error ? err.message : 'An error occurred')
      setIsLoading(false)
      setShowProgress(false)
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
        <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg animate-slide-down transform transition-all duration-300 hover:scale-[1.02]">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400 shrink-0 animate-pulse" />
            <p className="text-red-800 dark:text-red-200">{error}</p>
          </div>
        </div>
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

// Research result page (for viewing historical results)
const ResearchResultPage = () => {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [results, setResults] = useState<ResearchResult | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [useIdeaBrowser] = useState(() => {
    try {
      const saved = localStorage.getItem('useIdeaBrowser')
      return saved ? JSON.parse(saved) : false
    } catch {
      return false
    }
  })

  useEffect(() => {
    const loadResults = async () => {
      if (!id) return

      try {
        const data = await api.getResearchResults(id)
        
        // Check if this is a failed/cancelled research response
        if ('status' in data && (data.status === 'failed' || data.status === 'cancelled')) {
          setError(`Research ${data.status}`)
          return
        }
        
        // Check if this is a still-processing research
        if (data.status && data.status !== 'completed' && data.status !== 'failed' && data.status !== 'cancelled') {
          // Research is still in progress, redirect to home page to show progress
          navigate('/')
          return
        }
        
        setResults(data)
      } catch (err) {
        // Failed to load research results
        setError('Failed to load research results')
      } finally {
        setIsLoading(false)
      }
    }

    loadResults()
  }, [id, navigate])

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col items-center justify-center py-12">
          <Loader2 className="h-12 w-12 text-blue-600 dark:text-blue-400 animate-spin mb-4" />
          <p className="text-text-muted">Loading research results...</p>
        </div>
      </div>
    )
  }

  if (error || !results) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="card p-8 text-center">
          <AlertCircle className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-text mb-2">
            Research Unavailable
          </h2>
          <p className="text-text-muted mb-6">
            {error || 'Results not found'}
          </p>
          <div className="flex gap-4 justify-center">
            <button
              onClick={() => window.history.back()}
              className="btn-secondary"
            >
              Go Back
            </button>
            <Link
              to="/history"
              className="btn-primary"
            >
              View History
            </Link>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {useIdeaBrowser ? (
        <ResultsDisplayIdeaBrowser results={results} />
      ) : (
        <ResultsDisplayEnhanced results={results} />
      )}
    </div>
  )
}

// Main App component
function App() {
  const [darkMode, setDarkMode] = useState(() => {
    try {
      const saved = localStorage.getItem('darkMode')
      return saved ? JSON.parse(saved) : false
    } catch {
      return false
    }
  })

  const toggleDarkMode = () => {
    setDarkMode((prev: boolean) => {
      const newValue = !prev
      localStorage.setItem('darkMode', JSON.stringify(newValue))
      // Add transition class to body for smooth theme change
      document.body.classList.add('theme-transition')
      setTimeout(() => {
        document.body.classList.remove('theme-transition')
      }, 300)
      return newValue
    })
  }

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  return (
    <ThemeContext.Provider value={{ darkMode, toggleDarkMode }}>
      <AuthProvider>
        <Router>
          <div className="min-h-screen bg-surface transition-colors duration-200">
            <Toaster
              position="top-right"
              toastOptions={{
                className: 'dark:bg-gray-800 dark:text-gray-100',
                style: {
                  background: darkMode ? '#1f2937' : undefined,
                  color: darkMode ? '#f3f4f6' : undefined,
                },
              }}
            />
            <Navigation />
            <Routes>
              {/* Public routes */}
              <Route path="/login" element={<LoginForm />} />
              <Route path="/register" element={<RegisterForm />} />

              {/* Protected routes */}
              <Route path="/" element={
                <ProtectedRoute>
                  <PageTransition>
                    <ResearchPage />
                  </PageTransition>
                </ProtectedRoute>
              } />

              <Route path="/research/:id" element={
                <ProtectedRoute>
                  <PageTransition>
                    <ResearchResultPage />
                  </PageTransition>
                </ProtectedRoute>
              } />

              <Route path="/history" element={
                <ProtectedRoute>
                  <PageTransition>
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                      <ResearchHistory />
                    </div>
                  </PageTransition>
                </ProtectedRoute>
              } />

              <Route path="/profile" element={
                <ProtectedRoute>
                  <PageTransition>
                    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                      <UserProfile />
                    </div>
                  </PageTransition>
                </ProtectedRoute>
              } />

              <Route path="/metrics" element={
                <ProtectedRoute>
                  <PageTransition>
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                      <h1 className="text-2xl font-bold text-text mb-6 animate-slide-down">System Metrics</h1>
                      <MetricsDashboard />
                    </div>
                  </PageTransition>
                </ProtectedRoute>
              } />

              {/* Default redirect */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </Router>
      </AuthProvider>
    </ThemeContext.Provider>
  )
}

export default App
