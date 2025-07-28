import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, Navigate, useParams, useLocation } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { Home, History, User, BarChart3, Menu, X, Sun, Moon, AlertCircle, Loader2 } from 'lucide-react'
import { PageTransition } from './components/ui/PageTransition'

// Context providers
import { AuthProvider } from './contexts/AuthContext'
import { useAuth } from './hooks/useAuth'
import { ThemeContext, useTheme } from './contexts/ThemeContext'

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
      dolores: 'hover:text-red-600 dark:hover:text-red-400',
      bernard: 'hover:text-blue-600 dark:hover:text-blue-400',
      teddy: 'hover:text-orange-600 dark:hover:text-orange-400',
      maeve: 'hover:text-green-600 dark:hover:text-green-400'
    }
    return paradigmHoverColors[paradigm] || ''
  }

  return (
    <nav className="bg-white dark:bg-gray-800 shadow-lg border-b border-gray-200 dark:border-gray-700 animate-slide-down transition-all duration-300 backdrop-blur-sm bg-opacity-95 dark:bg-opacity-95">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center gap-8">
            <Link
              to="/"
              className="text-xl font-bold text-gray-900 dark:text-gray-100 hover:text-blue-600 dark:hover:text-blue-400 transition-all duration-300 flex items-center gap-2 group"
              onClick={closeMobileMenu}
            >
              <span className="text-2xl group-hover:rotate-12 transition-transform duration-300 inline-block" role="img" aria-label="Theater masks">🎭</span>
              <span className="hidden sm:inline bg-gradient-to-r from-red-600 via-blue-600 to-green-600 dark:from-red-400 dark:via-blue-400 dark:to-green-400 bg-clip-text text-transparent">Four Hosts Research</span>
              <span className="sm:hidden bg-gradient-to-r from-red-600 via-blue-600 to-green-600 dark:from-red-400 dark:via-blue-400 dark:to-green-400 bg-clip-text text-transparent">4H Research</span>
            </Link>
            <div className="hidden md:flex items-center gap-2">
              {navItems.map(({ path, icon: Icon, label, paradigm }) => {
                return (
                  <Link
                    key={path}
                    to={path}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 ${
                      isActive(path)
                        ? 'bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 text-blue-700 dark:text-blue-300 shadow-lg scale-105'
                        : `text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 ${getParadigmHoverClass(paradigm)}`
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
            <span className="hidden md:block text-sm text-gray-600 dark:text-gray-400 animate-fade-in">
              Welcome, <span className="font-medium text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-pink-600 dark:from-purple-400 dark:to-pink-400">{user?.username}</span>
            </span>

            {/* Dark mode toggle with animation */}
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-300 transform hover:scale-110 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
              aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              <div className="relative w-5 h-5">
                <Sun className={`absolute inset-0 h-5 w-5 transform transition-all duration-300 ${darkMode ? 'opacity-0 rotate-90 scale-0' : 'opacity-100 rotate-0 scale-100'}`} />
                <Moon className={`absolute inset-0 h-5 w-5 transform transition-all duration-300 ${darkMode ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 -rotate-90 scale-0'}`} />
              </div>
            </button>

            <Link
              to="/profile"
              className={`hidden md:flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 hover-lift focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 ${
                isActive('/profile')
                  ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 shadow-md'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
              aria-current={isActive('/profile') ? 'page' : undefined}
            >
              <User className="h-4 w-4 transition-transform duration-200 hover:scale-110" />
              <span>Profile</span>
            </Link>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
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
            {navItems.map(({ path, label }, index) => (
              <Link
                key={path}
                to={path}
                onClick={closeMobileMenu}
                className={`block px-4 py-2 transition-all duration-300 transform hover:translate-x-2 ${
                  isActive(path)
                    ? 'bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 text-blue-700 dark:text-blue-300'
                    : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                style={{ animationDelay: `${index * 50}ms` }}
              >
                {label}
              </Link>
            ))}
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

  const handleSubmit = async (query: string, options: ResearchOptions) => {
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
        } catch {
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
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fade-in">
      <div className="mb-8 text-center animate-slide-down">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">
          Discover Insights Through Four Perspectives
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Let our AI hosts guide your research with their unique paradigms
        </p>
      </div>

      <ResearchFormEnhanced onSubmit={handleSubmit} isLoading={isLoading} />

      {error && (
        <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg animate-slide-down transform transition-all duration-300 hover:scale-[1.02]">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0 animate-pulse" />
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
        
        // Check if this is a failed/cancelled research response
        if (data.status === 'failed' || data.status === 'cancelled') {
          setError(data.message || `Research ${data.status}`)
          return
        }
        
        // Check if this is a still-processing research
        if (data.status && data.status !== 'completed') {
          setError(`Research is still ${data.status}. Please wait for completion.`)
          return
        }
        
        setResults(data)
      } catch (err) {
        console.error('Failed to load research results:', err)
        setError('Failed to load research results')
      } finally {
        setIsLoading(false)
      }
    }

    loadResults()
  }, [id])

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col items-center justify-center py-12">
          <Loader2 className="h-12 w-12 text-blue-600 dark:text-blue-400 animate-spin mb-4" />
          <p className="text-gray-600 dark:text-gray-400">Loading research results...</p>
        </div>
      </div>
    )
  }

  if (error || !results) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 text-center">
          <AlertCircle className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Research Unavailable
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            {error || 'Results not found'}
          </p>
          <div className="flex gap-4 justify-center">
            <button
              onClick={() => window.history.back()}
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              Go Back
            </button>
            <button
              onClick={() => window.location.href = '/history'}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              View History
            </button>
          </div>
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
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode')
    return saved ? JSON.parse(saved) : false
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
          <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
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
                      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6 animate-slide-down">System Metrics</h1>
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
