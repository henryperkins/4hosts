import { useEffect, lazy, Suspense } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { PageTransition } from './components/ui/PageTransition'
import { useAuthStore } from './store/authStore'
import { useThemeStore, useDarkMode } from './store/themeStore'
import { CSRFProtection } from './services/csrf-protection'

// Core components (loaded immediately)
import { Navigation } from './components/Navigation'
import { ProtectedRoute } from './components/auth/ProtectedRoute'

// Lazy-loaded components
const ResearchPage = lazy(() => import('./components/ResearchPage').then(m => ({ default: m.ResearchPage })))
const ResearchResultPage = lazy(() => import('./components/ResearchResultPage').then(m => ({ default: m.ResearchResultPage })))
const LoginForm = lazy(() => import('./components/auth/LoginForm').then(m => ({ default: m.LoginForm })))
const RegisterForm = lazy(() => import('./components/auth/RegisterForm').then(m => ({ default: m.RegisterForm })))
const UserProfile = lazy(() => import('./components/UserProfile').then(m => ({ default: m.UserProfile })))
const ResearchHistory = lazy(() => import('./components/ResearchHistory').then(m => ({ default: m.ResearchHistory })))
const MetricsDashboard = lazy(() => import('./components/MetricsDashboard').then(m => ({ default: m.MetricsDashboard })))

// Loading component
const LoadingSpinner = () => (
  <div className="flex items-center justify-center min-h-screen">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
  </div>
)

// Main App component
function App() {
  const darkMode = useDarkMode()
  const { setDarkMode } = useThemeStore()

  // Initialize theme on mount
  useEffect(() => {
    const saved = localStorage.getItem('theme-store')
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        if (parsed.state?.darkMode !== undefined) {
          setDarkMode(parsed.state.darkMode)
        }
      } catch {
        // Ignore parse errors
      }
    }
  }, [setDarkMode])

  // Initialize CSRF token and check auth on mount
  useEffect(() => {
    // Initialize CSRF token
    CSRFProtection.getToken().catch(console.error)
    // Check auth
    useAuthStore.getState().checkAuth()
  }, [])

  return (
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
          <Suspense fallback={<LoadingSpinner />}>
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
          </Suspense>
          </div>
        </Router>
  )
}

export default App