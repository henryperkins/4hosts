import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { PageTransition } from './components/ui/PageTransition'

// Context providers
import { AuthProvider } from './contexts/AuthContext'
import { ThemeContext, useTheme } from './contexts/ThemeContext'

// Components
import { Navigation } from './components/Navigation'
import { ResearchPage } from './components/ResearchPage'
import { ResearchResultPage } from './components/ResearchResultPage'
import { LoginForm } from './components/auth/LoginForm'
import { RegisterForm } from './components/auth/RegisterForm'
import { ProtectedRoute } from './components/auth/ProtectedRoute'
import { UserProfile } from './components/UserProfile'
import { ResearchHistory } from './components/ResearchHistory'
import { MetricsDashboard } from './components/MetricsDashboard'

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