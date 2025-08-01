import React from 'react'
import { BrowserRouter as Router } from 'react-router-dom'
import { Navigation } from './components/Navigation'
import { ResearchPage } from './components/ResearchPage'
import { ResearchResultPage } from './components/ResearchResultPage'
import { Button } from './components/ui/Button'
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from './components/ui/Card'
import { LoadingSpinner } from './components/ui/LoadingSpinner'
import { Alert } from './components/ui/Alert'
import { ToggleSwitch } from './components/ui/ToggleSwitch'
import { PageTransition } from './components/ui/PageTransition'
import { ThemeContext } from './contexts/ThemeContext'
import { AuthProvider } from './contexts/AuthContext'
import { Home, Settings, User, AlertCircle } from 'lucide-react'

// Mock theme provider for preview
const MockThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const [darkMode, setDarkMode] = React.useState(false)
  
  React.useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  return (
    <ThemeContext.Provider value={{ darkMode, toggleDarkMode: () => setDarkMode(!darkMode) }}>
      {children}
    </ThemeContext.Provider>
  )
}

// Component showcase for preview
const ComponentShowcase = () => {
  const [loading, setLoading] = React.useState(false)
  const [showAlert, setShowAlert] = React.useState(true)
  const [toggleValue, setToggleValue] = React.useState(false)

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-text mb-4">Four Hosts Research - Component Preview</h1>
        <p className="text-text-muted">Showcasing the refactored components and UI library</p>
      </div>

      {/* Button Examples */}
      <Card>
        <CardHeader>
          <CardTitle>Button Components</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            <Button variant="primary" icon={Home}>Primary</Button>
            <Button variant="secondary" icon={Settings}>Secondary</Button>
            <Button variant="success">Success</Button>
            <Button variant="danger">Danger</Button>
            <Button variant="ghost" icon={User}>Ghost</Button>
            <Button variant="primary" loading={loading} onClick={() => setLoading(!loading)}>
              {loading ? 'Loading...' : 'Toggle Loading'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Card Examples */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card variant="default">
          <CardHeader>
            <CardTitle>Default Card</CardTitle>
          </CardHeader>
          <CardContent>
            <p>This is a default card with standard styling.</p>
          </CardContent>
        </Card>

        <Card variant="interactive" className="cursor-pointer">
          <CardHeader>
            <CardTitle>Interactive Card</CardTitle>
          </CardHeader>
          <CardContent>
            <p>This card has hover effects and is interactive.</p>
          </CardContent>
        </Card>

        <Card variant="paradigm" paradigm="dolores">
          <CardHeader>
            <CardTitle>Paradigm Card</CardTitle>
          </CardHeader>
          <CardContent>
            <p>This card uses the Dolores paradigm theme.</p>
          </CardContent>
        </Card>
      </div>

      {/* Loading Spinner Examples */}
      <Card>
        <CardHeader>
          <CardTitle>Loading Spinners</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-8">
            <LoadingSpinner size="sm" variant="primary" text="Small" />
            <LoadingSpinner size="md" variant="secondary" text="Medium" />
            <LoadingSpinner size="lg" variant="primary" text="Large" />
            <LoadingSpinner size="xl" variant="primary" text="Extra Large" />
          </div>
        </CardContent>
      </Card>

      {/* Alert Examples */}
      <div className="space-y-4">
        {showAlert && (
          <Alert variant="info" title="Information" dismissible onDismiss={() => setShowAlert(false)}>
            This is an informational alert that can be dismissed.
          </Alert>
        )}
        <Alert variant="success" title="Success">
          Operation completed successfully!
        </Alert>
        <Alert variant="warning" title="Warning">
          Please review your settings before proceeding.
        </Alert>
        <Alert variant="error" title="Error">
          An error occurred while processing your request.
        </Alert>
      </div>

      {/* Toggle Switch Examples */}
      <Card>
        <CardHeader>
          <CardTitle>Toggle Switches</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <span>Small Toggle:</span>
              <ToggleSwitch size="sm" checked={toggleValue} onChange={setToggleValue} />
            </div>
            <div className="flex items-center gap-4">
              <span>Medium Toggle:</span>
              <ToggleSwitch size="md" checked={!toggleValue} onChange={(val) => setToggleValue(!val)} />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error State Example */}
      <Card className="text-center">
        <CardHeader>
          <div className="flex justify-center mb-4">
            <AlertCircle className="h-16 w-16 text-red-500" />
          </div>
          <CardTitle>Error State Example</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-text-muted mb-6">
            This demonstrates how error states are handled in the application.
          </p>
        </CardContent>
        <CardFooter>
          <div className="flex gap-4 justify-center">
            <Button variant="secondary">Go Back</Button>
            <Button variant="primary">Try Again</Button>
          </div>
        </CardFooter>
      </Card>
    </div>
  )
}

// Main preview app
const App = () => {
  return (
    <MockThemeProvider>
      <AuthProvider>
        <Router>
          <div className="min-h-screen bg-surface transition-colors duration-200">
            <PageTransition>
              <ComponentShowcase />
            </PageTransition>
          </div>
        </Router>
      </AuthProvider>
    </MockThemeProvider>
  )
}

export default App