import { Component } from 'react'
import type { ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch() {
    // Error caught by boundary - log to monitoring service in production
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="min-h-screen flex items-center justify-center bg-surface">
          <div className="max-w-md w-full bg-surface shadow-lg rounded-lg p-8 border border-border">
            <h2 className="text-2xl font-bold text-error mb-4">Something went wrong</h2>
            <p className="text-text-muted mb-4">
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            {this.state.error?.message?.includes('backend') && (
              <div className="bg-surface-subtle border border-border rounded p-4 mb-4">
                <p className="text-sm text-text">
                  The backend server appears to be offline. Please ensure it's running on http://localhost:8000
                </p>
              </div>
            )}
            <button
              onClick={() => window.location.reload()}
              className="btn-primary"
            >
              Reload Page
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
