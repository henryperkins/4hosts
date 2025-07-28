const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

// Feature flags from environment (available for conditional feature enabling)
// const ENABLE_WEBSOCKET = import.meta.env.VITE_ENABLE_WEBSOCKET !== 'false'
// const ENABLE_EXPORT = import.meta.env.VITE_ENABLE_EXPORT !== 'false'
// const ENABLE_METRICS = import.meta.env.VITE_ENABLE_METRICS !== 'false'

// Type definitions for API responses
export interface AuthTokens {
  access_token: string
  token_type: string
  refresh_token?: string
}

export interface User {
  id: string
  username: string
  email: string
  created_at: string
  preferences?: Record<string, any>
}

export interface ResearchOptions {
  depth?: 'quick' | 'standard' | 'deep'
  paradigm_override?: string | null
  include_secondary?: boolean
  max_sources?: number
  language?: string
  region?: string
  enable_real_search?: boolean
  enable_ai_classification?: boolean
}

export interface ResearchSubmission {
  research_id: string
  status: string
  paradigm_classification: any
  created_at: string
}

export interface ResearchStatus {
  id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress?: number
  message?: string
  updated_at: string
}

export interface CredibilityScore {
  domain: string
  score: number
  factors: {
    domain_age: number
    reputation: number
    content_quality: number
    transparency: number
  }
  last_updated: string
}

export interface SystemStats {
  total_queries: number
  active_research: number
  paradigm_distribution: Record<string, number>
  average_processing_time: number
  cache_hit_rate?: number
  system_health: 'healthy' | 'degraded' | 'critical'
}

export interface WebSocketMessage {
  type: 'status_update' | 'progress' | 'result' | 'error'
  research_id: string
  data: any
}

class APIService {
  private authToken: string | null = null
  private wsConnection: WebSocket | null = null
  private wsHandlers: Map<string, (message: WebSocketMessage) => void> = new Map()

  constructor() {
    // Load token from localStorage if available
    this.authToken = localStorage.getItem('auth_token')
  }

  // Helper method for authenticated requests
  private async fetchWithAuth(url: string, options: RequestInit = {}): Promise<Response> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    }

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`
    }

    try {
      console.log('API Request:', {
        url: `${API_BASE_URL}${url}`,
        method: options.method || 'GET',
        headers,
        body: options.body,
        bodyParsed: options.body ? JSON.parse(options.body as string) : null
      })

      const response = await fetch(`${API_BASE_URL}${url}`, {
        ...options,
        headers,
      })

      console.log('API Response:', {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok
      })

      return response
    } catch (error) {
      // Handle network errors (backend not running)
      console.error('Network error:', error)
      throw new Error('Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000')
    }
  }

  // Authentication endpoints
  async register(username: string, email: string, password: string): Promise<AuthTokens> {
    const response = await this.fetchWithAuth('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, email, password, role: 'free' }),
    })

    if (!response.ok) {
      let errorMessage = 'Registration failed'
      try {
        const errorText = await response.text()
        console.error('Registration error response text:', errorText)
        try {
          const error = JSON.parse(errorText)
          console.error('Registration error parsed:', error)
          errorMessage = error.detail || error.message || errorMessage
          if (Array.isArray(error.detail)) {
            errorMessage = error.detail[0]?.msg || errorMessage
          }
        } catch (jsonError) {
          console.error('Failed to parse error as JSON:', jsonError)
          errorMessage = errorText || errorMessage
        }
      } catch (e) {
        console.error('Failed to read error response:', e)
      }
      throw new Error(errorMessage)
    }

    const tokens = await response.json()
    this.authToken = tokens.access_token
    localStorage.setItem('auth_token', tokens.access_token)
    return tokens
  }

  async login(email: string, password: string): Promise<AuthTokens> {
    const response = await this.fetchWithAuth('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })

    if (!response.ok) {
      let errorMessage = 'Login failed'
      try {
        const errorText = await response.text()
        console.error('Login error response text:', errorText)
        try {
          const error = JSON.parse(errorText)
          console.error('Login error parsed:', error)
          errorMessage = error.detail || error.message || errorMessage
        } catch (jsonError) {
          console.error('Failed to parse error as JSON:', jsonError)
          errorMessage = errorText || errorMessage
        }
      } catch (e) {
        console.error('Failed to read error response:', e)
      }
      throw new Error(errorMessage)
    }

    const tokens = await response.json()
    this.authToken = tokens.access_token
    localStorage.setItem('auth_token', tokens.access_token)
    return tokens
  }

  async logout(): Promise<void> {
    await this.fetchWithAuth('/auth/logout', { method: 'POST' })
    this.authToken = null
    localStorage.removeItem('auth_token')
    this.disconnectWebSocket()
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.fetchWithAuth('/auth/user')
    
    if (!response.ok) {
      throw new Error('Failed to get user info')
    }

    return response.json()
  }

  async updateUserPreferences(preferences: Record<string, any>): Promise<User> {
    const response = await this.fetchWithAuth('/auth/preferences', {
      method: 'PUT',
      body: JSON.stringify({ preferences }),
    })

    if (!response.ok) {
      throw new Error('Failed to update preferences')
    }

    return response.json()
  }

  // Paradigm classification
  async classifyQuery(query: string): Promise<any> {
    const params = new URLSearchParams({ query })
    const response = await this.fetchWithAuth(`/paradigms/classify?${params}`, {
      method: 'POST',
    })

    if (!response.ok) {
      throw new Error('Failed to classify query')
    }

    return response.json()
  }

  // Research endpoints
  async submitResearch(query: string, options: ResearchOptions = {}): Promise<ResearchSubmission> {
    const response = await this.fetchWithAuth('/research/query', {
      method: 'POST',
      body: JSON.stringify({ query, options }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to submit research')
    }

    return response.json()
  }

  async getResearchStatus(researchId: string): Promise<ResearchStatus> {
    const response = await this.fetchWithAuth(`/research/status/${researchId}`)

    if (!response.ok) {
      throw new Error('Failed to get research status')
    }

    return response.json()
  }

  async getResearchResults(researchId: string): Promise<any> {
    const response = await this.fetchWithAuth(`/research/results/${researchId}`)

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('Research not found or still processing')
      }
      throw new Error('Failed to get research results')
    }

    return response.json()
  }

  async getUserResearchHistory(limit: number = 10, offset: number = 0): Promise<any[]> {
    const response = await this.fetchWithAuth(`/research/history?limit=${limit}&offset=${offset}`)

    if (!response.ok) {
      throw new Error('Failed to get research history')
    }

    return response.json()
  }

  async exportResearch(researchId: string, format: 'json' | 'pdf' | 'markdown' = 'json'): Promise<Blob> {
    const response = await this.fetchWithAuth(`/research/export/${researchId}?format=${format}`)

    if (!response.ok) {
      throw new Error('Failed to export research')
    }

    return response.blob()
  }

  // Source credibility
  async getSourceCredibility(domain: string): Promise<CredibilityScore> {
    const response = await this.fetchWithAuth(`/sources/credibility/${domain}`)

    if (!response.ok) {
      throw new Error('Failed to get credibility score')
    }

    return response.json()
  }

  // System endpoints
  async getSystemStats(): Promise<SystemStats> {
    const response = await this.fetchWithAuth('/system/stats')

    if (!response.ok) {
      throw new Error('Failed to get system stats')
    }

    return response.json()
  }

  async getHealthCheck(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/health`)

    if (!response.ok) {
      throw new Error('Health check failed')
    }

    return response.json()
  }

  async getMetrics(): Promise<string> {
    const response = await this.fetchWithAuth('/metrics')

    if (!response.ok) {
      throw new Error('Failed to get metrics')
    }

    return response.text()
  }

  // WebSocket support for real-time updates
  connectWebSocket(researchId: string, onMessage: (message: WebSocketMessage) => void): void {
    if (this.wsConnection?.readyState === WebSocket.OPEN) {
      // Already connected, just add the handler
      this.wsHandlers.set(researchId, onMessage)
      return
    }

    const wsUrl = API_BASE_URL.replace('http', 'ws')
    const token = this.authToken ? `?token=${this.authToken}` : ''
    this.wsConnection = new WebSocket(`${wsUrl}/ws/research/${researchId}${token}`)

    this.wsConnection.onopen = () => {
      console.log('WebSocket connected')
      // Subscribe to specific research ID
      this.wsConnection?.send(JSON.stringify({
        action: 'subscribe',
        research_id: researchId
      }))
    }

    this.wsConnection.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data)
        const handler = this.wsHandlers.get(message.research_id)
        if (handler) {
          handler(message)
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }

    this.wsConnection.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    this.wsConnection.onclose = () => {
      console.log('WebSocket disconnected')
      this.wsConnection = null
      this.wsHandlers.clear()
    }

    this.wsHandlers.set(researchId, onMessage)
  }

  disconnectWebSocket(): void {
    if (this.wsConnection?.readyState === WebSocket.OPEN) {
      this.wsConnection.close()
    }
    this.wsConnection = null
    this.wsHandlers.clear()
  }

  unsubscribeFromResearch(researchId: string): void {
    this.wsHandlers.delete(researchId)
    if (this.wsConnection?.readyState === WebSocket.OPEN) {
      this.wsConnection.send(JSON.stringify({
        action: 'unsubscribe',
        research_id: researchId
      }))
    }
  }

  // Custom document analysis - NOT IMPLEMENTED IN BACKEND
  // async analyzeDocument(file: File, paradigm?: string): Promise<any> {
  //   throw new Error('Document analysis not yet implemented in backend')
  // }
}

// Export singleton instance
export const api = new APIService()
export default api