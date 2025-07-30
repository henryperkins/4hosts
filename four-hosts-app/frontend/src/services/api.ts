import type { ParadigmClassification, ResearchResult, ResearchHistoryItem, UserPreferences } from '../types'

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
  preferences?: Record<string, unknown>
}

export interface ResearchOptions {
  depth?: 'quick' | 'standard' | 'deep' | 'deep_research'
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
  paradigm_classification: ParadigmClassification
  created_at?: string
  estimated_completion?: string
  websocket_url?: string
}

export interface ResearchStatus {
  id: string
  research_id?: string
  status: 'pending' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
  paradigm?: string
  started_at?: string
  progress?: any
  cost_info?: any
  error?: string
  message?: string
  updated_at?: string
  can_cancel?: boolean
  can_retry?: boolean
  cancelled_at?: string
  cancelled_by?: string
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
  type: 'status_update' | 'progress' | 'result' | 'error' | 'connected' | 'system_notification' | 'research_started' | 'research_phase_change' | 'research_progress' | 'source_found' | 'source_analyzed' | 'research_completed' | 'research_failed' | 'search.started' | 'search.completed' | 'credibility.check' | 'deduplication.progress'
  research_id?: string
  data: unknown
  id?: string
  timestamp?: string
  metadata?: Record<string, any>
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
      Accept: 'application/json',
      ...(options.headers as Record<string, string> || {}),
    }

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`
    }

    try {
      const fullUrl = `${API_BASE_URL}${url}`;

      // Skip detailed logging for GET requests to reduce noise
      if (options.method !== 'GET') {
        console.log('API Request:', {
          url: fullUrl,
          method: options.method || 'GET',
          headers: { ...headers, Authorization: this.authToken ? '[TOKEN]' : '[NONE]' },
        })
      }

      // Add timeout for network requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      const response = await fetch(fullUrl, {
        ...options,
        headers,
        signal: controller.signal,
      }).finally(() => clearTimeout(timeoutId));

      // Log response for non-GET requests
      if (options.method !== 'GET' || !response.ok) {
        console.log('API Response:', {
          status: response.status,
          statusText: response.statusText,
          ok: response.ok,
          url: fullUrl,
        })
      }

      return response
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Request timeout - backend may be slow');
      }

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

      // Handle network-level errors (backend not running)
      if (response.status === 0 || response.status === 503) {
        throw new Error('Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000');
      }

      try {
        const errorData = await response.json();
        if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail[0]?.msg || errorMessage;
        } else {
          errorMessage = errorData.detail || errorData.message || errorMessage;
        }
      } catch {
        errorMessage = response.statusText || 'Registration request failed';
      }

      throw new Error(errorMessage);
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
      let errorMessage = 'Login failed';

      // Handle network-level errors (backend not running)
      if (response.status === 0 || response.status === 503) {
        throw new Error('Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000');
      }

      // Handle authentication errors
      if (response.status === 401) {
        throw new Error('Invalid email or password');
      }

      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorData.message || 'Login failed';
      } catch {
        errorMessage = response.statusText || 'Login request failed';
      }

      throw new Error(errorMessage);
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

  async updateUserPreferences(preferences: UserPreferences): Promise<User> {
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
  async classifyQuery(query: string): Promise<ParadigmClassification> {
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

    const data = await response.json()
    // Add created_at timestamp if missing from backend
    if (!data.created_at) {
      data.created_at = new Date().toISOString()
    }
    return data
  }

  async getResearchStatus(researchId: string): Promise<ResearchStatus> {
    const response = await this.fetchWithAuth(`/research/status/${researchId}`)

    if (!response.ok) {
      throw new Error('Failed to get research status')
    }

    const data = await response.json()
    // Map backend fields to frontend expectations
    return {
      id: data.research_id || researchId,
      research_id: data.research_id,
      status: data.status,
      paradigm: data.paradigm,
      started_at: data.started_at,
      progress: data.progress,
      cost_info: data.cost_info,
      error: data.error,
      message: data.message,
      updated_at: data.updated_at || data.started_at || new Date().toISOString(),
      can_cancel: data.can_retry,
      can_retry: data.can_retry,
      cancelled_at: data.cancelled_at,
      cancelled_by: data.cancelled_by
    }
  }

  async getResearchResults(researchId: string): Promise<ResearchResult> {
    const response = await this.fetchWithAuth(`/research/results/${researchId}`)

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('Research not found or still processing')
      }
      throw new Error('Failed to get research results')
    }

    return response.json()
  }

  async submitResearchFeedback(researchId: string, satisfactionScore: number, paradigmFeedback?: string): Promise<any> {
    const response = await this.fetchWithAuth(`/research/feedback/${researchId}`, {
      method: 'POST',
      body: JSON.stringify({
        satisfaction_score: satisfactionScore,
        paradigm_feedback: paradigmFeedback
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to submit feedback')
    }

    return response.json()
  }

  async cancelResearch(researchId: string): Promise<{ message: string; cancelled: boolean; cancelled_at?: string; status?: string; research_id?: string }> {
    const response = await this.fetchWithAuth(`/research/cancel/${researchId}`, {
      method: 'POST',
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to cancel research')
    }

    return response.json()
  }

  async getUserResearchHistory(limit: number = 10, offset: number = 0): Promise<ResearchHistoryItem[]> {
    const response = await this.fetchWithAuth(`/research/history?limit=${limit}&offset=${offset}`)

    if (!response.ok) {
      throw new Error('Failed to get research history')
    }

    const data = await response.json()
    return data.history || []
  }

  async exportResearch(researchId: string, format: 'json' | 'pdf' | 'markdown' = 'json'): Promise<Blob> {
    const response = await this.fetchWithAuth(`/research/export/${researchId}?format=${format}`)

    if (!response.ok) {
      throw new Error('Failed to export research')
    }

    return response.blob()
  }

  // Deep Research endpoints
  async submitDeepResearch(
    query: string,
    paradigm?: string,
    searchContextSize?: string,
    userLocation?: { country?: string; city?: string; region?: string; timezone?: string }
  ): Promise<ResearchSubmission> {
    const response = await this.fetchWithAuth('/research/deep', {
      method: 'POST',
      body: JSON.stringify({
        query,
        paradigm,
        search_context_size: searchContextSize,
        user_location: userLocation
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to submit deep research')
    }

    const data = await response.json()
    if (!data.created_at) {
      data.created_at = new Date().toISOString()
    }
    return data
  }

  async getDeepResearchStatus(): Promise<{ total: number; deep_research_queries: any[] }> {
    const response = await this.fetchWithAuth('/research/deep/status')

    if (!response.ok) {
      throw new Error('Failed to get deep research status')
    }

    return response.json()
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
      const error = await response.json()
      const errorObject = new Error(error.detail || 'Failed to get system stats')

      // Add status code to error object for more granular error handling
      Object.defineProperty(errorObject, 'status', {
        value: response.status,
        writable: true,
        enumerable: true,
        configurable: true
      })

      throw errorObject
    }

    return response.json()
  }

  async getPublicSystemStats(): Promise<SystemStats> {
    const response = await this.fetchWithAuth('/system/public-stats')

    if (!response.ok) {
      throw new Error('Failed to get public system stats')
    }

    return response.json()
  }

  async getHealthCheck(): Promise<Record<string, unknown>> {
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

  // Admin endpoints for enhanced features
  async adminForceParadigmSwitch(queryId: string, newParadigm: string, reason: string): Promise<any> {
    const response = await this.fetchWithAuth('/admin/paradigm/force-switch', {
      method: 'POST',
      body: JSON.stringify({
        query_id: queryId,
        new_paradigm: newParadigm,
        reason: reason
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to force paradigm switch')
    }

    return response.json()
  }

  async adminTriggerMLRetraining(): Promise<any> {
    const response = await this.fetchWithAuth('/admin/ml/retrain', {
      method: 'POST',
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to trigger ML retraining')
    }

    return response.json()
  }

  async adminGetSystemHealth(): Promise<any> {
    const response = await this.fetchWithAuth('/admin/system/health')

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to get system health')
    }

    return response.json()
  }

  // WebSocket support for real-time updates
  connectWebSocket(researchId: string, onMessage: (message: WebSocketMessage) => void): void {
    if (this.wsConnection?.readyState === WebSocket.OPEN) {
      // Already connected, just add the handler and subscribe to research
      this.wsHandlers.set(researchId, onMessage)
      this.wsConnection.send(JSON.stringify({
        type: 'subscribe',
        research_id: researchId
      }))
      return
    }

    // If API_BASE_URL is an absolute URL (starts with http/https) convert it to ws/wss.
    // Otherwise (relative like "/api"), leave blank so the client connects to same origin
    // and lets the Vite proxy `/ws` rule forward the request.
    const wsUrl = API_BASE_URL.startsWith('http')
      ? API_BASE_URL.replace(/^http/, 'ws').replace('/api', '')
      : ''
    const token = this.authToken ? `?token=${this.authToken}` : ''
    this.wsConnection = new WebSocket(`${wsUrl}/ws${token}`)

    this.wsConnection.onopen = () => {
      console.log('WebSocket connected')
      // Send subscription message for the research ID
      this.wsConnection!.send(JSON.stringify({
        type: 'subscribe',
        research_id: researchId
      }))
    }

    // Add the handler to the map before connection is established
    this.wsHandlers.set(researchId, onMessage)

    this.wsConnection.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data)
        // Extract research_id from data if not at top level
        const researchIdToUse = message.research_id || (message.data as any)?.research_id || researchId

        // Ensure research_id is present for handler routing
        const messageWithId = {
          ...message,
          research_id: researchIdToUse
        }

        const handler = this.wsHandlers.get(researchIdToUse)
        if (handler) {
          handler(messageWithId)
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
        type: 'unsubscribe',
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
