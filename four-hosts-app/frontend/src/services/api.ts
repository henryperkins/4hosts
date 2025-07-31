import type { ParadigmClassification, ResearchResult, ResearchHistoryItem, UserPreferences, Paradigm, ResearchOptions, User } from '../types'
import { AuthErrorHandler } from './api-auth'

const API_BASE_URL = import.meta.env.VITE_API_URL || ''

// Type definitions for API responses
export interface AuthTokens {
  access_token: string
  token_type: string
  refresh_token?: string
}

export interface SystemStats {
  total_queries: number
  active_research: number
  paradigm_distribution: Record<string, number>
  average_processing_time: number
  cache_hit_rate: number
  system_health: 'healthy' | 'degraded' | 'critical'
}

export interface WSMessage {
  type: string
  data: Record<string, unknown>
  timestamp?: string
}

export interface ResearchStatusResponse {
  research_id: string
  status: string
  paradigm: string
  started_at: string
  progress?: Record<string, unknown>
  cost_info?: Record<string, number>
  message?: string
  can_retry?: boolean
  can_cancel?: boolean
  error?: string
  cancelled_at?: string
  cancelled_by?: string
}

// ... (rest of the interfaces are the same)

class APIService {
  private authToken: string | null = null
  private isRefreshing = false
  private failedQueue: { resolve: (value: string) => void; reject: (reason: Error) => void }[] = []

  constructor() {
    this.authToken = localStorage.getItem('auth_token')
  }

  private processFailedQueue(error: Error | null, token: string | null = null) {
    this.failedQueue.forEach(prom => {
      if (error) {
        prom.reject(error);
      } else if (token) {
        prom.resolve(token);
      }
    });
    this.failedQueue = [];
  }

  private async fetchWithAuth(url: string, options: RequestInit = {}, isRetry = false): Promise<Response> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...(options.headers as Record<string, string> || {}),
    }

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`
    }

    const fullUrl = `${API_BASE_URL}${url}`;
    const response = await fetch(fullUrl, { ...options, headers });

    if (response.status === 401 && !isRetry) {
      if (this.isRefreshing) {
        return new Promise((resolve, reject) => {
          this.failedQueue.push({ resolve, reject });
        })
        .then(() => this.fetchWithAuth(url, options, true));
      }

      this.isRefreshing = true;

      try {
        const newTokens = await this.refreshToken();
        this.processFailedQueue(null, newTokens.access_token);
        return this.fetchWithAuth(url, options, true);
      } catch (error) {
        this.processFailedQueue(error instanceof Error ? error : new Error('Unknown error'), null);
        this.logout(); // Or handle logout more gracefully
        return Promise.reject(error);
      } finally {
        this.isRefreshing = false;
      }
    }

    return response;
  }

  async refreshToken(): Promise<AuthTokens> {
    const refreshToken = AuthErrorHandler.getRefreshToken();
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await this.fetchWithAuth('/auth/refresh', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    }, true); // isRetry = true to prevent infinite loop

    if (!response.ok) {
      throw new Error('Failed to refresh token');
    }

    const tokens = await response.json();
    this.authToken = tokens.access_token;
    AuthErrorHandler.storeAuthTokens(tokens.access_token, tokens.refresh_token);
    return tokens;
  }

  async register(username: string, email: string, password: string): Promise<AuthTokens> {
    const response = await this.fetchWithAuth('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, email, password, role: 'free' }),
    })

    if (!response.ok) {
      // ... (error handling as before)
    }

    const tokens = await response.json()
    this.authToken = tokens.access_token
    AuthErrorHandler.storeAuthTokens(tokens.access_token, tokens.refresh_token)
    return tokens
  }

  async login(email: string, password: string): Promise<AuthTokens> {
    const response = await this.fetchWithAuth('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })

    if (!response.ok) {
      // ... (error handling as before)
    }

    const tokens = await response.json()
    this.authToken = tokens.access_token
    AuthErrorHandler.storeAuthTokens(tokens.access_token, tokens.refresh_token)
    return tokens
  }

  async logout(): Promise<void> {
    const refreshToken = AuthErrorHandler.getRefreshToken()
    try {
      await this.fetchWithAuth('/auth/logout', { 
        method: 'POST',
        body: JSON.stringify({ refresh_token: refreshToken })
      })
    } catch {
      // If logout fails (e.g., 401), we still want to clear local tokens
      console.log('Logout request failed, clearing local tokens anyway')
    }
    this.authToken = null
    AuthErrorHandler.clearAuthTokens()
    this.disconnectWebSocket()
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.fetchWithAuth('/auth/user')
    if (!response.ok) {
      throw new Error('Failed to get current user')
    }
    return await response.json()
  }

  // Research APIs
  async submitResearch(query: string, options: ResearchOptions): Promise<{ research_id: string }> {
    const response = await this.fetchWithAuth('/research/query', {
      method: 'POST',
      body: JSON.stringify({ query, options })
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to submit research')
    }

    return response.json()
  }

  async submitDeepResearch(query: string, paradigm?: Paradigm, searchContextSize?: 'small' | 'medium' | 'large', userLocation?: { country?: string; city?: string }): Promise<{ research_id: string }> {
    const response = await this.fetchWithAuth('/research/deep', {
      method: 'POST',
      body: JSON.stringify({
        query,
        paradigm,
        search_context_size: searchContextSize,
        user_location: userLocation
      })
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to submit deep research')
    }

    return response.json()
  }

  async getResearchStatus(researchId: string): Promise<ResearchStatusResponse> {
    const response = await this.fetchWithAuth(`/research/status/${researchId}`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to get research status')
    }

    return response.json()
  }

  async getResearchResults(researchId: string): Promise<ResearchResult> {
    const response = await this.fetchWithAuth(`/research/results/${researchId}`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to get research results')
    }

    return response.json()
  }

  async cancelResearch(researchId: string): Promise<void> {
    const response = await this.fetchWithAuth(`/research/cancel/${researchId}`, {
      method: 'POST'
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to cancel research')
    }
  }

  async getUserResearchHistory(limit = 10, offset = 0): Promise<ResearchHistoryItem[]> {
    const response = await this.fetchWithAuth(`/research/history?limit=${limit}&offset=${offset}`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to get research history')
    }

    const data = await response.json()
    return data.history || []
  }

  async exportResearch(researchId: string, format: 'pdf' | 'json' | 'csv' = 'pdf'): Promise<Blob> {
    const response = await this.fetchWithAuth(`/research/export/${researchId}?format=${format}`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to export research')
    }

    return response.blob()
  }

  // Paradigm APIs
  async classifyQuery(query: string): Promise<ParadigmClassification> {
    const response = await this.fetchWithAuth('/paradigms/classify', {
      method: 'POST',
      body: JSON.stringify({ query })
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to classify query')
    }

    const data = await response.json()
    return data.classification
  }

  async getParadigmExplanation(paradigm: Paradigm): Promise<Record<string, unknown>> {
    const response = await this.fetchWithAuth(`/paradigms/explanation/${paradigm}`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to get paradigm explanation')
    }

    return response.json()
  }

  // User preference APIs
  async updateUserPreferences(preferences: UserPreferences): Promise<void> {
    const response = await this.fetchWithAuth('/auth/preferences', {
      method: 'PUT',
      body: JSON.stringify({ preferences })
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to update preferences')
    }
  }

  async getUserPreferences(): Promise<UserPreferences> {
    const response = await this.fetchWithAuth('/auth/preferences')

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to get preferences')
    }

    const data = await response.json()
    return data.preferences || {}
  }

  // System Stats APIs
  async getSystemStats(): Promise<SystemStats> {
    const response = await this.fetchWithAuth('/system/stats')

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to get system stats')
    }

    return response.json()
  }

  async getPublicSystemStats(): Promise<Partial<SystemStats>> {
    const response = await this.fetchWithAuth('/system/public-stats')

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to get public system stats')
    }

    return response.json()
  }

  // WebSocket Management
  private wsConnections: Map<string, WebSocket> = new Map()
  private wsCallbacks: Map<string, (message: WSMessage) => void> = new Map()

  connectWebSocket(researchId: string, onMessage: (message: WSMessage) => void): void {
    if (this.wsConnections.has(researchId)) {
      this.disconnectWebSocket(researchId)
    }

    // Use the API_BASE_URL and convert it to WebSocket URL
    const apiUrl = new URL(API_BASE_URL || window.location.origin)
    const wsProtocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${wsProtocol}//${apiUrl.host}/ws/research/${researchId}`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log(`WebSocket connected for research ${researchId}`)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        onMessage(message)
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    ws.onerror = (error) => {
      console.error(`WebSocket error for research ${researchId}:`, error)
    }

    ws.onclose = () => {
      console.log(`WebSocket disconnected for research ${researchId}`)
      this.wsConnections.delete(researchId)
      this.wsCallbacks.delete(researchId)
    }

    this.wsConnections.set(researchId, ws)
    this.wsCallbacks.set(researchId, onMessage)
  }

  disconnectWebSocket(researchId?: string): void {
    if (researchId) {
      const ws = this.wsConnections.get(researchId)
      if (ws) {
        ws.close()
        this.wsConnections.delete(researchId)
        this.wsCallbacks.delete(researchId)
      }
    } else {
      // Disconnect all WebSockets
      this.wsConnections.forEach((ws) => ws.close())
      this.wsConnections.clear()
      this.wsCallbacks.clear()
    }
  }

  unsubscribeFromResearch(researchId: string): void {
    this.disconnectWebSocket(researchId)
  }

  // Feedback API
  async submitFeedback(researchId: string, satisfactionScore: number, paradigmFeedback?: string): Promise<void> {
    const response = await this.fetchWithAuth(`/research/feedback/${researchId}`, {
      method: 'POST',
      body: JSON.stringify({
        satisfaction_score: satisfactionScore,
        paradigm_feedback: paradigmFeedback
      })
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to submit feedback')
    }
  }

  // Deep Research Management
  async resumeDeepResearch(researchId: string): Promise<void> {
    const response = await this.fetchWithAuth(`/research/deep/${researchId}/resume`, {
      method: 'POST'
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to resume deep research')
    }
  }

  async getDeepResearchStatus(): Promise<Record<string, unknown>> {
    const response = await this.fetchWithAuth('/research/deep/status')

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to get deep research status')
    }

    return response.json()
  }

  // Source credibility
  async getSourceCredibility(domain: string, paradigm: Paradigm): Promise<Record<string, unknown>> {
    const response = await this.fetchWithAuth(`/sources/credibility/${domain}?paradigm=${paradigm}`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to get source credibility')
    }

    return response.json()
  }
}

export const api = new APIService()
export default api
