import type {
  ParadigmClassification,
  ResearchResult,
  ResearchHistoryItem,
  UserPreferences,
  Paradigm,
  ResearchOptions,
  User,
  GeneratedAnswer,
  AnswerSection,
  ActionItem,
  Citation,
} from '../types'
import { CSRFProtection } from './csrf-protection'
import type {
  AuthTokenResponse,
  LoginResponse,
  MetricsData,
  WebSocketMessage,
  ResearchResponse,
  TriageBoardSnapshot,
  TelemetrySummary
} from '../types/api-types'
import type { ExtendedStatsSnapshot } from '../types/api-types'
import { isExtendedStatsSnapshot, ResearchResponseSchema } from '../types/api-types'
import {
  isErrorResponse
} from '../types/api-types'
import {
  validateLoginResponse,
  validateWebSocketMessage,
  validateMetricsData
} from '../utils/validation'
import { isValidParadigm } from '../constants/paradigm'
type AuthStore = typeof import('../store/authStore')['useAuthStore']

const API_BASE_URL = import.meta.env.VITE_API_URL || '' // keep empty to use Vite proxy-relative paths in dev
const API_PREFIX = '/v1'

let authStorePromise: Promise<AuthStore> | null = null

function canUseAuthStore(): boolean {
  if (typeof window === 'undefined') {
    return false
  }
  try {
    // Accessing sessionStorage can throw (Safari private mode), so probe inside try/catch
    return typeof window.sessionStorage !== 'undefined'
  } catch {
    // Even if storage is blocked, we still want the auth store for in-memory fallback
    return true
  }
}

async function getAuthStore(): Promise<AuthStore | null> {
  if (!canUseAuthStore()) {
    return null
  }
  if (!authStorePromise) {
    authStorePromise = import('../store/authStore').then(mod => mod.useAuthStore)
  }
  try {
    return await authStorePromise
  } catch (error) {
    authStorePromise = null
    throw error
  }
}

function normalizePath(url: string): string {
  try {
    if (!url || url.startsWith(API_PREFIX)) return url
    const needsPrefix = [/^\/auth\//, /^\/research\//, /^\/paradigms\//, /^\/users\//, /^\/search\//, /^\/sources\//, /^\/system\//, /^\/webhooks\//, /^\/feedback\//]
      .some((re) => re.test(url))
    return needsPrefix ? `${API_PREFIX}${url}` : url
  } catch {
    return url
  }
}

const DEFAULT_PARADIGM: Paradigm = 'bernard'

function coerceParadigm(value: unknown): Paradigm {
  return typeof value === 'string' && isValidParadigm(value) ? value : DEFAULT_PARADIGM
}

function ensureStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is string => typeof item === 'string' && item.length > 0)
}

function normalizeAnswerSection(section: unknown): AnswerSection {
  const record = (section && typeof section === 'object') ? section as Record<string, unknown> : {}
  return {
    title: typeof record.title === 'string' ? record.title : '',
    paradigm: coerceParadigm(record.paradigm),
    content: typeof record.content === 'string' ? record.content : '',
    confidence: typeof record.confidence === 'number' ? record.confidence : 0,
    sources_count: typeof record.sources_count === 'number' ? record.sources_count : 0,
    citations: ensureStringArray(record.citations),
    key_insights: ensureStringArray(record.key_insights),
  }
}

function normalizeActionItem(item: unknown): ActionItem {
  const record = (item && typeof item === 'object') ? item as Record<string, unknown> : {}
  return {
    priority: typeof record.priority === 'string' && record.priority ? record.priority : 'low',
    action: typeof record.action === 'string' ? record.action : '',
    timeframe: typeof record.timeframe === 'string' ? record.timeframe : '',
    paradigm: coerceParadigm(record.paradigm),
    owner: typeof record.owner === 'string' && record.owner.trim() ? record.owner : undefined,
    due_date: typeof record.due_date === 'string' && record.due_date.trim() ? record.due_date : undefined,
  }
}

function normalizeCitation(item: unknown): Citation {
  const record = (item && typeof item === 'object') ? item as Record<string, unknown> : {}
  const url = typeof record.url === 'string' ? record.url : ''
  const idSource = typeof record.id === 'string' && record.id
    ? record.id
    : typeof record.source === 'string' && record.source
      ? `${record.source}-${Math.random().toString(36).slice(2, 10)}`
      : `citation-${Math.random().toString(36).slice(2, 10)}`
  return {
    id: idSource,
    source: typeof record.source === 'string' ? record.source : '',
    title: typeof record.title === 'string' ? record.title : (url || 'Untitled source'),
    url,
    credibility_score: typeof record.credibility_score === 'number' ? record.credibility_score : 0,
    paradigm_alignment: coerceParadigm(record.paradigm_alignment),
  }
}

// Re-export for backward compatibility
export type AuthTokens = AuthTokenResponse
export type SystemStats = MetricsData
export type WSMessage = WebSocketMessage

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
  // Single-flight refresh via shared promise to prevent stampedes
  private refreshPromise: Promise<void> | null = null

  constructor() {
    // Proactive cleanup of any open WebSockets when the page is closed or navigated away
    if (typeof window !== 'undefined' && window.addEventListener) {
      window.addEventListener('beforeunload', () => {
        try {
          this.disconnectWebSocket()
        } catch {
          // ignore cleanup failures
        }
      })
    }
  }

  // Deprecated queue mechanism removed in favor of shared refreshPromise

  private async fetchWithAuth(url: string, options: RequestInit = {}, isRetry = false, csrfRetry = false): Promise<Response> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...(options.headers as Record<string, string> || {}),
    }

    // Remove any Authorization header since we're using cookies
    delete headers['Authorization']

    // Add CSRF token for state-changing requests
    if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(options.method || 'GET')) {
      const csrfToken = await CSRFProtection.getToken()
      if (csrfToken) {
        headers['X-CSRF-Token'] = csrfToken
      } else {
        // Fallback: attempt to fetch CSRF then retry once with the token
        if (!isRetry) {
          try {
            await CSRFProtection.getToken(true)
            return this.fetchWithAuth(url, options, true, true)
          } catch {
            // Continue without CSRF token for non-protected endpoints
          }
        }
      }
    }

    // Use relative path when API_BASE_URL is empty to leverage Vite proxy and same-origin cookies
    const path = normalizePath(url)
    const fullUrl = API_BASE_URL ? `${API_BASE_URL}${path}` : path;
    const response = await fetch(fullUrl, {
      ...options,
      headers,
      // Ensure CORS includes credentials and Vite proxy preserves cookies
      credentials: 'include'
    });

    let authStore: AuthStore | null = null

    // Handle CSRF token mismatch
    if (response.status === 403 && !csrfRetry) {
      try {
        const errorData = await response.json();
        if (errorData.detail === 'CSRF token mismatch') {
          console.log('CSRF token mismatch detected, refreshing token...');
          CSRFProtection.clearToken();
          await CSRFProtection.getToken(true); // Force refresh
          return this.fetchWithAuth(url, options, isRetry, true);
        }
      } catch {
        // If we can't parse the error, continue with normal flow
      }
    }

    if (response.status === 401) {
      // Determine if this is an auth bootstrap endpoint or we are clearly unauthenticated
      const lowerPath = path.toLowerCase()
      const isAuthBootstrap = [
        '/v1/auth/login', '/auth/login',
        '/v1/auth/register', '/auth/register',
        '/v1/auth/user', '/auth/user',
        '/v1/auth/refresh', '/auth/refresh',
        '/v1/api/session/create', '/api/session/create'
      ].some(p => lowerPath === p)
      authStore = await getAuthStore()
      const isAuthed = authStore?.getState().isAuthenticated ?? false

      if (!isAuthed && isAuthBootstrap) {
        // Do not attempt refresh loops for initial unauthenticated requests
        return response
      }

      console.log(`401 error for ${url}, isRetry: ${isRetry}`);

      if (isRetry) {
        // Already retried after refresh, authentication failed
        console.error('Authentication failed after token refresh');
        this.logout();
        const errorMessage = 'Authentication required. Please log in again.';
        return Promise.reject(new Error(errorMessage));
      }

      // Single-flight refresh logic
      try {
        if (!this.refreshPromise) {
          console.log('Starting token refresh')
          this.refreshPromise = this.refreshToken()
        } else {
          console.log('Token refresh already in progress, awaiting...')
        }
        await this.refreshPromise
        console.log('Token refresh successful, retrying request')
        // Add small delay to ensure cookies are set
        await new Promise(resolve => setTimeout(resolve, 100))
        // After successful refresh, clear CSRF token to force refresh on next request
        CSRFProtection.clearToken()
        return this.fetchWithAuth(url, options, true)
      } catch (error) {
        console.error('Token refresh failed:', error)
        // Clear auth state on token refresh failure
        const store = authStore ?? await getAuthStore()
        store?.getState().reset()
        return Promise.reject(error)
      } finally {
        this.refreshPromise = null
      }
    }

    return response;
  }

  async refreshToken(): Promise<void> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      'X-CSRF-Token': await CSRFProtection.getToken()
    }
    const path = normalizePath('/auth/refresh')
    const fullUrl = API_BASE_URL ? `${API_BASE_URL}${path}` : path;
    const response = await fetch(fullUrl, {
      method: 'POST',
      headers,
      credentials: 'include' // Refresh token in httpOnly cookie
    });

    if (!response.ok) {
      // surface more helpful message
      try {
        const err = await response.json()
        throw new Error(err.detail || 'Failed to refresh token')
      } catch {
        throw new Error('Failed to refresh token')
      }
    }

    // Backend sets new tokens as httpOnly cookies
    // No need to handle response data
  }

  async register(username: string, email: string, password: string): Promise<AuthTokenResponse> {
    const response = await this.fetchWithAuth('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, email, password, role: 'free' }),
    })

    if (!response.ok) {
      const errorData = await response.json()
      if (isErrorResponse(errorData)) {
        const message = typeof errorData.detail === 'string'
          ? errorData.detail
          : errorData.error
        throw new Error(message)
      }
      throw new Error('Registration failed')
    }

    const tokens = await response.json()
    // Tokens now stored as httpOnly cookies by backend
    return tokens
  }

  async login(emailOrUsername: string, password: string): Promise<LoginResponse> {
    // Email-only login enforced by backend
    const isEmail = emailOrUsername.includes('@')
    if (!isEmail) {
      throw new Error('Please use your email address to login.')
    }

    const response = await this.fetchWithAuth('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email: emailOrUsername, password }),
    })

    if (!response.ok) {
      let message = 'Login failed'
      try {
        const err = await response.json()
        if (isErrorResponse(err)) {
          const detail = typeof err.detail === 'string' ? err.detail : err.error
          if (response.status === 401 || (detail && /Invalid credentials/i.test(detail))) {
            message = 'Invalid credentials'
          } else if (response.status === 503) {
            message = 'Service unavailable. Please try again later.'
          } else if (detail) {
            message = detail
          }
        }
      } catch {
        // keep default
      }
      throw new Error(message)
    }

    const loginData = await response.json()
    // Validate response at runtime
    return validateLoginResponse(loginData)
  }

  async logout(): Promise<void> {
    try {
      // Don't use fetchWithAuth for logout to avoid infinite loops
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      }

      // Add CSRF token if available
      const csrfToken = await CSRFProtection.getToken()
      if (csrfToken) {
        headers['X-CSRF-Token'] = csrfToken
      }

      const path = normalizePath('/auth/logout')
      const fullUrl = API_BASE_URL ? `${API_BASE_URL}${path}` : path;
      await fetch(fullUrl, {
        method: 'POST',
        headers,
        credentials: 'include'
      })
    } catch {
      // If logout fails (e.g., 401), we still want to clear session
      // Logout request failed, but cookies will expire
    }

    // Clear CSRF token on logout
    CSRFProtection.clearToken()
    this.disconnectWebSocket()

    // Clear auth store state to ensure frontend/backend sync
    const store = await getAuthStore()
    store?.getState().reset()
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.fetchWithAuth('/auth/user')
    if (!response.ok) {
      const errorData = await response.json().catch(() => null)
      const message = errorData?.detail || 'Failed to get current user'
      throw new Error(message)
    }
    const userData = await response.json()

    // Ensure the user data has the expected shape
    if (!userData.id || !userData.email) {
      throw new Error('Invalid user data received from server')
    }

    return userData
  }

  // Research APIs
  async submitResearch(query: string, options: ResearchOptions): Promise<{ research_id: string; paradigm_classification?: ParadigmClassification; status?: string; estimated_completion?: string; websocket_url?: string }> {
    const response = await this.fetchWithAuth('/research/query', {
      method: 'POST',
      body: JSON.stringify({ query, options })
    })

    if (!response.ok) {
      let message = 'Failed to submit research'
      try {
        const error = await response.json()
        const detail = (error && error.detail) || (error && error.error)
        if (Array.isArray(detail) && detail.length > 0) {
          message = String(detail[0])
        } else if (typeof detail === 'string' && detail.trim()) {
          message = detail
        }
      } catch {
        // Failed to parse error response
      }
      throw new Error(message)
    }

    return response.json()
  }

  async submitDeepResearch(query: string, paradigm?: Paradigm, searchContextSize?: 'small' | 'medium' | 'large', userLocation?: { country?: string; city?: string }): Promise<{ research_id: string; paradigm_classification?: ParadigmClassification; status?: string; estimated_completion?: string; websocket_url?: string }> {
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
      let message = 'Failed to submit deep research'
      try {
        const error = await response.json()
        const detail = (error && error.detail) || (error && error.error)
        if (Array.isArray(detail) && detail.length > 0) {
          message = String(detail[0])
        } else if (typeof detail === 'string' && detail.trim()) {
          message = detail
        }
      } catch {
        // Failed to parse error response
      }
      throw new Error(message)
    }

    const data = await response.json()
    return data
  }

  async getResearchStatus(researchId: string): Promise<ResearchStatusResponse> {
    const safeResearchId = encodeURIComponent(researchId)
    const response = await this.fetchWithAuth(`/research/status/${safeResearchId}`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to get research status')
    }

    const data = await response.json()
    return data
  }

  async getResearchResults(researchId: string): Promise<ResearchResult> {
    const safeResearchId = encodeURIComponent(researchId)
    const response = await this.fetchWithAuth(`/research/results/${safeResearchId}`)

    if (!response.ok) {
      let message = 'Failed to get research results'
      try {
        const error = await response.json()
        const detail = typeof error.detail === 'string' ? error.detail : undefined
        if (detail) {
          message = detail
        }
      } catch {
        // ignore parse errors
      }
      const err = new Error(message) as Error & { status?: number }
      err.status = response.status
      throw err
    }

    const data = await response.json()

    const isStaged =
      data && typeof data === 'object' &&
      typeof data.status === 'string' &&
      (!data.answer || !data.sources)

    if (isStaged) {
      return data as ResearchResult
    }

    const parsed = ResearchResponseSchema.safeParse(data)
    if (!parsed.success) {
      throw new Error(`Invalid research result shape from server: ${parsed.error.message}`)
    }

    const normalized: ResearchResponse = parsed.data

    if (!normalized.results.length && normalized.sources.length) {
      normalized.results = normalized.sources
    }

    if (normalized.answer) {
      const sections: AnswerSection[] = Array.isArray(normalized.answer.sections)
        ? normalized.answer.sections.map(normalizeAnswerSection)
        : []
      const actionItems: ActionItem[] = Array.isArray(normalized.answer.action_items)
        ? normalized.answer.action_items.map(normalizeActionItem)
        : []
      const citations: Citation[] = Array.isArray(normalized.answer.citations)
        ? normalized.answer.citations.map(normalizeCitation)
        : []

      normalized.answer = {
        summary: typeof normalized.answer.summary === 'string' ? normalized.answer.summary : '',
        sections,
        action_items: actionItems,
        citations,
        confidence_score: typeof normalized.answer.confidence_score === 'number'
          ? normalized.answer.confidence_score
          : undefined,
        metadata: normalized.answer.metadata ?? {},
      }
    } else {
      const fallbackAnswer: GeneratedAnswer = {
        summary: '',
        sections: [],
        action_items: [],
        citations: [],
        metadata: {},
      }
      normalized.answer = fallbackAnswer
    }

    if (normalized.integrated_synthesis && normalized.integrated_synthesis.secondary_perspective) {
      normalized.integrated_synthesis.secondary_perspective = normalizeAnswerSection(
        normalized.integrated_synthesis.secondary_perspective
      )
    }

    const meta = normalized.metadata
    meta.total_sources_analyzed = meta.total_sources_analyzed || normalized.sources.length
    meta.high_quality_sources = meta.high_quality_sources || 0

    normalized.cost_info = normalized.cost_info || {}

    return normalized as unknown as ResearchResult
  }

  async cancelResearch(researchId: string): Promise<void> {
    const safeResearchId = encodeURIComponent(researchId)
    const response = await this.fetchWithAuth(`/research/cancel/${safeResearchId}`, {
      method: 'POST'
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to cancel research')
    }
  }

  /* ------------------------------------------------------------------
   * Paradigm override – post-submission switch of paradigm.
   * ------------------------------------------------------------------ */

  async overrideParadigm(researchId: string, paradigm: Paradigm): Promise<void> {
    const response = await this.fetchWithAuth('/paradigms/override', {
      method: 'POST',
      body: JSON.stringify({ research_id: researchId, paradigm })
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(
        typeof error.detail === 'string' ? error.detail : 'Failed to override paradigm'
      )
    }
  }

  /* ------------------------------------------------------------------
   * Webhook management
   * ------------------------------------------------------------------ */

  async listWebhooks(): Promise<{ id: string; url: string; event: string }[]> {
    const response = await this.fetchWithAuth('/webhooks')
    if (!response.ok) {
      throw new Error('Failed to fetch webhooks')
    }
    return response.json()
  }

  async createWebhook(url: string, event: string): Promise<void> {
    const response = await this.fetchWithAuth('/webhooks', {
      method: 'POST',
      body: JSON.stringify({ url, event })
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(
        typeof error.detail === 'string' ? error.detail : 'Failed to create webhook'
      )
    }
  }

  async deleteWebhook(id: string): Promise<void> {
    const response = await this.fetchWithAuth(`/webhooks/${id}`, {
      method: 'DELETE'
    })
    if (!response.ok) {
      throw new Error('Failed to delete webhook')
    }
  }

  async getUserResearchHistory(limit = 10, offset = 0): Promise<{ history: ResearchHistoryItem[]; total: number; limit: number; offset: number }> {
    const response = await this.fetchWithAuth(`/research/history?limit=${limit}&offset=${offset}`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to get research history')
    }

    const data = await response.json()
    return {
      history: data.history || [],
      total: typeof data.total === 'number' ? data.total : (data.history ? data.history.length : 0),
      limit: typeof data.limit === 'number' ? data.limit : limit,
      offset: typeof data.offset === 'number' ? data.offset : offset
    }
  }

  async exportResearch(researchId: string, format: 'pdf' | 'json' | 'csv' | 'markdown' | 'excel' = 'pdf'): Promise<Blob> {
    const safeResearchId = encodeURIComponent(researchId)
    const response = await this.fetchWithAuth(`/v1/export/research/${safeResearchId}`, {
      method: 'POST',
      body: JSON.stringify({ format }),
      headers: {
        Accept: 'application/octet-stream'
      }
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to export research')
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
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to classify query')
    }

    const data = await response.json()
    const classification: ParadigmClassification = data.classification
    // Attach optional signals if present
    if (data.signals && typeof data.signals === 'object') {
      classification.signals = data.signals as ParadigmClassification['signals']
    }
    return classification
  }

  async getParadigmExplanation(paradigm: Paradigm): Promise<Record<string, unknown>> {
    const response = await this.fetchWithAuth(`/paradigms/explanation/${paradigm}`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to get paradigm explanation')
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
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to update preferences')
    }
  }

  async getUserPreferences(): Promise<UserPreferences> {
    const response = await this.fetchWithAuth('/auth/preferences')

    if (!response.ok) {
      const error = await response.json()
      if (isErrorResponse(error)) {
        const message = typeof error.detail === 'string' ? error.detail : error.error
        throw new Error(message || 'Failed to get preferences')
      }
      throw new Error('Failed to get preferences')
    }

    const data = await response.json()
    return data.preferences || {}
  }

  // System Stats APIs
  async getSystemStats(): Promise<MetricsData> {
    const response = await this.fetchWithAuth('/system/stats')

    if (!response.ok) {
      const error = await response.json().catch(() => null)
      if (isErrorResponse(error)) {
        const message = typeof error.detail === 'string' ? error.detail : error.error
        throw new Error(message || 'Failed to get system stats')
      }
      throw new Error('Failed to get system stats')
    }

    // Map backend keys if needed (system_status -> system_health)
    const data = await response.json()
    if (data && typeof data === 'object') {
      if ('system_status' in data && !('system_health' in data)) {
        data.system_health = data.system_status
      }
    }
    return validateMetricsData(data)
  }

  async getSystemStatsSafe(): Promise<MetricsData | Partial<MetricsData>> {
    try {
      return await this.getSystemStats();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes('Insufficient permissions') || msg.includes('403')) {
        return this.getPublicSystemStats();
      }
      // Some fetch implementations won’t include the message; try explicit status handling
      try {
        return this.getPublicSystemStats();
      } catch {
        // Fall through
      }
      throw e;
    }
  }

  // Extended Stats (richer latency, fallbacks, usage)
  async getExtendedStats(): Promise<ExtendedStatsSnapshot> {
    const response = await this.fetchWithAuth('/system/extended-stats')
    if (!response.ok) {
      const error = await response.json().catch(() => null)
      if (error && typeof error === 'object' && 'error' in error) {
        const msg = (error as { error?: string }).error
        throw new Error(msg || 'Failed to get extended stats')
      }
      throw new Error('Failed to get extended stats')
    }
    const data = await response.json()
    if (isExtendedStatsSnapshot(data)) {
      return data
    }
    // Basic structural fallback
    return {
      latency: {},
      fallback_rates: {},
      llm_usage: {},
      paradigm_distribution: {},
      quality: {
        critic_avg_score: 0,
        hallucination_rate: 0,
        evidence_coverage_ratio: 0
      },
      counters: {},
      timestamp: new Date().toISOString()
    }
  }

  async getExtendedStatsSafe(): Promise<ExtendedStatsSnapshot | null> {
    try {
      return await this.getExtendedStats()
    } catch (e) {
      console.warn('Extended stats unavailable', e)
      return null
    }
  }

  // Context Metrics (W‑S‑C‑I)
  async getContextMetrics(): Promise<Record<string, unknown>> {
    const response = await this.fetchWithAuth('/system/context-metrics')
    if (!response.ok) {
      const error = await response.json().catch(() => null)
      throw new Error((error && (error.detail || error.error)) || 'Failed to get context metrics')
    }
    return response.json()
  }

  async getPublicSystemStats(): Promise<Partial<MetricsData>> {
    const response = await this.fetchWithAuth('/system/public-stats')

    if (!response.ok) {
      const error = await response.json().catch(() => null)
      if (isErrorResponse(error)) {
        const message = typeof error.detail === 'string' ? error.detail : error.error
        throw new Error(message || 'Failed to get public system stats')
      }
      throw new Error('Failed to get public system stats')
    }

    const data = await response.json()
    if (data && typeof data === 'object') {
      if ('system_status' in data && !('system_health' in data)) {
        data.system_health = data.system_status
      }
    }
    return data as Partial<MetricsData>
  }

  async getTriageBoard(): Promise<TriageBoardSnapshot> {
    const response = await this.fetchWithAuth('/system/triage-board')

    if (!response.ok) {
      const error = await response.json().catch(() => null)
      if (isErrorResponse(error)) {
        const message = typeof error.detail === 'string' ? error.detail : error.error
        throw new Error(message || 'Failed to load triage board')
      }
      throw new Error('Failed to load triage board')
    }

    const data = await response.json()
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid triage board response')
    }

    const lanes = (data as TriageBoardSnapshot).lanes || {}
    const hydrated: TriageBoardSnapshot = {
      updated_at: typeof (data as TriageBoardSnapshot).updated_at === 'string' ? (data as TriageBoardSnapshot).updated_at : new Date().toISOString(),
      entry_count: typeof (data as TriageBoardSnapshot).entry_count === 'number' ? (data as TriageBoardSnapshot).entry_count : Object.values(lanes).reduce((acc, list) => acc + (Array.isArray(list) ? list.length : 0), 0),
      lanes,
    }
    return hydrated
  }

  async getTelemetrySummary(limit = 50): Promise<TelemetrySummary> {
    const safeLimit = Math.max(1, Math.min(limit, 500))
    const response = await this.fetchWithAuth(`/system/telemetry/summary?limit=${safeLimit}`)

    if (!response.ok) {
      const error = await response.json().catch(() => null)
      if (isErrorResponse(error)) {
        const message = typeof error.detail === 'string' ? error.detail : error.error
        throw new Error(message || 'Failed to load telemetry summary')
      }
      throw new Error('Failed to load telemetry summary')
    }

    const data = await response.json()
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid telemetry summary response')
    }

    const summary = data as TelemetrySummary
    summary.providers = summary.providers || { costs: {}, usage: {} }
    summary.coverage = summary.coverage || { avg_grounding: 0, avg_evidence_quotes: 0, avg_evidence_documents: 0 }
    summary.agent_loop = summary.agent_loop || { avg_iterations: 0, avg_new_queries: 0 }
    summary.stages = summary.stages || {}
    summary.paradigms = summary.paradigms || {}
    summary.depths = summary.depths || {}
    summary.recent_events = summary.recent_events || []
    return summary
  }

  // WebSocket Management
  private wsConnections: Map<string, WebSocket> = new Map()
  private wsCallbacks: Map<string, (message: WSMessage) => void> = new Map()
  private wsReconnectFlags: Map<string, boolean> = new Map()
  private wsReconnectTimers: Map<string, ReturnType<typeof setTimeout>> = new Map()

  connectWebSocket(researchId: string, onMessage: (message: WSMessage) => void): void {
    if (this.wsConnections.has(researchId)) {
      this.disconnectWebSocket(researchId)
    }

    // Mark this socket as eligible for reconnection
    this.wsReconnectFlags.set(researchId, true)
    // Clear any pending reconnect timers
    const t = this.wsReconnectTimers.get(researchId)
    if (t) {
      clearTimeout(t)
      this.wsReconnectTimers.delete(researchId)
    }

    // Use the API_BASE_URL and convert it to WebSocket URL
    const apiUrl = new URL(API_BASE_URL || window.location.origin)
    const wsProtocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:'
    const safeResearchId = encodeURIComponent(researchId)
    const wsUrl = `${wsProtocol}//${apiUrl.host}/ws/research/${safeResearchId}`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      // Emit a synthetic 'connected' event so UIs can clear connecting state
      const synthetic: WebSocketMessage = {
        type: 'connected',
        data: { message: 'connected' },
        timestamp: new Date().toISOString()
      }
      onMessage(synthetic)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        const validated = validateWebSocketMessage(message)
        if (validated) {
          onMessage(validated)
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    ws.onerror = (error) => {
      console.error(`WebSocket error for research ${researchId}:`, error)
    }

    ws.onclose = () => {
      // Drop current connection reference
      this.wsConnections.delete(researchId)
      const shouldReconnect = this.wsReconnectFlags.get(researchId)
      if (shouldReconnect) {
        // Attempt reconnection after 5s; reuse existing callback
        const timer = setTimeout(() => {
          const cb = this.wsCallbacks.get(researchId)
          if (cb) {
            this.connectWebSocket(researchId, cb)
          }
        }, 5000)
        this.wsReconnectTimers.set(researchId, timer)
      } else {
        // Manual shutdown: clean up callback map
        this.wsCallbacks.delete(researchId)
      }
    }

    this.wsConnections.set(researchId, ws)
    this.wsCallbacks.set(researchId, onMessage)
  }

  disconnectWebSocket(researchId?: string): void {
    if (researchId) {
      // Stop reconnection attempts for this id
      this.wsReconnectFlags.set(researchId, false)
      const t = this.wsReconnectTimers.get(researchId)
      if (t) { clearTimeout(t); this.wsReconnectTimers.delete(researchId) }
      const ws = this.wsConnections.get(researchId)
      if (ws) { ws.close() }
      this.wsConnections.delete(researchId)
      this.wsCallbacks.delete(researchId)
    } else {
      // Disconnect all WebSockets
      // Disable reconnect across the board and clear timers
      this.wsConnections.forEach((_ws, id) => {
        this.wsReconnectFlags.set(id, false)
        const t = this.wsReconnectTimers.get(id)
        if (t) { clearTimeout(t) }
      })
      this.wsReconnectTimers.clear()
      this.wsConnections.forEach((ws) => ws.close())
      this.wsConnections.clear()
      this.wsCallbacks.clear()
    }
  }

  unsubscribeFromResearch(researchId: string): void {
    this.disconnectWebSocket(researchId)
  }

  // Feedback API
  async submitClassificationFeedback(payload: {
    research_id?: string
    query: string
    original: {
      primary: string
      secondary?: string | null
      distribution?: Record<string, number>
      confidence?: number
    }
    user_correction?: string
    rationale?: string
  }): Promise<void> {
    const response = await this.fetchWithAuth('/feedback/classification', {
      method: 'POST',
      body: JSON.stringify(payload)
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to submit classification feedback')
    }
  }

  async submitAnswerFeedback(payload: {
    research_id: string
    rating: number
    reason?: string
    improvements?: string[]
    helpful?: boolean
  }): Promise<void> {
    const response = await this.fetchWithAuth('/feedback/answer', {
      method: 'POST',
      body: JSON.stringify(payload)
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to submit answer feedback')
    }
  }

  // Deep Research Management
  async resumeDeepResearch(researchId: string): Promise<void> {
    const safeResearchId = encodeURIComponent(researchId)
    const response = await this.fetchWithAuth(`/research/deep/${safeResearchId}/resume`, {
      method: 'POST'
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to resume deep research')
    }
  }

  async getDeepResearchStatus(): Promise<Record<string, unknown>> {
    const response = await this.fetchWithAuth('/research/deep/status')

    if (!response.ok) {
      const error = await response.json()
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to get deep research status')
    }

    return response.json()
  }

  // Responses API (debug/util)
  async getResponseDetails(responseId: string, include?: string[]): Promise<Record<string, unknown>> {
    const params = new URLSearchParams()
    if (Array.isArray(include)) {
      include.forEach((i) => params.append('include[]', i))
    }
    const query = params.toString() ? `?${params.toString()}` : ''
    const response = await this.fetchWithAuth(`/responses/${encodeURIComponent(responseId)}${query}`)
    if (!response.ok) {
      const error = await response.json().catch(() => null)
      throw new Error((error && (error.detail || error.error)) || 'Failed to get response details')
    }
    return response.json()
  }

  // Source credibility
  async getSourceCredibility(domain: string, paradigm: Paradigm): Promise<Record<string, unknown>> {
    const response = await this.fetchWithAuth(`/sources/credibility/${domain}?paradigm=${paradigm}`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Failed to get source credibility')
    }

    return response.json()
  }
}

export const api = new APIService()
export default api
