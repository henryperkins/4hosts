import { CSRFProtection } from './csrf-protection'

interface RequestOptions extends RequestInit {
  skipCSRF?: boolean
}

export class SecureAPIClient {
  private baseURL: string

  constructor(baseURL: string) {
    this.baseURL = baseURL
  }

  private async request<T>(
    endpoint: string,
    options: RequestOptions = {},
    csrfRetry = false
  ): Promise<T> {
    const { skipCSRF = false, ...fetchOptions } = options

    // Default options for security
    const secureOptions: RequestInit = {
      ...fetchOptions,
      credentials: 'include', // Always include cookies
      headers: {
        'Content-Type': 'application/json',
        ...fetchOptions.headers
      }
    }

    // Add CSRF token for state-changing requests
    if (!skipCSRF && ['POST', 'PUT', 'DELETE', 'PATCH'].includes(options.method || 'GET')) {
      const csrfToken = await CSRFProtection.getToken()
      if (csrfToken) {
        (secureOptions.headers as Record<string, string>)['X-CSRF-Token'] = csrfToken
      }
    }

    // Use relative path when baseURL is empty to leverage Vite proxy
    const url = this.baseURL ? `${this.baseURL}${endpoint}` : endpoint
    const response = await fetch(url, secureOptions)

    // Handle CSRF token mismatch
    if (response.status === 403 && !csrfRetry) {
      try {
        const errorData = await response.json()
        if (errorData.detail === 'CSRF token mismatch') {
          console.log('CSRF token mismatch detected, refreshing token...')
          CSRFProtection.clearToken()
          await CSRFProtection.getToken(true) // Force refresh
          return this.request<T>(endpoint, options, true)
        }
      } catch {
        // If we can't parse the error, continue with normal flow
      }
    }

    // Handle token refresh
    if (response.status === 401) {
      const refreshed = await this.refreshToken()
      if (refreshed) {
        // Retry original request
        return this.request<T>(endpoint, options)
      }
    }

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  }

  private async refreshToken(): Promise<boolean> {
    try {
      const url = this.baseURL ? `${this.baseURL}/auth/refresh` : `/auth/refresh`
      const response = await fetch(url, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'X-CSRF-Token': await CSRFProtection.getToken()
        }
      })

      return response.ok
    } catch {
      return false
    }
  }

  // Safe API methods
  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET', skipCSRF: true })
  }

  async post<T>(endpoint: string, body?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined
    })
  }

  async put<T>(endpoint: string, body?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: body ? JSON.stringify(body) : undefined
    })
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' })
  }
}
