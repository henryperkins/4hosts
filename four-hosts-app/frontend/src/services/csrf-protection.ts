export class CSRFProtection {
  private static csrfToken: string | null = null
  private static tokenPromise: Promise<string> | null = null

  static async getToken(forceRefresh = false): Promise<string> {
    if (this.csrfToken && !forceRefresh) {
      return this.csrfToken
    }

    if (this.tokenPromise && !forceRefresh) {
      return this.tokenPromise
    }

    this.tokenPromise = this.fetchCSRFToken()
    this.csrfToken = await this.tokenPromise
    this.tokenPromise = null
    
    return this.csrfToken
  }

  private static async fetchCSRFToken(): Promise<string> {
    try {
      // The CSRF token endpoint is intentionally version-agnostic and lives
      // at `/api/csrf-token` on the backend (see `backend/core/app.py`).
      // Using the versioned path here results in a 404 which blocks every
      // subsequent state-changing request. Align the frontend to the actual
      // backend route so the token can be fetched correctly.

      const response = await fetch('/api/csrf-token', {
        credentials: 'include'
      })
      
      if (!response.ok) {
        throw new Error('Failed to fetch CSRF token')
      }
      
      const data = await response.json()
      return data.csrf_token
    } catch (error) {
      console.error('CSRF token fetch failed:', error)
      return ''
    }
  }

  static clearToken(): void {
    this.csrfToken = null
    this.tokenPromise = null
  }

  static addCSRFHeader(headers: HeadersInit = {}): HeadersInit {
    const headersObj = headers instanceof Headers ? 
      Object.fromEntries(headers.entries()) : 
      headers as Record<string, string>
    
    if (this.csrfToken) {
      headersObj['X-CSRF-Token'] = this.csrfToken
    }
    
    return headersObj
  }
}
