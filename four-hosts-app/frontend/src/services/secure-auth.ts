import { SecureAPIClient } from './secure-api-client'

const apiClient = new SecureAPIClient('/api')

export class SecureAuthHandler {
  static async clearAuthTokens(): Promise<void> {
    try {
      await apiClient.post('/auth/logout')
    } catch {
      // Handle logout failure silently
    }
  }

  static async refreshAccessToken(): Promise<boolean> {
    try {
      await apiClient.post('/auth/refresh')
      return true
    } catch {
      return false
    }
  }
}