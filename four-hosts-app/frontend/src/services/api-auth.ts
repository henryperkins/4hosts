interface AuthError {
  status: number;
  message: string;
  detail?: string;
}

export class AuthErrorHandler {
  /**
   * Check for authentication errors and provide user-friendly messages
   */
  static handleAuthError(response: Response): AuthError {
    const error: AuthError = {
      status: response.status,
      message: 'Authentication failed',
      detail: response.statusText
    };

    switch (response.status) {
      case 401:
        error.message = 'Invalid email or password';
        error.detail = 'Please check your credentials and try again';
        break;
      case 422:
        error.message = 'Invalid input';
        break;
      case 502:
      case 503:
        error.message = 'Server connection error';
        error.detail = 'Please ensure the backend server is running on http://localhost:8000';
        break;
    }

    return error;
  }

  /**
   * Extract detailed error messages from backend responses
   */
  static async extractErrorMessage(response: Response): Promise<string> {
    try {
      const errorText = await response.text();
      const errorData = JSON.parse(errorText);

      return errorData.detail || errorData.message || `HTTP ${response.status}: ${response.statusText}`;
    } catch {
      return `HTTP ${response.status}: ${response.statusText}`;
    }
  }

  /**
   * Store auth and refresh tokens with better error handling
   */
  static storeAuthTokens(accessToken: string, refreshToken?: string): void {
    try {
      localStorage.setItem('auth_token', accessToken);
      if (refreshToken) {
        localStorage.setItem('refresh_token', refreshToken);
      }
    } catch {
      // Failed to store authentication tokens
      throw new Error('Browser storage unavailable. Please enable cookies and try again.');
    }
  }

  /**
   * Retrieve the refresh token
   */
  static getRefreshToken(): string | null {
    try {
      return localStorage.getItem('refresh_token');
    } catch {
      // Failed to retrieve refresh token
      return null;
    }
  }

  /**
   * Clear auth and refresh tokens
   */
  static clearAuthTokens(): void {
    try {
      localStorage.removeItem('auth_token');
      localStorage.removeItem('refresh_token');
    } catch {
      // Failed to clear authentication tokens
    }
  }
}

