import { AuthTokens } from './api';

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
   * Store auth token with better error handling
   */
  static storeAuthToken(token: string): void {
    try {
      localStorage.setItem('auth_token', token);
    } catch (error) {
      console.error('Failed to store authentication token:', error);
      throw new Error('Browser storage unavailable. Please enable cookies and try again.');
    }
  }

  /**
   * Clear auth token
   */
  static clearAuthToken(): void {
    try {
      localStorage.removeItem('auth_token');
    } catch (error) {
      console.error('Failed to clear authentication token:', error);
    }
  }
}

// Enhanced auth tokens interface for better error handling
export interface AuthTokens {
  access_token: string;
  token_type: string;
  refresh_token?: string;
  expires_in: number;
  user?: {
    id: string;
    email: string;
    username: string;
  };
}
