// Auth client for handling CSRF and authentication flow

class AuthClient {
  private csrfToken: string | null = null;

  async getCsrfToken(): Promise<string> {
    try {
      const response = await fetch('/api/csrf-token', {
        method: 'GET',
        credentials: 'include', // Important: include cookies
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to get CSRF token: ${response.status}`);
      }

      const data = await response.json();
      this.csrfToken = data.csrf_token;
      if (!this.csrfToken) {
        throw new Error('CSRF token not received from server');
      }
      return this.csrfToken;
    } catch (error) {
      console.error('Failed to get CSRF token:', error);
      throw error;
    }
  }

  async login(email: string, password: string): Promise<{ success: boolean; user: { id: string; email: string; name?: string; role: string }; message?: string }> {
    // Ensure we have a CSRF token
    if (!this.csrfToken) {
      await this.getCsrfToken();
    }

    const response = await fetch('/auth/login', {
      method: 'POST',
      credentials: 'include', // Important: include cookies
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': this.csrfToken!,
      },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    return response.json();
  }

  async register(email: string, password: string, fullName: string): Promise<{ success: boolean; user: { id: string; email: string; name?: string; role: string }; message?: string }> {
    // Ensure we have a CSRF token
    if (!this.csrfToken) {
      await this.getCsrfToken();
    }

    const response = await fetch('/auth/register', {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': this.csrfToken!,
      },
      body: JSON.stringify({ 
        email, 
        password, 
        full_name: fullName 
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Registration failed');
    }

    return response.json();
  }

  async logout(): Promise<void> {
    if (!this.csrfToken) {
      await this.getCsrfToken();
    }

    await fetch('/auth/logout', {
      method: 'POST',
      credentials: 'include',
      headers: {
        'X-CSRF-Token': this.csrfToken!,
      },
    });

    // Clear the CSRF token after logout
    this.csrfToken = null;
  }

  async refreshToken(): Promise<{ access_token: string; refresh_token?: string; token_type: string; expires_in?: number }> {
    if (!this.csrfToken) {
      await this.getCsrfToken();
    }

    const response = await fetch('/auth/refresh', {
      method: 'POST',
      credentials: 'include',
      headers: {
        'X-CSRF-Token': this.csrfToken!,
      },
    });

    if (!response.ok) {
      throw new Error('Token refresh failed');
    }

    return response.json();
  }

  async makeAuthenticatedRequest(url: string, options: RequestInit = {}): Promise<Response> {
    if (!this.csrfToken) {
      await this.getCsrfToken();
    }

    return fetch(url, {
      ...options,
      credentials: 'include',
      headers: {
        ...options.headers,
        'X-CSRF-Token': this.csrfToken!,
      },
    });
  }

  // Get the current CSRF token (for other services to use)
  getCurrentCsrfToken(): string | null {
    return this.csrfToken;
  }
}

export const authClient = new AuthClient();