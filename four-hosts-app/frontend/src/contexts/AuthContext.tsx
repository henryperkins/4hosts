import React, { useState, useEffect } from 'react'
import type { ReactNode } from 'react'
import type { AuthState, UserPreferences } from '../types'
import { AuthContext } from './AuthContextDefinition'
import api from '../services/api'
import toast from 'react-hot-toast'

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    user: null,
    loading: true,
  })

  useEffect(() => {
    // Check if user is already logged in
    const checkAuth = async () => {
      const token = localStorage.getItem('auth_token')
      if (token) {
        try {
          const user = await api.getCurrentUser()
          setAuthState({
            isAuthenticated: true,
            user,
            loading: false,
          })
        } catch {
          // Token invalid, clear it
          localStorage.removeItem('auth_token')
          setAuthState({
            isAuthenticated: false,
            user: null,
            loading: false,
          })
        }
      } else {
        setAuthState({
          isAuthenticated: false,
          user: null,
          loading: false,
        })
      }
    }

    checkAuth()
  }, [])

  const login = async (emailOrUsername: string, password: string) => {
    try {
      // Backend requires email-based login; validate input early for UX
      if (!emailOrUsername.includes('@')) {
        throw new Error('Please use your email address to login.')
      }

      await api.login(emailOrUsername, password)
      const user = await api.getCurrentUser()
      setAuthState({
        isAuthenticated: true,
        user,
        loading: false,
      })
      toast.success('Logged in successfully!')
    } catch (error) {
      // Provide more user-friendly error messages
      let errorMessage = 'Login failed';
      if (error instanceof Error) {
        const msg = error.message || ''
        if (msg.includes('connect to backend')) {
          errorMessage = 'Cannot connect to server. Please ensure the backend is running.'
        } else if (msg.includes('Invalid credentials') || msg.includes('Invalid email') || msg.includes('401')) {
          errorMessage = 'Invalid email or password. Please try again.'
        } else if (msg.includes('email address')) {
          errorMessage = 'Please use your email address to login.'
        } else {
          errorMessage = msg
        }
      }
      toast.error(errorMessage)
      throw error
    }
  }

  const register = async (username: string, email: string, password: string) => {
    try {
      // Register returns tokens, not user
      await api.register(username, email, password)
      // Get user info after registration
      const user = await api.getCurrentUser()
      setAuthState({
        isAuthenticated: true,
        user,
        loading: false,
      })
      toast.success('Registration successful!')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Registration failed')
      throw error
    }
  }

  const logout = async () => {
    try {
      await api.logout()
      setAuthState({
        isAuthenticated: false,
        user: null,
        loading: false,
      })
      toast.success('Logged out successfully')
    } catch {
      // Even if logout fails on server, clear local state
      setAuthState({
        isAuthenticated: false,
        user: null,
        loading: false,
      })
    }
  }

  const updatePreferences = async (preferences: UserPreferences) => {
    try {
      await api.updateUserPreferences(preferences)
      // Refresh user data after updating preferences
      const updatedUser = await api.getCurrentUser()
      setAuthState(prev => ({
        ...prev,
        user: updatedUser,
      }))
      toast.success('Preferences updated')
    } catch (error) {
      toast.error('Failed to update preferences')
      throw error
    }
  }

  return (
    <AuthContext.Provider
      value={{
        ...authState,
        login,
        register,
        logout,
        updatePreferences,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}
