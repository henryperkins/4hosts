import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { subscribeWithSelector } from 'zustand/middleware'
import api from '../services/api'
import toast from 'react-hot-toast'
import type { User, UserPreferences } from '../types'

interface AuthState {
  // Auth state
  isAuthenticated: boolean
  user: User | null
  loading: boolean
  
  // UI state
  wsConnected: boolean
  activeResearchId: string | null
  
  // Actions
  login: (email: string, password: string) => Promise<void>
  register: (username: string, email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  updatePreferences: (preferences: UserPreferences) => Promise<void>
  setUser: (user: User | null) => void
  setLoading: (loading: boolean) => void
  setWsConnected: (connected: boolean) => void
  setActiveResearch: (id: string | null) => void
  checkAuth: () => Promise<void>
  reset: () => void
}

export const useAuthStore = create<AuthState>()(
  subscribeWithSelector(
    persist(
      (set, get) => ({
        // Initial state
        isAuthenticated: false,
        user: null,
        loading: true,
        wsConnected: false,
        activeResearchId: null,

        // Auth actions
        login: async (email: string, password: string) => {
          try {
            set({ loading: true })
            
            if (!email.includes('@')) {
              throw new Error('Please use your email address to login.')
            }

            await api.login(email, password)
            const user = await api.getCurrentUser()
            
            set({
              isAuthenticated: true,
              user,
              loading: false
            })
            
            toast.success('Logged in successfully!')
          } catch (error) {
            set({ loading: false })
            
            let errorMessage = 'Login failed'
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
            throw new Error(errorMessage)
          }
        },

        register: async (username: string, email: string, password: string) => {
          try {
            set({ loading: true })
            
            await api.register(username, email, password)
            const user = await api.getCurrentUser()
            
            set({
              isAuthenticated: true,
              user,
              loading: false
            })
            
            toast.success('Registration successful!')
          } catch (error) {
            set({ loading: false })
            
            let errorMessage = 'Registration failed'
            if (error instanceof Error) {
              errorMessage = error.message || errorMessage
            }
            
            toast.error(errorMessage)
            throw new Error(errorMessage)
          }
        },

        logout: async () => {
          try {
            await api.logout()
          } catch {
            // Continue with logout even if API call fails
          } finally {
            get().reset()
            toast.success('Logged out successfully')
          }
        },

        updatePreferences: async (preferences: UserPreferences) => {
          const currentUser = get().user
          if (!currentUser) {
            throw new Error('No user logged in')
          }

          try {
            await api.updateUserPreferences(preferences)
            
            const updatedUser = {
              ...currentUser,
              preferences: {
                ...currentUser.preferences,
                ...preferences
              }
            }
            
            set({ user: updatedUser })
            toast.success('Preferences updated')
          } catch (error) {
            toast.error('Failed to update preferences')
            throw error
          }
        },

        checkAuth: async () => {
          try {
            const user = await api.getCurrentUser()
            set({
              isAuthenticated: true,
              user,
              loading: false
            })
          } catch {
            set({
              isAuthenticated: false,
              user: null,
              loading: false
            })
          }
        },

        // UI actions
        setUser: (user) => set({ user, isAuthenticated: !!user }),
        setLoading: (loading) => set({ loading }),
        setWsConnected: (wsConnected) => set({ wsConnected }),
        setActiveResearch: (activeResearchId) => set({ activeResearchId }),

        reset: () => set({
          isAuthenticated: false,
          user: null,
          loading: false,
          wsConnected: false,
          activeResearchId: null
        })
      }),
      {
        name: 'auth-store',
        version: 3, // Increment version to force storage reset
        storage: createJSONStorage(() => sessionStorage), // Use sessionStorage for better security
        partialize: () => ({
          // Don't persist authentication state - always verify with backend
          // This prevents stale auth state issues
        }),
        onRehydrateStorage: () => (state) => {
          // Always check auth status with backend on app load
          // This ensures frontend and backend are synchronized
          state?.checkAuth()
        }
      }
    )
  )
)

// Selectors for performance optimization
export const useIsAuthenticated = () => useAuthStore((state) => state.isAuthenticated)
export const useCurrentUser = () => useAuthStore((state) => state.user)
export const useAuthLoading = () => useAuthStore((state) => state.loading)
export const useWsConnected = () => useAuthStore((state) => state.wsConnected)
export const useActiveResearchId = () => useAuthStore((state) => state.activeResearchId)