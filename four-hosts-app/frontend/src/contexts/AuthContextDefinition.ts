import { createContext } from 'react'
import type { AuthState, UserPreferences } from '../types'

export interface AuthContextType extends AuthState {
  login: (username: string, password: string) => Promise<void>
  register: (username: string, email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  updatePreferences: (preferences: UserPreferences) => Promise<void>
}

export const AuthContext = createContext<AuthContextType | null>(null)
