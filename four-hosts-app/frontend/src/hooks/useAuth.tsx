import { useAuthStore } from '../store/authStore'

// Compatibility hook to match the old AuthContext interface
export function useAuth() {
  const {
    isAuthenticated,
    user,
    loading,
    login,
    register,
    logout,
    updatePreferences
  } = useAuthStore()

  return {
    isAuthenticated,
    user,
    loading,
    login,
    register,
    logout,
    updatePreferences
  }
}