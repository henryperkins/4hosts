import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

interface ThemeState {
  darkMode: boolean
  toggleDarkMode: () => void
  setDarkMode: (darkMode: boolean) => void
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      darkMode: false,
      
      toggleDarkMode: () => set((state) => {
        const newValue = !state.darkMode
        
        // Update DOM
        if (newValue) {
          document.documentElement.classList.add('dark')
        } else {
          document.documentElement.classList.remove('dark')
        }
        
        return { darkMode: newValue }
      }),
      
      setDarkMode: (darkMode) => set(() => {
        // Update DOM
        if (darkMode) {
          document.documentElement.classList.add('dark')
        } else {
          document.documentElement.classList.remove('dark')
        }
        
        return { darkMode }
      })
    }),
    {
      name: 'theme-store',
      storage: createJSONStorage(() => localStorage),
      onRehydrateStorage: () => (state) => {
        // Apply theme on rehydration
        if (state?.darkMode) {
          document.documentElement.classList.add('dark')
        } else {
          document.documentElement.classList.remove('dark')
        }
      }
    }
  )
)

// Selectors
export const useDarkMode = () => useThemeStore((state) => state.darkMode)
export const useToggleDarkMode = () => useThemeStore((state) => state.toggleDarkMode)