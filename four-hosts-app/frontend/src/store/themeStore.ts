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
      
      toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),
      
      setDarkMode: (darkMode) => set(() => ({ darkMode }))
    }),
    {
      name: 'theme-store',
      storage: createJSONStorage(() => localStorage),
      onRehydrateStorage: () => (state) => {
        // Apply theme on rehydration
        if (state?.darkMode) {
          document.documentElement.classList.add('dark')
          document.documentElement.style.colorScheme = 'dark'
        } else {
          document.documentElement.classList.remove('dark')
          document.documentElement.style.colorScheme = 'light'
        }
      }
    }
  )
)

// Selectors
export const useDarkMode = () => useThemeStore((state) => state.darkMode)
export const useToggleDarkMode = () => useThemeStore((state) => state.toggleDarkMode)
