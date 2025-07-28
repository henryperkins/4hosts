import React from 'react'

// Theme context - separate file to avoid fast refresh issues
export const ThemeContext = React.createContext<{
  darkMode: boolean
  toggleDarkMode: () => void
}>({
  darkMode: false,
  toggleDarkMode: () => {}
})

export const useTheme = () => React.useContext(ThemeContext)
