import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { Home, History, User, BarChart3, Menu, X } from 'lucide-react'
import { ToggleSwitch } from './ui/ToggleSwitch'
import { Button } from './ui/Button'
import { useAuth } from '../hooks/useAuth'
import { useDarkMode, useToggleDarkMode } from '../store/themeStore'

export const Navigation = () => {
  const { isAuthenticated, user, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const darkMode = useDarkMode()
  const toggleDarkMode = useToggleDarkMode()

  if (!isAuthenticated) return null

  const isActive = (path: string) => {
    return location.pathname === path
  }

  const closeMobileMenu = () => {
    setMobileMenuOpen(false)
  }

  // Paradigm-themed navigation items
  const navItems = [
    { path: '/', icon: Home, label: 'Research', paradigm: 'dolores' },
    { path: '/history', icon: History, label: 'History', paradigm: 'bernard' },
    { path: '/metrics', icon: BarChart3, label: 'Metrics', paradigm: 'teddy' },
  ]

  const getParadigmHoverClass = (paradigm?: string) => {
    if (!paradigm) return ''
    const paradigmHoverColors: Record<string, string> = {
      dolores: 'hover:text-paradigm-dolores',
      bernard: 'hover:text-paradigm-bernard',
      teddy: 'hover:text-paradigm-teddy',
      maeve: 'hover:text-paradigm-maeve'
    }
    return paradigmHoverColors[paradigm] || ''
  }

  return (
    <>
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-primary text-white px-4 py-2 rounded-md z-50">
        Skip to main content
      </a>
      <nav className="bg-surface shadow-lg border-b border-border animate-slide-down transition-all duration-300 backdrop-blur-sm bg-opacity-95 dark:bg-opacity-95">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center gap-8">
            <Link
              to="/"
              className="text-xl font-bold text-text hover:text-blue-600 dark:hover:text-blue-400 transition-all duration-300 flex items-center gap-2 group touch-target"
              onClick={closeMobileMenu}
            >
              <span className="text-2xl group-hover:rotate-12 transition-transform duration-300 inline-block" role="img" aria-label="Theater masks">🎭</span>
              <span className="hidden sm:inline gradient-brand text-responsive-xl">Four Hosts Research</span>
              <span className="sm:hidden gradient-brand text-responsive-lg">4H Research</span>
            </Link>
            <div className="hidden md:flex items-center gap-2">
              {navItems.map(({ path, icon: Icon, label, paradigm }) => {
                return (
                  <Link
                    key={path}
                    to={path}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 hover-lift focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 touch-target ${
                      isActive(path)
                        ? 'bg-linear-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 text-blue-700 dark:text-blue-300 shadow-lg scale-105'
                        : `text-text-muted hover:bg-surface-subtle ${getParadigmHoverClass(paradigm)} active:scale-95`
                    }`}
                    aria-current={isActive(path) ? 'page' : undefined}
                  >
                    <Icon className={`h-4 w-4 transition-all duration-300 ${isActive(path) ? 'animate-pulse' : 'group-hover:rotate-12'}`} />
                    <span>{label}</span>
                  </Link>
                )
              })}
            </div>
          </div>

          <div className="flex items-center gap-2 md:gap-4">
            <span className="hidden md:block text-sm text-text-muted animate-fade-in">
              Welcome, <span className="font-medium gradient-accent">{user?.username}</span>
            </span>

            {/* Dark mode toggle */}
            <ToggleSwitch
              checked={darkMode}
              onChange={toggleDarkMode}
              aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              size="sm"
              className="hidden md:inline-flex"
            />

            <Link
              to="/profile"
              className={`hidden md:flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 hover-lift focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                isActive('/profile')
                  ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 shadow-md'
                  : 'text-text-muted hover:text-text hover:bg-surface-subtle'
              }`}
              aria-current={isActive('/profile') ? 'page' : undefined}
            >
              <User className="h-4 w-4 transition-transform duration-200 hover:scale-110" />
              <span>Profile</span>
            </Link>

            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="sm"
              icon={mobileMenuOpen ? X : Menu}
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden"
              aria-label="Toggle menu"
              aria-expanded={mobileMenuOpen}
            />
          </div>
        </div>

        {/* Mobile menu with smooth transitions */}
        <div className={`md:hidden transition-all duration-300 ease-in-out overflow-hidden ${
          mobileMenuOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
        }`}>
          <div className="py-2 space-y-1">
            {navItems.map(({ path, label }) => (
              <Link
                key={path}
                to={path}
                onClick={closeMobileMenu}
                className={`block px-4 py-2 transition-all duration-300 transform hover:translate-x-2 ${
                  isActive(path)
                  ? 'bg-linear-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 text-blue-700 dark:text-blue-300'
                    : 'text-text-muted hover:bg-surface-subtle'
                }`}
              >
                {label}
              </Link>
            ))}
            <div className="px-4 py-2 border-t border-border mt-2 pt-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-muted">Dark Mode</span>
                <ToggleSwitch
                  checked={darkMode}
                  onChange={toggleDarkMode}
                  size="sm"
                />
              </div>
            </div>
            <Button
              variant="ghost"
              fullWidth
              onClick={async () => {
                await logout()
                closeMobileMenu()
                navigate('/login')
              }}
              className="text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 justify-start"
            >
              Logout
            </Button>
          </div>
        </div>
      </div>
    </nav>
    </>
  )
}