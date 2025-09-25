import { useState, useMemo, useCallback } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { FiHome, FiClock, FiUser, FiBarChart2, FiMenu, FiX, FiLayers } from 'react-icons/fi'
import { ToggleSwitch } from './ui/ToggleSwitch'
import { Button } from './ui/Button'
import { useAuth } from '../hooks/useAuth'
import { useThemeStore, useDarkMode } from '../store/themeStore'
import api from '../services/api'

export const Navigation = () => {
  const { isAuthenticated, user, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const darkMode = useDarkMode()
  const { setDarkMode } = useThemeStore()

  const navItems = useMemo(() => ([
    { path: '/', icon: FiHome, label: 'Research', paradigm: 'dolores' },
    { path: '/history', icon: FiClock, label: 'History', paradigm: 'bernard' },
    { path: '/metrics', icon: FiBarChart2, label: 'Metrics', paradigm: 'teddy' },
  ]), [])

  const paradigmHoverColors = useMemo(() => ({
    dolores: 'hover:text-paradigm-dolores',
    bernard: 'hover:text-paradigm-bernard',
    teddy: 'hover:text-paradigm-teddy',
    maeve: 'hover:text-paradigm-maeve'
  } as Record<string, string>), [])

  const getParadigmHoverClass = useCallback((paradigm?: string) => {
    if (!paradigm) return ''
    return paradigmHoverColors[paradigm] || ''
  }, [paradigmHoverColors])

  const handleLogout = useCallback(async () => {
    await logout()
    api.disconnectWebSocket()
    setMobileMenuOpen(false)
    navigate('/login')
  }, [logout, navigate])

  if (!isAuthenticated) return null

  const isActive = (path: string) => {
    return location.pathname === path
  }

  const closeMobileMenu = () => {
    setMobileMenuOpen(false)
  }

  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-primary text-white px-4 py-2 rounded-md z-50">
        Skip to main content
      </a>
      {/*
        Include mobile safe-area padding to ensure the navigation bar does not
        collide with device notches / rounded corners on modern phones. The
        utility is defined globally in `index.css`.
      */}
      <nav className="bg-surface shadow-lg border-b border-border animate-slide-down transition-all duration-300 backdrop-blur-sm bg-opacity-95 dark:bg-opacity-95 mobile-safe-area">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/*
            Use a smaller horizontal gap on very small screens to avoid the
            logo + hamburger wrapping when viewport width is ≤ 375 px. The
            default gap is reduced to `gap-4`; we restore the previous spacing
            from the `sm:` breakpoint upwards.
          */}
          <div className="flex items-center gap-4 sm:gap-8 min-w-0">
            <Link
              to="/"
              className="text-xl font-bold text-text hover:text-primary transition-all duration-300 flex items-center gap-2 group touch-target min-w-0 overflow-hidden"
              onClick={closeMobileMenu}
            >
              <FiLayers className="h-6 w-6 group-hover:rotate-12 transition-transform duration-300" aria-hidden="true" />
              <span className="hidden sm:inline gradient-brand text-responsive-xl">Four Hosts Research</span>
              <span className="sm:hidden gradient-brand text-responsive-lg truncate">Four Hosts</span>
            </Link>
            <div className="hidden md:flex items-center gap-2">
              {navItems.map(({ path, icon: Icon, label, paradigm }) => {
                return (
                  <Link
                    key={path}
                    to={path}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 hover-lift focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary touch-target ${
                      isActive(path)
                        ? 'bg-primary/10 text-primary shadow-lg scale-105'
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
              onChange={(v) => setDarkMode(v)}
              aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              size="sm"
              className="hidden md:inline-flex"
            />

            <Link
              to="/profile"
              className={`hidden md:flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 hover-lift focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary ${
                isActive('/profile')
                  ? 'bg-primary/10 text-primary shadow-md'
                  : 'text-text-muted hover:text-text hover:bg-surface-subtle'
              }`}
              aria-current={isActive('/profile') ? 'page' : undefined}
            >
              <FiUser className="h-4 w-4 transition-transform duration-200 hover:scale-110" />
              <span>Profile</span>
            </Link>

            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="sm"
              icon={mobileMenuOpen ? FiX : FiMenu}
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
                  ? 'bg-primary/10 text-primary'
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
                  onChange={(v) => setDarkMode(v)}
                  size="sm"
                />
              </div>
            </div>
            <Button
              variant="ghost"
              fullWidth
              onClick={handleLogout}
              className="text-error hover:bg-error/10 justify-start"
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
