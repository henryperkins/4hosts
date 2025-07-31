import React, { useState, useRef } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../../hooks/useAuth'
import { LogIn, Mail, Lock, Eye, EyeOff, Check, AlertCircle } from 'lucide-react'
import { Button } from '../ui/Button'
import { InputField } from '../ui/InputField'

// Animation and timing constants
const REDIRECT_DELAY = 500
const ERROR_SHAKE_DURATION = 500

export const LoginForm: React.FC = () => {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [formTouched, setFormTouched] = useState({ username: false, password: false })
  const [loginSuccess, setLoginSuccess] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()
  const formRef = useRef<HTMLFormElement>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')

    try {
      await login(username, password)
      setLoginSuccess(true)
      setTimeout(() => {
        navigate('/')
      }, REDIRECT_DELAY)
    } catch (error) {
      // Error is handled by AuthContext, but we should also show it here
      setError(error instanceof Error ? error.message : 'Login failed')
      // Add error shake animation
      if (formRef.current) {
        formRef.current.style.animation = 'shake 0.5s'
        setTimeout(() => {
          if (formRef.current) formRef.current.style.animation = ''
        }, ERROR_SHAKE_DURATION)
      }
    } finally {
      setIsLoading(false)
    }
  }

  const validateEmail = (email: string) => {
    return email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)
  }

  const isValidUsername = (username: string) => {
    // Username validation: alphanumeric and underscore, 3-20 chars
    return /^[a-zA-Z0-9_]{3,20}$/.test(username)
  }

  const validateInput = (input: string) => {
    // Check if input is email format or valid username
    return validateEmail(input) || isValidUsername(input)
  }

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword)
  }

  const getInputStatus = (field: 'username' | 'password'): 'error' | 'success' | 'strong' | '' => {
    if (!formTouched[field]) return ''

    if (field === 'username') {
      if (!username) return ''
      return validateInput(username) ? 'success' : 'error'
    }

    if (field === 'password') {
      if (!password) return ''
      if (password.length < 6) return 'error'
      if (password.length >= 8 && /(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(password)) {
        return 'strong'
      }
      return 'success'
    }

    return ''
  }

  const usernameStatus = (getInputStatus('username') as 'error' | 'success') || undefined
  const pwdRaw = getInputStatus('password')
  const passwordStatus: 'error' | 'success' | undefined =
    pwdRaw === 'error' ? 'error' : pwdRaw ? 'success' : undefined
  const pwdIconOk = pwdRaw === 'success' || pwdRaw === 'strong'

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 animate-fade-in">
        <div>
          <h2 className="mt-6 text-center text-2xl font-bold text-text">
            Sign in to your account
          </h2>
          <p className="mt-2 text-center text-sm text-text-muted">
            Or{' '}
            <Link to="/register" className="font-medium text-primary hover:text-primary/80 transition-colors">
              create a new account
            </Link>
          </p>
        </div>
        <form ref={formRef} className="mt-8 space-y-6" onSubmit={handleSubmit}>
          {error && (
            <div className="rounded-md bg-error/10 p-4 animate-slide-up border border-error/20">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 text-error mr-2 shrink-0" />
                <p className="text-sm text-error">{error}</p>
              </div>
            </div>
          )}
          {loginSuccess && (
            <div className="rounded-md bg-success/10 p-4 animate-slide-up border border-success/20">
              <div className="flex items-center">
                <Check className="h-5 w-5 text-success mr-2 animate-fade-in" />
                <p className="text-sm text-success">
                  Login successful! Redirecting...
                </p>
              </div>
            </div>
          )}
          <div className="space-y-4">
            <div className="relative">
              <label htmlFor="username" className="sr-only">
                Email or Username
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 flex items-center pointer-events-none pl-3">
                  <Mail className={`h-5 w-5 transition-colors ${
                    usernameStatus === 'error' ? 'text-error' :
                    usernameStatus === 'success' ? 'text-success' :
                    'text-text-muted'
                  }`} />
                </div>
                <InputField
                  id="username"
                  name="username"
                  type="text"
                  autoComplete="username"
                  required
                  status={(getInputStatus('username') as 'error' | 'success') || undefined}
                  className="pl-10"
                  placeholder="Email or Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  onBlur={() => setFormTouched(prev => ({ ...prev, username: true }))}
                />
              </div>
              {formTouched.username && username && !validateInput(username) && (
                <p className="mt-1 text-sm text-error animate-slide-up flex items-center">
                  <AlertCircle className="h-4 w-4 mr-1" />
                  Please enter a valid email address or username
                </p>
              )}
              {formTouched.username && username && validateInput(username) && (
                <p className="mt-1 text-sm text-success animate-slide-up flex items-center">
                  <Check className="h-4 w-4 mr-1" />
                  Valid input
                </p>
              )}
            </div>
            <div className="relative">
              <label htmlFor="password" className="sr-only">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 flex items-center pointer-events-none pl-3">
                  <Lock className={`h-5 w-5 transition-colors ${
                    passwordStatus === 'error' ? 'text-error' :
                    pwdIconOk ? 'text-success' :
                    'text-text-muted'
                  }`} />
                </div>
                <InputField
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  required
                  status={passwordStatus}
                  className="pl-10 pr-10"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onBlur={() => setFormTouched(prev => ({ ...prev, password: true }))}
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                  onClick={togglePasswordVisibility}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? (
                    <EyeOff className="h-5 w-5 text-text-muted hover:text-text transition-colors duration-200" />
                  ) : (
                    <Eye className="h-5 w-5 text-text-muted hover:text-text transition-colors duration-200" />
                  )}
                </button>
              </div>
              {formTouched.password && password && (
                <div className="mt-1 space-y-1">
                  {password.length < 6 && (
                    <p className="text-sm text-error animate-slide-up flex items-center">
                      <AlertCircle className="h-4 w-4 mr-1" />
                      Password must be at least 6 characters
                    </p>
                  )}
                  {password.length >= 6 && password.length < 8 && (
                    <p className="text-sm text-text-subtle animate-slide-up flex items-center">
                      <AlertCircle className="h-4 w-4 mr-1" />
                      Consider using at least 8 characters
                    </p>
                  )}
                  {pwdRaw === 'strong' && (
                    <p className="text-sm text-success animate-slide-up flex items-center">
                      <Check className="h-4 w-4 mr-1" />
                      Strong password
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>

          <div>
            <Button
              type="submit"
              variant="primary"
              disabled={!username || !password}
              loading={isLoading}
              fullWidth
              icon={LogIn}
            >
              {isLoading ? 'Signing in' : 'Sign in'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
