import React, { useState } from 'react'
import { FiUser, FiSettings, FiSave, FiLogOut, FiMoon, FiSun, FiDatabase, FiCpu, FiCrosshair } from 'react-icons/fi'
import { useAuth } from '../hooks/useAuth'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import type { UserPreferences, Paradigm } from '../types'

export const UserProfile: React.FC = () => {
  const { user, updatePreferences, logout } = useAuth()
  const navigate = useNavigate()
  const [isEditing, setIsEditing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  const [preferences, setPreferences] = useState<UserPreferences>({
    default_paradigm: user?.preferences?.default_paradigm,
    default_depth: user?.preferences?.default_depth || 'standard',
    enable_real_search: user?.preferences?.enable_real_search || false,
    enable_ai_classification: user?.preferences?.enable_ai_classification || false,
    theme: user?.preferences?.theme || 'light',
  })

  const handleSave = async () => {
    setIsSaving(true)
    try {
      await updatePreferences(preferences)
      setIsEditing(false)
      toast.success('Preferences saved successfully')
    } catch {
      toast.error('Failed to save preferences')
    } finally {
      setIsSaving(false)
    }
  }

  const handleLogout = async () => {
    await logout()
    navigate('/login')
  }

  if (!user) return null

  return (
    <div className="bg-surface border border-border rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-text">User Profile</h2>
        <button
          onClick={handleLogout}
          className="flex items-center gap-2 px-4 py-2 text-error hover:bg-error/10 rounded-lg transition-colors"
        >
          <FiLogOut className="h-4 w-4" />
          Logout
        </button>
      </div>

      {/* User Info */}
      <div className="mb-8 p-4 bg-surface-subtle rounded-lg">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
            <FiUser className="h-8 w-8 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-lg text-text">{user.username}</h3>
            <p className="text-text-muted">{user.email}</p>
            <div className="flex items-center gap-4 mt-1">
              <p className="text-sm text-text-subtle">
                Member since {user.created_at ? new Date(user.created_at).toLocaleDateString() : 'Unknown'}
              </p>
              {/* Role Badge */}
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                user.role === 'admin' ? 'bg-error/10 text-error' :
                user.role === 'enterprise' ? 'bg-primary/10 text-primary' :
                user.role === 'pro' ? 'bg-success/10 text-success' :
                user.role === 'basic' ? 'bg-surface-subtle text-text' :
                'bg-surface-subtle text-text'
              }`}>
                {user.role?.toUpperCase() || 'FREE'}
              </span>
            </div>
          </div>
        </div>

        {/* Role-specific features notice */}
        {user.role === 'free' && (
          <div className="mt-4 p-3 bg-primary/10 border border-primary/30 rounded-lg">
            <p className="text-sm text-primary">
              <FiCrosshair className="inline-block mr-1" aria-hidden="true" />
              Upgrade to unlock deep research, export features, and advanced analytics!
            </p>
          </div>
        )}
      </div>

      {/* Preferences */}
      <div className="space-y-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text">Preferences</h3>
          {!isEditing ? (
            <button
              onClick={() => setIsEditing(true)}
              className="flex items-center gap-2 px-4 py-2 text-primary hover:bg-primary/10 rounded-lg transition-colors"
            >
              <FiSettings className="h-4 w-4" />
              Edit
            </button>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={() => setIsEditing(false)}
                className="px-4 py-2 text-text-muted hover:bg-surface-subtle rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={isSaving}
                className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                <FiSave className="h-4 w-4" />
                {isSaving ? 'Saving...' : 'Save'}
              </button>
            </div>
          )}
        </div>

        {/* Default Paradigm */}
        <div>
          <label className="block text-sm font-medium text-text mb-2">
            Default Paradigm
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {(['dolores', 'teddy', 'bernard', 'maeve'] as Paradigm[]).map((paradigm) => (
              <button
                key={paradigm}
                type="button"
                disabled={!isEditing}
                onClick={() => setPreferences({ ...preferences, default_paradigm: paradigm })}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  preferences.default_paradigm === paradigm
                    ? 'bg-primary text-white'
                    : 'bg-surface-muted text-text hover:bg-surface-subtle'
                } ${!isEditing ? 'cursor-not-allowed opacity-60' : ''}`}
              >
                {paradigm.charAt(0).toUpperCase() + paradigm.slice(1)}
              </button>
            ))}
            <button
              type="button"
              disabled={!isEditing}
              onClick={() => setPreferences({ ...preferences, default_paradigm: undefined })}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                !preferences.default_paradigm
                  ? 'bg-primary text-white'
                  : 'bg-surface-muted text-text hover:bg-surface-subtle'
              } ${!isEditing ? 'cursor-not-allowed opacity-60' : ''}`}
            >
              Auto-detect
            </button>
          </div>
        </div>

        {/* Default Research Depth */}
        <div>
          <label className="block text-sm font-medium text-text mb-2">
            Default Research Depth
          </label>
          <div className="grid grid-cols-3 gap-2">
            {(['quick', 'standard', 'deep'] as const).map((depth) => (
              <button
                key={depth}
                type="button"
                disabled={!isEditing}
                onClick={() => setPreferences({ ...preferences, default_depth: depth })}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  preferences.default_depth === depth
                    ? 'bg-primary text-white'
                    : 'bg-surface-muted text-text hover:bg-surface-subtle'
                } ${!isEditing ? 'cursor-not-allowed opacity-60' : ''}`}
              >
                {depth.charAt(0).toUpperCase() + depth.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Feature Toggles */}
        <div className="space-y-3">
          <label className="flex items-center justify-between p-3 bg-surface-subtle rounded-lg">
            <div className="flex items-center gap-3">
              <FiDatabase className="h-5 w-5 text-text-muted" />
              <div>
                <p className="font-medium text-text">Enable Real Search</p>
                <p className="text-sm text-text-muted">Use live search APIs for current data</p>
              </div>
            </div>
            <input
              type="checkbox"
              disabled={!isEditing}
              checked={preferences.enable_real_search || false}
              onChange={(e) => setPreferences({ ...preferences, enable_real_search: e.target.checked })}
              className="h-5 w-5 text-primary focus:ring-primary border-border rounded disabled:opacity-50"
            />
          </label>

          <label className="flex items-center justify-between p-3 bg-surface-subtle rounded-lg">
            <div className="flex items-center gap-3">
              <FiCpu className="h-5 w-5 text-text-muted" />
              <div>
                <p className="font-medium text-text">Enable AI Classification</p>
                <p className="text-sm text-text-muted">Use advanced AI for paradigm detection</p>
              </div>
            </div>
            <input
              type="checkbox"
              disabled={!isEditing}
              checked={preferences.enable_ai_classification || false}
              onChange={(e) => setPreferences({ ...preferences, enable_ai_classification: e.target.checked })}
              className="h-5 w-5 text-primary focus:ring-primary border-border rounded disabled:opacity-50"
            />
          </label>

          <label className="flex items-center justify-between p-3 bg-surface-subtle rounded-lg">
            <div className="flex items-center gap-3">
              {preferences.theme === 'dark' ? (
                <FiMoon className="h-5 w-5 text-text-muted" />
              ) : (
                <FiSun className="h-5 w-5 text-text-muted" />
              )}
              <div>
                <p className="font-medium text-text">Theme</p>
                <p className="text-sm text-text-muted">Choose your preferred theme</p>
              </div>
            </div>
            <select
              disabled={!isEditing}
              value={preferences.theme || 'light'}
              onChange={(e) => setPreferences({ ...preferences, theme: e.target.value as 'light' | 'dark' })}
              className="px-3 py-1 border border-border rounded-lg text-sm focus:ring-primary focus:border-primary disabled:opacity-50"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </label>
        </div>
      </div>
    </div>
  )
}
