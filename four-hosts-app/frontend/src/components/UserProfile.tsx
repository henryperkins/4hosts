import React, { useState } from 'react'
import { User, Settings, Save, LogOut, Moon, Sun, Database, Brain } from 'lucide-react'
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
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">User Profile</h2>
        <button
          onClick={handleLogout}
          className="flex items-center gap-2 px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
        >
          <LogOut className="h-4 w-4" />
          Logout
        </button>
      </div>

      {/* User Info */}
      <div className="mb-8 p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
            <User className="h-8 w-8 text-blue-600" />
          </div>
          <div>
            <h3 className="font-semibold text-lg text-gray-900">{user.username}</h3>
            <p className="text-gray-600">{user.email}</p>
            <p className="text-sm text-gray-500">
              Member since {user.created_at ? new Date(user.created_at).toLocaleDateString() : 'Unknown'}
            </p>
          </div>
        </div>
      </div>

      {/* Preferences */}
      <div className="space-y-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Preferences</h3>
          {!isEditing ? (
            <button
              onClick={() => setIsEditing(true)}
              className="flex items-center gap-2 px-4 py-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
            >
              <Settings className="h-4 w-4" />
              Edit
            </button>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={() => setIsEditing(false)}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={isSaving}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                <Save className="h-4 w-4" />
                {isSaving ? 'Saving...' : 'Save'}
              </button>
            </div>
          )}
        </div>

        {/* Default Paradigm */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
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
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
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
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              } ${!isEditing ? 'cursor-not-allowed opacity-60' : ''}`}
            >
              Auto-detect
            </button>
          </div>
        </div>

        {/* Default Research Depth */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
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
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                } ${!isEditing ? 'cursor-not-allowed opacity-60' : ''}`}
              >
                {depth.charAt(0).toUpperCase() + depth.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Feature Toggles */}
        <div className="space-y-3">
          <label className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-3">
              <Database className="h-5 w-5 text-gray-600" />
              <div>
                <p className="font-medium text-gray-900">Enable Real Search</p>
                <p className="text-sm text-gray-600">Use live search APIs for current data</p>
              </div>
            </div>
            <input
              type="checkbox"
              disabled={!isEditing}
              checked={preferences.enable_real_search || false}
              onChange={(e) => setPreferences({ ...preferences, enable_real_search: e.target.checked })}
              className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded disabled:opacity-50"
            />
          </label>

          <label className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-3">
              <Brain className="h-5 w-5 text-gray-600" />
              <div>
                <p className="font-medium text-gray-900">Enable AI Classification</p>
                <p className="text-sm text-gray-600">Use advanced AI for paradigm detection</p>
              </div>
            </div>
            <input
              type="checkbox"
              disabled={!isEditing}
              checked={preferences.enable_ai_classification || false}
              onChange={(e) => setPreferences({ ...preferences, enable_ai_classification: e.target.checked })}
              className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded disabled:opacity-50"
            />
          </label>

          <label className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-3">
              {preferences.theme === 'dark' ? (
                <Moon className="h-5 w-5 text-gray-600" />
              ) : (
                <Sun className="h-5 w-5 text-gray-600" />
              )}
              <div>
                <p className="font-medium text-gray-900">Theme</p>
                <p className="text-sm text-gray-600">Choose your preferred theme</p>
              </div>
            </div>
            <select
              disabled={!isEditing}
              value={preferences.theme || 'light'}
              onChange={(e) => setPreferences({ ...preferences, theme: e.target.value as 'light' | 'dark' })}
              className="px-3 py-1 border border-gray-300 rounded-lg text-sm focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
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