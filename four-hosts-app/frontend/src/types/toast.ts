// Toast types and constants - separate file to avoid fast refresh issues
export interface Toast {
  id: string
  type: 'success' | 'error' | 'info'
  message: string
  duration?: number
}

export const TOAST_ICONS = {
  success: 'CheckCircle',
  error: 'XCircle',
  info: 'AlertCircle'
} as const

export const TOAST_STYLES = {
  success: 'bg-green-50 text-green-800 border-green-200',
  error: 'bg-red-50 text-red-800 border-red-200',
  info: 'bg-blue-50 text-blue-800 border-blue-200'
} as const

export const TOAST_GLOW_STYLES = {
  success: 'success-glow',
  error: 'error-glow',
  info: ''
} as const
