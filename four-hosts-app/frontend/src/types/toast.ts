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
  success: 'bg-success/15 text-success border border-success/30',
  error: 'bg-error/15 text-error border border-error/30',
  info: 'bg-primary/15 text-primary border border-primary/30'
} as const

export const TOAST_GLOW_STYLES = {
  success: 'success-glow',
  error: 'error-glow',
  info: ''
} as const
