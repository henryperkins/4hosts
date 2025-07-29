import React, { useState } from 'react'
import { 
  AlertCircle, 
  CheckCircle, 
  Info, 
  XCircle, 
  X,
  type LucideIcon 
} from 'lucide-react'

interface AlertProps {
  variant: 'info' | 'success' | 'warning' | 'error'
  title?: string
  children: React.ReactNode
  icon?: LucideIcon | false
  dismissible?: boolean
  onDismiss?: () => void
  className?: string
}

export const Alert: React.FC<AlertProps> = ({
  variant,
  title,
  children,
  icon,
  dismissible = false,
  onDismiss,
  className = ''
}) => {
  const [isVisible, setIsVisible] = useState(true)

  if (!isVisible) return null

  const variantStyles = {
    info: {
      container: 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800',
      icon: 'text-blue-600 dark:text-blue-400',
      title: 'text-blue-800 dark:text-blue-200',
      content: 'text-blue-700 dark:text-blue-300',
      defaultIcon: Info
    },
    success: {
      container: 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800',
      icon: 'text-green-600 dark:text-green-400',
      title: 'text-green-800 dark:text-green-200',
      content: 'text-green-700 dark:text-green-300',
      defaultIcon: CheckCircle
    },
    warning: {
      container: 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800',
      icon: 'text-yellow-600 dark:text-yellow-400',
      title: 'text-yellow-800 dark:text-yellow-200',
      content: 'text-yellow-700 dark:text-yellow-300',
      defaultIcon: AlertCircle
    },
    error: {
      container: 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800',
      icon: 'text-red-600 dark:text-red-400',
      title: 'text-red-800 dark:text-red-200',
      content: 'text-red-700 dark:text-red-300',
      defaultIcon: XCircle
    }
  }

  const styles = variantStyles[variant]
  const Icon = icon === false ? null : (icon || styles.defaultIcon)

  const handleDismiss = () => {
    setIsVisible(false)
    onDismiss?.()
  }

  return (
    <div
      role="alert"
      className={`
        relative border rounded-lg p-4 animate-fade-in
        ${styles.container}
        ${className}
      `}
    >
      <div className="flex">
        {Icon && (
          <div className="flex-shrink-0">
            <Icon className={`h-5 w-5 ${styles.icon}`} aria-hidden="true" />
          </div>
        )}
        
        <div className={`${Icon ? 'ml-3' : ''} flex-1`}>
          {title && (
            <h3 className={`text-sm font-medium ${styles.title} mb-1`}>
              {title}
            </h3>
          )}
          <div className={`text-sm ${styles.content}`}>
            {children}
          </div>
        </div>

        {dismissible && (
          <div className="ml-auto pl-3">
            <button
              onClick={handleDismiss}
              className={`
                inline-flex rounded-md p-1.5 
                ${styles.icon} 
                hover:bg-black/10 dark:hover:bg-white/10
                focus-visible:outline-none focus-visible:ring-2 
                focus-visible:ring-offset-2 focus-visible:ring-blue-500
                transition-colors duration-200
              `}
              aria-label="Dismiss alert"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

// Alert Dialog for important messages
interface AlertDialogProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
  title: string
  description: string
  confirmText?: string
  cancelText?: string
  variant?: 'info' | 'warning' | 'danger'
}

export const AlertDialog: React.FC<AlertDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  description,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'warning'
}) => {
  if (!isOpen) return null

  const variantStyles = {
    info: 'text-blue-600',
    warning: 'text-yellow-600',
    danger: 'text-red-600'
  }

  const variantButtons = {
    info: 'btn-primary',
    warning: 'bg-yellow-600 hover:bg-yellow-700 text-white',
    danger: 'bg-red-600 hover:bg-red-700 text-white'
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm animate-fade-in" 
        onClick={onClose}
        aria-hidden="true"
      />
      
      <div 
        role="alertdialog"
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
        className="relative w-full max-w-md bg-surface rounded-xl shadow-2xl animate-scale-in p-6"
      >
        <div className="flex items-start gap-4">
          <div className={`p-2 rounded-full bg-surface-muted ${variantStyles[variant]}`}>
            <AlertCircle className="h-6 w-6" />
          </div>
          
          <div className="flex-1">
            <h3 id="alert-dialog-title" className="text-lg font-semibold text-text mb-2">
              {title}
            </h3>
            <p id="alert-dialog-description" className="text-text-muted">
              {description}
            </p>
          </div>
        </div>
        
        <div className="flex gap-3 justify-end mt-6">
          <button
            onClick={onClose}
            className="btn-secondary"
          >
            {cancelText}
          </button>
          <button
            onClick={() => {
              onConfirm()
              onClose()
            }}
            className={`btn ${variantButtons[variant]}`}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}