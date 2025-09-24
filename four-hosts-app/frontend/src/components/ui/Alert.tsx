import React, { useState } from 'react'
import { 
  FiAlertCircle, 
  FiCheckCircle, 
  FiInfo, 
  FiXCircle, 
  FiX 
} from 'react-icons/fi'
import type { IconType } from 'react-icons'

interface AlertProps {
  variant: 'info' | 'success' | 'warning' | 'error'
  title?: string
  children: React.ReactNode
  icon?: IconType | false
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
      container: 'bg-primary/15 border border-primary/30',
      icon: 'text-primary',
      title: 'text-primary',
      content: 'text-text',
      defaultIcon: FiInfo
    },
    success: {
      container: 'bg-success/15 border border-success/30',
      icon: 'text-success',
      title: 'text-success',
      content: 'text-text',
      defaultIcon: FiCheckCircle
    },
    warning: {
      container: 'bg-warning/15 border border-warning/30',
      icon: 'text-warning',
      title: 'text-warning',
      content: 'text-text',
      defaultIcon: FiAlertCircle
    },
    error: {
      container: 'bg-error/15 border border-error/30',
      icon: 'text-error',
      title: 'text-error',
      content: 'text-text',
      defaultIcon: FiXCircle
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
                hover:bg-surface-muted
                focus-visible:outline-none focus-visible:ring-2 
                focus-visible:ring-offset-2 focus-visible:ring-primary
                transition-colors duration-200
              `}
              aria-label="Dismiss alert"
            >
              <FiX className="h-4 w-4" />
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
    info: 'text-primary',
    warning: 'text-warning',
    danger: 'text-error'
  }

  const variantButtons = {
    info: 'btn btn-primary',
    warning: 'btn btn-warning',
    danger: 'btn btn-danger'
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
            <FiAlertCircle className="h-6 w-6" />
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
            className="btn btn-secondary"
          >
            {cancelText}
          </button>
          <button
            onClick={() => {
              onConfirm()
              onClose()
            }}
            className={variantButtons[variant]}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}
