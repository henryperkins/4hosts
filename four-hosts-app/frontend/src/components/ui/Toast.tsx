import React, { useEffect, useState, useCallback } from 'react'
import { CheckCircle, XCircle, AlertCircle, X } from 'lucide-react'
import type { Toast } from '../../types/toast'

interface ToastNotificationProps {
  toast: Toast
  onClose: (id: string) => void
}

export const ToastNotification: React.FC<ToastNotificationProps> = ({ toast, onClose }) => {
  const [isVisible, setIsVisible] = useState(false)
  const [isExiting, setIsExiting] = useState(false)

  const handleClose = useCallback(() => {
    setIsExiting(true)
    setTimeout(() => {
      onClose(toast.id)
    }, 300)
  }, [onClose, toast.id])

  useEffect(() => {
    // Trigger entrance animation
    requestAnimationFrame(() => {
      setIsVisible(true)
    })

    // Auto-dismiss
    const timer = setTimeout(() => {
      handleClose()
    }, toast.duration || 5000)

    return () => clearTimeout(timer)
  }, [toast.duration, handleClose])

  const icons = {
    success: <CheckCircle className="h-5 w-5 text-green-400" />,
    error: <XCircle className="h-5 w-5 text-red-400" />,
    info: <AlertCircle className="h-5 w-5 text-blue-400" />
  }

  const styles = {
    success: 'bg-green-50 text-green-800 border-green-200',
    error: 'bg-red-50 text-red-800 border-red-200',
    info: 'bg-blue-50 text-blue-800 border-blue-200'
  }

  const glowStyles = {
    success: 'success-glow',
    error: 'error-glow',
    info: ''
  }

  return (
    <div
      className={`
        flex items-center p-4 mb-4 rounded-lg border-2 shadow-lg
        transition-all duration-300 ease-out
        ${styles[toast.type]} ${glowStyles[toast.type]}
        ${isVisible && !isExiting ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}
      `}
    >
      <div className="shrink-0 animate-scale-in">
        {icons[toast.type]}
      </div>
      <div className="ml-3 flex-1">
        <p className="text-sm font-medium">{toast.message}</p>
      </div>
      <button
        onClick={handleClose}
        className="ml-4 shrink-0 inline-flex text-gray-400 hover:text-gray-600 focus:outline-none transition-colors duration-200"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}

interface ToastContainerProps {
  toasts: Toast[]
  onClose: (id: string) => void
}

export const ToastContainer: React.FC<ToastContainerProps> = ({ toasts, onClose }) => {
  return (
    <div className="fixed top-4 right-4 z-50 pointer-events-none">
      <div className="max-w-sm w-full pointer-events-auto">
        {toasts.map((toast) => (
          <ToastNotification key={toast.id} toast={toast} onClose={onClose} />
        ))}
      </div>
    </div>
  )
}
