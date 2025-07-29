import { forwardRef, useId } from 'react'
import type { InputHTMLAttributes, TextareaHTMLAttributes } from 'react'

type BaseProps = {
  label?: string
  textarea?: boolean
  className?: string
  status?: 'error' | 'success'
  errorMessage?: string
  successMessage?: string
  hint?: string
}

export type InputFieldProps = 
  (InputHTMLAttributes<HTMLInputElement> | TextareaHTMLAttributes<HTMLTextAreaElement>) &
  BaseProps

export const InputField = forwardRef<
  HTMLInputElement | HTMLTextAreaElement,
  InputFieldProps
>(({ 
  label, 
  textarea = false, 
  className = '', 
  status, 
  errorMessage, 
  successMessage,
  hint,
  ...props 
}, ref) => {
  const generatedId = useId()
  const inputId = props.id || generatedId
  const descriptionId = `${inputId}-description`
  const errorId = `${inputId}-error`
  
  const statusClass = 
    status === 'error' 
      ? 'input-error animate-pulse-border' 
      : status === 'success' 
      ? 'input-success' 
      : ''
  
  const ariaProps = {
    'aria-invalid': status === 'error' ? true : undefined,
    'aria-describedby': [
      hint && descriptionId,
      status === 'error' && errorMessage && errorId,
      status === 'success' && successMessage && descriptionId
    ].filter(Boolean).join(' ') || undefined
  }
  
  return (
    <div className="space-y-1">
      {label && (
        <label
          htmlFor={inputId}
          className="block text-sm font-medium text-text"
        >
          {label}
          {props.required && (
            <span className="text-red-500 ml-1" aria-label="required">*</span>
          )}
        </label>
      )}
      
      {hint && !status && (
        <p id={descriptionId} className="text-sm text-text-muted">
          {hint}
        </p>
      )}
      
      {textarea ? (
        <textarea
          ref={ref as React.Ref<HTMLTextAreaElement>}
          id={inputId}
          className={`input-field resize-none ${statusClass} ${className}`}
          {...ariaProps}
          {...(props as TextareaHTMLAttributes<HTMLTextAreaElement>)}
        />
      ) : (
        <input
          ref={ref as React.Ref<HTMLInputElement>}
          id={inputId}
          className={`input-field ${statusClass} ${className}`}
          {...ariaProps}
          {...(props as InputHTMLAttributes<HTMLInputElement>)}
        />
      )}
      
      {status === 'error' && errorMessage && (
        <p id={errorId} className="text-sm text-red-600 dark:text-red-400 flex items-center gap-1" role="alert">
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {errorMessage}
        </p>
      )}
      
      {status === 'success' && successMessage && (
        <p id={descriptionId} className="text-sm text-green-600 dark:text-green-400 flex items-center gap-1">
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {successMessage}
        </p>
      )}
    </div>
  )
})
InputField.displayName = 'InputField'
