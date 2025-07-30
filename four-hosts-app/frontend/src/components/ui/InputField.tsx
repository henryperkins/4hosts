import { forwardRef, useRef, useEffect, useId } from 'react'
import type { InputHTMLAttributes, TextareaHTMLAttributes } from 'react'

type BaseProps = {
  label?: string
  textarea?: boolean
  className?: string
  status?: 'error' | 'success'
  errorMessage?: string
  successMessage?: string
  hint?: string
  autoResize?: boolean
}

export type InputFieldProps = BaseProps & {
  onChange?: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => void
} & Omit<InputHTMLAttributes<HTMLInputElement> & TextareaHTMLAttributes<HTMLTextAreaElement>, 'onChange'>

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
  autoResize = true,
  id,
  ...props 
}, ref) => {
  const generatedId = useId()
  const inputId = id || generatedId
  const descriptionId = `${inputId}-description`
  const errorId = `${inputId}-error`
  const internalRef = useRef<HTMLTextAreaElement>(null)
  const textareaRef = textarea ? (ref as React.Ref<HTMLTextAreaElement>) || internalRef : null
  
  // Auto-resize textarea
  useEffect(() => {
    if (textarea && autoResize && textareaRef && 'current' in textareaRef && textareaRef.current) {
      const adjustHeight = () => {
        const element = textareaRef.current
        if (element) {
          element.style.height = 'auto'
          element.style.height = `${element.scrollHeight}px`
        }
      }
      
      adjustHeight()
      
      const handleInput = () => adjustHeight()
      const element = textareaRef.current
      element.addEventListener('input', handleInput)
      
      return () => {
        element.removeEventListener('input', handleInput)
      }
    }
  }, [textarea, autoResize, textareaRef, props.value])
  
  const statusClass = 
    status === 'error' 
      ? 'border-error focus:ring-error' 
      : status === 'success' 
      ? 'border-success focus:ring-success' 
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
          ref={textareaRef}
          id={inputId}
          className={`input resize-y min-h-[80px] text-base md:text-sm ${statusClass} ${className}`}
          style={autoResize ? { overflow: 'hidden' } : {}}
          {...ariaProps}
          {...(props as TextareaHTMLAttributes<HTMLTextAreaElement>)}
        />
      ) : (
        <input
          ref={ref as React.Ref<HTMLInputElement>}
          id={inputId}
          className={`input text-base md:text-sm ${statusClass} ${className}`}
          {...ariaProps}
          {...(props as InputHTMLAttributes<HTMLInputElement>)}
        />
      )}
      
      {status === 'error' && errorMessage && (
        <p id={errorId} className="text-sm text-error flex items-center gap-1" role="alert">
          <svg className="h-4 w-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>{errorMessage}</span>
        </p>
      )}
      
      {status === 'success' && successMessage && (
        <p id={descriptionId} className="text-sm text-success flex items-center gap-1">
          <svg className="h-4 w-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>{successMessage}</span>
        </p>
      )}
    </div>
  )
})
InputField.displayName = 'InputField'