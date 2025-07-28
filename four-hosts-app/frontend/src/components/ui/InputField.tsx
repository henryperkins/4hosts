import { forwardRef } from 'react'
import type { InputHTMLAttributes, TextareaHTMLAttributes } from 'react'

type BaseProps = {
  label?: string
  textarea?: boolean
  className?: string
  status?: 'error' | 'success'
}

export type InputFieldProps = 
  (InputHTMLAttributes<HTMLInputElement> | TextareaHTMLAttributes<HTMLTextAreaElement>) &
  BaseProps

export const InputField = forwardRef<
  HTMLInputElement | HTMLTextAreaElement,
  InputFieldProps
>(({ label, textarea = false, className = '', status, ...props }, ref) => {
  const statusClass = 
    status === 'error' 
      ? 'input-error animate-pulse-border' 
      : status === 'success' 
      ? 'input-success' 
      : ''
  
  return (
    <div className="space-y-1">
      {label && (
        <label
          htmlFor={props.id}
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          {label}
        </label>
      )}
      {textarea ? (
        <textarea
          ref={ref as React.Ref<HTMLTextAreaElement>}
          className={`input-field resize-none ${statusClass} ${className}`}
          {...(props as TextareaHTMLAttributes<HTMLTextAreaElement>)}
        />
      ) : (
        <input
          ref={ref as React.Ref<HTMLInputElement>}
          className={`input-field ${statusClass} ${className}`}
          {...(props as InputHTMLAttributes<HTMLInputElement>)}
        />
      )}
    </div>
  )
})
InputField.displayName = 'InputField'
