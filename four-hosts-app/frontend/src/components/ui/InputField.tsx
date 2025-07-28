import { forwardRef } from 'react'
import type { InputHTMLAttributes, TextareaHTMLAttributes } from 'react'

type BaseProps = {
  label?: string
  textarea?: boolean
  className?: string
}

export type InputFieldProps = 
  (InputHTMLAttributes<HTMLInputElement> | TextareaHTMLAttributes<HTMLTextAreaElement>) &
  BaseProps

export const InputField = forwardRef<
  HTMLInputElement | HTMLTextAreaElement,
  InputFieldProps
>(({ label, textarea = false, className = '', ...props }, ref) => {
  const Component: any = textarea ? 'textarea' : 'input'
  
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
      <Component
        ref={ref}
        className={`input-field ${textarea ? 'resize-none' : ''} ${className}`}
        {...props}
      />
    </div>
  )
})
InputField.displayName = 'InputField'
