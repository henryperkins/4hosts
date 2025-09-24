import React, { useId } from 'react'

interface ToggleSwitchProps {
  checked: boolean
  onChange: (checked: boolean) => void
  label?: string
  labelPosition?: 'left' | 'right'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  variant?: 'default' | 'success' | 'danger'
  className?: string
  id?: string
  'aria-label'?: string
  'aria-describedby'?: string
}

export const ToggleSwitch: React.FC<ToggleSwitchProps> = ({
  checked,
  onChange,
  label,
  labelPosition = 'right',
  size = 'md',
  disabled = false,
  variant = 'default',
  className = '',
  id: providedId,
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedBy
}) => {
  const generatedId = useId()
  const switchId = providedId || generatedId

  const sizeClasses = {
    sm: {
      switch: 'h-5 w-9',
      thumb: 'h-4 w-4',
      translate: 'translate-x-4'
    },
    md: {
      switch: 'h-6 w-11',
      thumb: 'h-5 w-5',
      translate: 'translate-x-5'
    },
    lg: {
      switch: 'h-7 w-14',
      thumb: 'h-6 w-6',
      translate: 'translate-x-7'
    }
  }

  const variantClasses = {
    default: {
      checked: 'bg-primary',
      unchecked: 'bg-surface-muted'
    },
    success: {
      checked: 'bg-success',
      unchecked: 'bg-surface-muted'
    },
    danger: {
      checked: 'bg-error',
      unchecked: 'bg-surface-muted'
    }
  }

  const currentSize = sizeClasses[size]
  const currentVariant = variantClasses[variant]

  const handleChange = () => {
    if (!disabled) {
      onChange(!checked)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault()
      handleChange()
    }
  }

  const switchElement = (
    <button
      id={switchId}
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={ariaLabel || label}
      aria-describedby={ariaDescribedBy}
      disabled={disabled}
      onClick={handleChange}
      onKeyDown={handleKeyDown}
      className={`
        relative inline-flex items-center rounded-full
        transition-colors duration-200 ease-in-out
        focus-visible:outline-none focus-visible:ring-2 
        focus-visible:ring-primary focus-visible:ring-offset-2
        disabled:opacity-50 disabled:cursor-not-allowed
        ${currentSize.switch}
        ${checked ? currentVariant.checked : currentVariant.unchecked}
        ${className}
      `}
    >
      <span
        className={`
          inline-block rounded-full bg-white
          transform transition-transform duration-200 ease-in-out
          ${currentSize.thumb}
          ${checked ? currentSize.translate : 'translate-x-0.5'}
        `}
      />
    </button>
  )

  if (!label) {
    return switchElement
  }

  return (
    <label 
      htmlFor={switchId}
      className={`
        inline-flex items-center gap-3
        ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}
      `}
    >
      {labelPosition === 'left' && (
        <span className="text-text select-none">{label}</span>
      )}
      {switchElement}
      {labelPosition === 'right' && (
        <span className="text-text select-none">{label}</span>
      )}
    </label>
  )
}

// Group of toggle switches with a shared label
interface ToggleSwitchGroupProps {
  label: string
  children: React.ReactNode
  className?: string
}

export const ToggleSwitchGroup: React.FC<ToggleSwitchGroupProps> = ({
  label,
  children,
  className = ''
}) => {
  return (
    <fieldset className={`space-y-3 ${className}`}>
      <legend className="text-base font-medium text-text mb-2">{label}</legend>
      <div className="space-y-2">
        {children}
      </div>
    </fieldset>
  )
}

// Icon toggle variant
interface IconToggleProps {
  checked: boolean
  onChange: (checked: boolean) => void
  checkedIcon: React.ReactNode
  uncheckedIcon: React.ReactNode
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  className?: string
  'aria-label': string
}

export const IconToggle: React.FC<IconToggleProps> = ({
  checked,
  onChange,
  checkedIcon,
  uncheckedIcon,
  size = 'md',
  disabled = false,
  className = '',
  'aria-label': ariaLabel
}) => {
  const sizeClasses = {
    sm: 'h-8 w-8 text-sm',
    md: 'h-10 w-10 text-base',
    lg: 'h-12 w-12 text-lg'
  }

  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={ariaLabel}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
      className={`
        inline-flex items-center justify-center
        rounded-lg transition-all duration-200
        ${checked 
          ? 'bg-primary/15 text-primary border border-primary/20' 
          : 'bg-surface-muted text-text-muted hover:bg-surface-subtle'
        }
        focus-visible:outline-none focus-visible:ring-2 
        focus-visible:ring-primary focus-visible:ring-offset-2
        disabled:opacity-50 disabled:cursor-not-allowed
        ${sizeClasses[size]}
        ${className}
      `}
    >
      {checked ? checkedIcon : uncheckedIcon}
    </button>
  )
}
