import React, { useState, useRef, useEffect, forwardRef, useId } from 'react'
import { FiChevronDown, FiCheck } from 'react-icons/fi'

export interface SelectOption {
  value: string
  label: string
  disabled?: boolean
}

interface SelectProps {
  options: SelectOption[]
  value?: string
  onChange: (value: string) => void
  placeholder?: string
  label?: string
  error?: string
  disabled?: boolean
  className?: string
  id?: string
}

export const Select = forwardRef<HTMLButtonElement, SelectProps>(
  ({ 
    options, 
    value, 
    onChange, 
    placeholder = 'Select an option', 
    label,
    error,
    disabled = false,
    className = '',
    id: providedId
  }, ref) => {
    const [isOpen, setIsOpen] = useState(false)
    const containerRef = useRef<HTMLDivElement>(null)
    const listRef = useRef<HTMLUListElement>(null)
    const generatedId = useId()
    const selectId = providedId || generatedId
    const listboxId = `${selectId}-listbox`
    const errorId = `${selectId}-error`

    const selectedOption = options.find(opt => opt.value === value)

    // Close dropdown when clicking outside
    useEffect(() => {
      const handleClickOutside = (event: MouseEvent) => {
        if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
          setIsOpen(false)
        }
      }

      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    // Handle keyboard navigation
    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (disabled) return

      switch (e.key) {
        case 'Enter':
        case ' ':
          e.preventDefault()
          setIsOpen(!isOpen)
          break
        case 'Escape':
          e.preventDefault()
          setIsOpen(false)
          break
        case 'ArrowDown':
          e.preventDefault()
          if (!isOpen) {
            setIsOpen(true)
          } else {
            const currentIndex = options.findIndex(opt => opt.value === value)
            const nextIndex = currentIndex < options.length - 1 ? currentIndex + 1 : 0
            const nextOption = options[nextIndex]
            if (!nextOption.disabled) {
              onChange(nextOption.value)
            }
          }
          break
        case 'ArrowUp':
          e.preventDefault()
          if (isOpen) {
            const currentIndex = options.findIndex(opt => opt.value === value)
            const prevIndex = currentIndex > 0 ? currentIndex - 1 : options.length - 1
            const prevOption = options[prevIndex]
            if (!prevOption.disabled) {
              onChange(prevOption.value)
            }
          }
          break
      }
    }

    // Scroll selected option into view when dropdown opens
    useEffect(() => {
      if (isOpen && listRef.current && selectedOption) {
        const selectedElement = listRef.current.querySelector('[aria-selected="true"]')
        if (selectedElement) {
          selectedElement.scrollIntoView({ block: 'nearest' })
        }
      }
    }, [isOpen, selectedOption])

    return (
      <div ref={containerRef} className="relative">
        {label && (
          <label htmlFor={selectId} className="block text-sm font-medium text-text mb-1">
            {label}
          </label>
        )}
        
        <button
          ref={ref}
          id={selectId}
          type="button"
          role="combobox"
          aria-expanded={isOpen}
          aria-haspopup="listbox"
          aria-controls={listboxId}
          aria-labelledby={label ? undefined : selectId}
          aria-describedby={error ? errorId : undefined}
          aria-invalid={error ? true : undefined}
          disabled={disabled}
          onClick={() => !disabled && setIsOpen(!isOpen)}
          onKeyDown={handleKeyDown}
          className={`
            w-full px-3 py-2 text-left
            bg-surface border border-border rounded-md
            hover:border-border-subtle
            focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2
            disabled:opacity-50 disabled:cursor-not-allowed
            transition-all duration-200
            ${error ? 'border-red-500 focus-visible:ring-red-500' : ''}
            ${className}
          `}
        >
          <span className="flex items-center justify-between">
            <span className={selectedOption ? 'text-text' : 'text-text-muted'}>
              {selectedOption ? selectedOption.label : placeholder}
            </span>
            <FiChevronDown 
              className={`
                h-4 w-4 text-text-muted transition-transform duration-200
                ${isOpen ? 'rotate-180' : ''}
              `}
            />
          </span>
        </button>

        {error && (
          <p id={errorId} className="mt-1 text-sm text-red-600 dark:text-red-400" role="alert">
            {error}
          </p>
        )}

        {isOpen && (
          <ul
            ref={listRef}
            id={listboxId}
            role="listbox"
            aria-labelledby={label ? `${selectId}-label` : selectId}
            className="
              absolute z-50 w-full mt-1 
              bg-surface border border-border rounded-md shadow-lg
              max-h-60 overflow-auto
              animate-slide-down
              focus-visible:outline-none
            "
          >
            {options.map((option) => (
              <li
                key={option.value}
                role="option"
                aria-selected={option.value === value}
                aria-disabled={option.disabled}
                onClick={() => {
                  if (!option.disabled) {
                    onChange(option.value)
                    setIsOpen(false)
                  }
                }}
                className={`
                  px-3 py-2 cursor-pointer flex items-center justify-between
                  transition-colors duration-150
                  ${option.value === value 
                    ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' 
                    : 'hover:bg-surface-subtle'
                  }
                  ${option.disabled 
                    ? 'opacity-50 cursor-not-allowed' 
                    : ''
                  }
                `}
              >
                <span>{option.label}</span>
                {option.value === value && (
                  <FiCheck className="h-4 w-4" aria-hidden="true" />
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    )
  }
)

Select.displayName = 'Select'