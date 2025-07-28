import { useState, useRef, useEffect } from 'react'
import { Search, Sparkles } from 'lucide-react'

interface ResearchFormProps {
  onSubmit: (query: string) => void
  isLoading: boolean
}

function ResearchForm({ onSubmit, isLoading }: ResearchFormProps) {
  const [query, setQuery] = useState('')
  const [isFocused, setIsFocused] = useState(false)
  const [charCount, setCharCount] = useState(0)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim().length >= 10) {
      onSubmit(query)
      // Add success animation
      const form = e.currentTarget
      form.classList.add('animate-scale-in')
      setTimeout(() => {
        form.classList.remove('animate-scale-in')
      }, 300)
    }
  }

  const handleQueryChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newQuery = e.target.value
    setQuery(newQuery)
    setCharCount(newQuery.trim().length)
    
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }

  useEffect(() => {
    // Initial height adjustment
    if (textareaRef.current) {
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [])
  
  return (
    <form 
      onSubmit={handleSubmit} 
      className={`bg-white shadow-lg rounded-lg p-6 transition-all duration-300 hover-lift ${
        isFocused ? 'ring-2 ring-blue-500/50' : ''
      }`}
    >
      <div>
        <div className="flex items-center justify-between mb-2">
          <label htmlFor="query" className="block text-sm font-medium text-gray-700">
            Research Query
          </label>
          <span className={`text-xs transition-colors duration-200 ${
            charCount < 10 ? 'text-gray-400' : 'text-green-600'
          }`}>
            {charCount}/10 {charCount >= 10 && '✓'}
          </span>
        </div>
        <div className="relative">
          <textarea
            ref={textareaRef}
            id="query"
            name="query"
            rows={3}
            className={`input-field resize-none transition-all duration-200 ${
              isFocused ? 'shadow-lg' : ''
            } ${
              charCount >= 10 ? 'border-green-500 focus:ring-green-500' : ''
            }`}
            placeholder="Enter your research question (minimum 10 characters)..."
            value={query}
            onChange={handleQueryChange}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            disabled={isLoading}
            style={{ minHeight: '80px' }}
          />
          {isFocused && (
            <div className="absolute -top-2 -right-2 animate-pulse">
              <Sparkles className="h-5 w-5 text-blue-500" />
            </div>
          )}
        </div>
        <p className="mt-2 text-sm text-gray-500 animate-fade-in">
          Ask any question and our AI will classify it into one of four paradigms for targeted research.
        </p>
      </div>
      
      <div className="mt-4 flex items-center space-x-4">
        <button
          type="submit"
          disabled={isLoading || query.trim().length < 10}
          className="btn-primary flex items-center space-x-2 group"
        >
          <Search className={`h-4 w-4 transition-transform duration-300 ${
            isLoading ? 'animate-spin' : 'group-hover:scale-110'
          }`} />
          <span>
            {isLoading ? (
              <span className="flex items-center">
                Researching<span className="loading-dots"></span>
              </span>
            ) : (
              'Start Research'
            )}
          </span>
        </button>
        {isLoading && (
          <div className="flex items-center text-sm text-gray-600 animate-fade-in">
            <div className="loading-spinner mr-2"></div>
            <span>Analyzing your query...</span>
          </div>
        )}
      </div>
    </form>
  )
}

export default ResearchForm