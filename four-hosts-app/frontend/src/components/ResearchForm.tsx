import { useState } from 'react'
import { Search, Sparkles } from 'lucide-react'
import { Button } from './ui/Button'
import { InputField } from './ui/InputField'

interface ResearchFormProps {
  onSubmit: (query: string) => void
  isLoading: boolean
}

function ResearchForm({ onSubmit, isLoading }: ResearchFormProps) {
  const [query, setQuery] = useState('')
  const [isFocused, setIsFocused] = useState(false)
  const [charCount, setCharCount] = useState(0)
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim().length >= 10) {
      onSubmit(query)
      // Add success animation
      const form = e.currentTarget
      form.classList.add('animate-fade-in')
      setTimeout(() => {
        form.classList.remove('animate-fade-in')
      }, 300)
    }
  }

  const handleQueryChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const newQuery = e.target.value
    setQuery(newQuery)
    setCharCount(newQuery.trim().length)
    
    // textarea now relies on CSS `field-sizing-content` for auto-size
  }


  
  return (
    <form 
      onSubmit={handleSubmit} 
      className={`card-hover ${
        isFocused ? 'ring-2 ring-primary/50' : ''
      }`}
    >
      <div>
        <div className="flex items-center justify-between mb-2">
          <label htmlFor="query" className="block text-sm font-medium text-text">
            Research Query
          </label>
          <span className={`text-xs transition-colors ${
            charCount < 10 ? 'text-text-muted' : 'text-success'
          }`}>
            {charCount}/10 {charCount >= 10 && '✓'}
          </span>
        </div>
        <div className="relative">
          <InputField
            id="query"
            name="query"
            textarea
            rows={3}
            className={charCount >= 10 ? 'border-success focus:ring-success' : ''}
            placeholder="Enter your research question (minimum 10 characters)..."
            value={query}
            onChange={handleQueryChange}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            disabled={isLoading}
          />
          {isFocused && (
            <div className="absolute -top-2 -right-2 animate-pulse">
              <Sparkles className="h-5 w-5 text-primary" />
            </div>
          )}
        </div>
        <p className="mt-2 text-sm text-text-subtle animate-fade-in">
          Ask any question and our AI will classify it into one of four paradigms for targeted research.
        </p>
      </div>
      
      <div className="mt-4 flex items-center space-x-4">
        <Button
          type="submit"
          disabled={query.trim().length < 10}
          loading={isLoading}
          icon={Search}
        >
          {isLoading ? 'Researching' : 'Start Research'}
        </Button>
        {isLoading && (
          <div className="flex items-center text-sm text-text-subtle animate-fade-in">
            <div className="spinner mr-2"></div>
            <span>Analyzing your query...</span>
          </div>
        )}
      </div>
    </form>
  )
}

export default ResearchForm