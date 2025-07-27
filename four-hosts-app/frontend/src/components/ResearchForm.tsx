import { useState } from 'react'

interface ResearchFormProps {
  onSubmit: (query: string) => void
  isLoading: boolean
}

function ResearchForm({ onSubmit, isLoading }: ResearchFormProps) {
  const [query, setQuery] = useState('')
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim().length >= 10) {
      onSubmit(query)
    }
  }
  
  return (
    <form onSubmit={handleSubmit} className="bg-white shadow rounded-lg p-6">
      <div>
        <label htmlFor="query" className="block text-sm font-medium text-gray-700">
          Research Query
        </label>
        <div className="mt-1">
          <textarea
            id="query"
            name="query"
            rows={3}
            className="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md p-3"
            placeholder="Enter your research question (minimum 10 characters)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
          />
        </div>
        <p className="mt-2 text-sm text-gray-500">
          Ask any question and our AI will classify it into one of four paradigms for targeted research.
        </p>
      </div>
      
      <div className="mt-4">
        <button
          type="submit"
          disabled={isLoading || query.trim().length < 10}
          className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Researching...' : 'Start Research'}
        </button>
      </div>
    </form>
  )
}

export default ResearchForm