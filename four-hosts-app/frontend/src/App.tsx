import { useState } from 'react'
import ResearchForm from './components/ResearchForm'
import ParadigmDisplay from './components/ParadigmDisplay'
import ResultsDisplay from './components/ResultsDisplay'
import type { ResearchResult, ParadigmClassification } from './types'

function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [paradigmClassification, setParadigmClassification] = useState<ParadigmClassification | null>(null)
  const [results, setResults] = useState<ResearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (query: string) => {
    setIsLoading(true)
    setError(null)
    setResults(null)
    
    try {
      // Submit research query
      const response = await fetch('http://localhost:8000/research/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          options: {
            depth: 'standard',
            include_secondary: true,
            max_sources: 100,
          }
        }),
      })
      
      if (!response.ok) {
        throw new Error('Failed to submit research query')
      }
      
      const data = await response.json()
      setParadigmClassification(data.paradigm_classification)
      
      // Poll for results
      const researchId = data.research_id
      let retries = 0
      const maxRetries = 30
      
      const pollInterval = setInterval(async () => {
        try {
          const resultsResponse = await fetch(`http://localhost:8000/research/results/${researchId}`)
          
          if (resultsResponse.ok) {
            const resultsData = await resultsResponse.json()
            setResults(resultsData)
            setIsLoading(false)
            clearInterval(pollInterval)
          }
        } catch (err) {
          // Continue polling
        }
        
        retries++
        if (retries >= maxRetries) {
          setError('Research timeout - please try again')
          setIsLoading(false)
          clearInterval(pollInterval)
        }
      }, 1000)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-3xl font-bold text-gray-900">Four Hosts Research System</h1>
          <p className="mt-1 text-sm text-gray-600">Paradigm-aware research for deeper insights</p>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <ResearchForm onSubmit={handleSubmit} isLoading={isLoading} />
        
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800">{error}</p>
          </div>
        )}
        
        {paradigmClassification && (
          <ParadigmDisplay classification={paradigmClassification} />
        )}
        
        {isLoading && (
          <div className="mt-8 text-center">
            <div className="inline-flex items-center">
              <svg className="animate-spin h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span className="ml-3 text-lg text-gray-700">Conducting paradigm-aware research...</span>
            </div>
          </div>
        )}
        
        {results && (
          <ResultsDisplay results={results} />
        )}
      </main>
    </div>
  )
}

export default App