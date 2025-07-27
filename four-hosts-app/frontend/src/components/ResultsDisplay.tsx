import type { ResearchResult } from '../types'

interface ResultsDisplayProps {
  results: ResearchResult
}

const paradigmColors = {
  dolores: 'text-red-700 bg-red-50 border-red-200',
  teddy: 'text-orange-700 bg-orange-50 border-orange-200',
  bernard: 'text-blue-700 bg-blue-50 border-blue-200',
  maeve: 'text-green-700 bg-green-50 border-green-200'
}

function ResultsDisplay({ results }: ResultsDisplayProps) {
  return (
    <div className="mt-8 space-y-6">
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Research Results</h2>
        
        <div className="prose max-w-none">
          <h3 className="text-lg font-medium text-gray-900 mb-2">Summary</h3>
          <p className="text-gray-700">{results.answer.summary}</p>
        </div>
        
        <div className="mt-6 space-y-4">
          {results.answer.sections.map((section, index) => (
            <div key={index} className={`border rounded-lg p-4 ${paradigmColors[section.paradigm]}`}>
              <h4 className="font-medium mb-2">{section.title}</h4>
              <p className="text-sm">{section.content}</p>
              <div className="mt-2 text-xs opacity-75">
                {section.sources_count} sources analyzed • {(section.confidence * 100).toFixed(0)}% confidence
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {results.answer.action_items.length > 0 && (
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Action Items</h3>
          <div className="space-y-3">
            {results.answer.action_items.map((item, index) => (
              <div key={index} className="flex items-start">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  item.priority === 'high' ? 'bg-red-100 text-red-800' :
                  item.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {item.priority}
                </span>
                <div className="ml-3 flex-1">
                  <p className="text-sm font-medium text-gray-900">{item.action}</p>
                  <p className="text-xs text-gray-500">Timeframe: {item.timeframe}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Sources & Citations</h3>
        <div className="space-y-2">
          {results.answer.citations.map((citation) => (
            <div key={citation.id} className="border-l-2 border-gray-200 pl-4 py-2">
              <a
                href={citation.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-medium text-blue-600 hover:text-blue-800"
              >
                {citation.title}
              </a>
              <p className="text-xs text-gray-500">
                {citation.source} • Credibility: {(citation.credibility_score * 100).toFixed(0)}%
              </p>
            </div>
          ))}
        </div>
        
        <div className="mt-4 pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-500">
            Analyzed {results.metadata.total_sources_analyzed} sources • 
            {results.metadata.high_quality_sources} high-quality sources • 
            {results.metadata.search_queries_executed} searches • 
            {results.metadata.processing_time_seconds.toFixed(1)}s processing time
          </p>
        </div>
      </div>
    </div>
  )
}

export default ResultsDisplay