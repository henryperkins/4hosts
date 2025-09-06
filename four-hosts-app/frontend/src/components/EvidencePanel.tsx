import React from 'react'
import { FiExternalLink, FiShield, FiAlertTriangle } from 'react-icons/fi'

interface EvidenceQuote {
  id?: string
  url: string
  title?: string
  domain?: string
  quote: string
  credibility_score?: number
  published_date?: string
}

interface EvidencePanelProps {
  quotes: EvidenceQuote[]
  maxInitial?: number
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({ quotes, maxInitial = 6 }) => {
  const [showAll, setShowAll] = React.useState(false)
  if (!Array.isArray(quotes) || quotes.length === 0) return null

  const list = showAll ? quotes : quotes.slice(0, maxInitial)

  const credibilityIcon = (score?: number) => {
    if (typeof score !== 'number') return null
    if (score >= 0.8) return <FiShield className="h-3.5 w-3.5 text-green-600" />
    if (score >= 0.6) return <FiAlertTriangle className="h-3.5 w-3.5 text-yellow-600" />
    return <FiAlertTriangle className="h-3.5 w-3.5 text-orange-600" />
  }

  return (
    <div className="mt-4 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800 transition-colors duration-200">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-semibold text-purple-900 dark:text-purple-100">Evidence quotes</h4>
        <span className="text-xs text-purple-700 dark:text-purple-300">{quotes.length} total</span>
      </div>
      <ul className="space-y-3">
        {list.map((q, idx) => (
          <li key={q.id || idx} className="bg-white dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-700 p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1">
                <p className="text-sm text-gray-900 dark:text-gray-100">“{q.quote}”</p>
                <div className="mt-2 flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                  {credibilityIcon(q.credibility_score)}
                  {q.domain ? <span className="font-medium">{q.domain}</span> : null}
                  {q.published_date ? <span>• {new Date(q.published_date).toLocaleDateString()}</span> : null}
                </div>
              </div>
              {q.url && (
                <a href={q.url} target="_blank" rel="noreferrer" className="inline-flex items-center text-xs text-blue-600 dark:text-blue-400 hover:underline whitespace-nowrap">
                  Source <FiExternalLink className="h-3.5 w-3.5 ml-1" />
                </a>
              )}
            </div>
          </li>
        ))}
      </ul>
      {quotes.length > maxInitial && (
        <div className="mt-3">
          <button
            onClick={() => setShowAll(!showAll)}
            className="text-xs text-purple-700 dark:text-purple-300 hover:underline"
          >
            {showAll ? 'Show fewer quotes' : `Show all ${quotes.length}`}
          </button>
        </div>
      )}
    </div>
  )
}

export default EvidencePanel

