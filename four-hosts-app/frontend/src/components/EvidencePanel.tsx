import React from 'react'
import { FiExternalLink, FiShield, FiAlertTriangle } from 'react-icons/fi'
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card'
import { Button } from './ui/Button'

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
    if (score >= 0.8) return <FiShield className="h-3.5 w-3.5 text-success" aria-label="High credibility" />
    if (score >= 0.6) return <FiAlertTriangle className="h-3.5 w-3.5 text-primary" aria-label="Moderate credibility" />
    return <FiAlertTriangle className="h-3.5 w-3.5 text-error" aria-label="Low credibility" />
  }

  return (
    <Card className="mt-4">
      <CardHeader className="mb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Evidence quotes</CardTitle>
          <span className="text-xs text-text-muted">{quotes.length} total</span>
        </div>
      </CardHeader>
      <CardContent>
        <ul className="space-y-3">
          {list.map((q, idx) => (
            <li key={q.id || idx} className="rounded-md border border-border p-3 bg-surface">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <p className="text-sm text-text">“{q.quote}”</p>
                  <div className="mt-2 flex items-center gap-2 text-xs text-text-muted">
                    {credibilityIcon(q.credibility_score)}
                    {q.domain ? <span className="font-medium text-text">{q.domain}</span> : null}
                    {q.published_date ? <span>• {new Date(q.published_date).toLocaleDateString()}</span> : null}
                  </div>
                </div>
                {q.url && (
                  <a
                    href={q.url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center text-xs text-primary hover:opacity-80 whitespace-nowrap"
                    aria-label="Open source in new tab"
                  >
                    Source <FiExternalLink className="h-3.5 w-3.5 ml-1" />
                  </a>
                )}
              </div>
            </li>
          ))}
        </ul>
        {quotes.length > maxInitial && (
          <div className="mt-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAll(!showAll)}
              className="text-text-muted hover:text-text"
            >
              {showAll ? 'Show fewer quotes' : `Show all ${quotes.length}`}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default EvidencePanel
