import React from 'react'
import { FiExternalLink, FiShield, FiAlertTriangle, FiAlertCircle } from 'react-icons/fi'
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card'
import { Button } from './ui/Button'
import { getCredibilityBand } from '../utils/credibility'

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
    const band = getCredibilityBand(score)
    if (band === 'high') return <FiShield className="h-3.5 w-3.5 text-success" aria-label="High credibility" />
    if (band === 'medium') return <FiAlertTriangle className="h-3.5 w-3.5 text-primary" aria-label="Medium credibility" />
    return <FiAlertCircle className="h-3.5 w-3.5 text-error" aria-label="Low credibility" />
  }

  return (
    <Card className="mt-4">
      <CardHeader className="mb-3">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <CardTitle className="text-sm">Evidence quotes</CardTitle>
          <span className="text-xs text-text-muted">{quotes.length} total</span>
        </div>
      </CardHeader>
      <CardContent>
        <ul className="space-y-3">
          {list.map((q, idx) => {
            // Generate a reasonably stable but safe key.  We attempt to base-64
            // encode the first ~30 chars of the quote + URL, but `btoa` only
            // supports Latin1 input.  If the quote contains non-ASCII
            // characters we fall back to `encodeURIComponent` -> `btoa` to
            // ensure the operation does not throw.  If *that* still fails, we
            // degrade gracefully to a simple incremental key.

            const rawKey = (q.quote || '').slice(0, 30) + (q.url || '')
            let encoded: string
            try {
              encoded = btoa(rawKey)
            } catch {
              try {
                encoded = btoa(unescape(encodeURIComponent(rawKey)))
              } catch {
                encoded = `fallback-${idx}`
              }
            }
            const stableKey = q.id || `quote-${encoded.replace(/[^a-zA-Z0-9]/g, '').slice(0, 10)}-${idx}`
            return (
            <li key={stableKey} className="rounded-md border border-border p-3 bg-surface">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="flex-1">
                  <p className="text-sm text-text">“{q.quote}”</p>
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-text-muted">
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
                    className="inline-flex items-center text-xs text-primary hover:opacity-80 whitespace-nowrap self-start sm:self-auto"
                    aria-label="Open source in new tab"
                  >
                    Source <FiExternalLink className="h-3.5 w-3.5 ml-1" />
                  </a>
                )}
              </div>
            </li>
          )})}
        </ul>
        {quotes.length > maxInitial && (
          <div className="mt-3 flex justify-center sm:justify-start">
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
