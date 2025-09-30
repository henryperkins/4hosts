import React from 'react'
import { FiExternalLink } from 'react-icons/fi'
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card'
import { Button } from '../ui/Button'
import { getCredibilityIcon } from '../../utils/research-display'

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
  const safeQuotes = React.useMemo(() => (Array.isArray(quotes) ? quotes : []), [quotes])
  const hasQuotes = safeQuotes.length > 0

  const locale = React.useMemo(() => (
    typeof navigator !== 'undefined' && navigator.language ? navigator.language : 'en-US'
  ), [])

  const list = React.useMemo(() => {
    if (showAll) return safeQuotes
    return safeQuotes.slice(0, maxInitial)
  }, [showAll, safeQuotes, maxInitial])

  const computeKey = React.useCallback((quote: EvidenceQuote) => {
    if (quote.id) return quote.id
    const rawKey = `${quote.quote || ''}|${quote.url || ''}`
    let encoded: string | null = null
    try {
      encoded = btoa(rawKey)
    } catch {
      try {
        encoded = btoa(unescape(encodeURIComponent(rawKey)))
      } catch {
        encoded = null
      }
    }
    if (encoded) {
      return `quote-${encoded.replace(/[^a-zA-Z0-9]/g, '').slice(0, 16)}`
    }
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
      return crypto.randomUUID()
    }
    let hash = 0
    for (let i = 0; i < rawKey.length; i += 1) {
      hash = (hash << 5) - hash + rawKey.charCodeAt(i)
      hash |= 0
    }
    return `quote-${Math.abs(hash).toString(36)}`
  }, [])

  if (!hasQuotes) return null

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
          {list.map((q) => {
            const stableKey = computeKey(q)
            return (
            <li key={stableKey} className="rounded-md border border-border p-3 bg-surface">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="flex-1">
                  <p className="text-sm text-text">"{q.quote}"</p>
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-text-muted">
                    {q.credibility_score !== undefined && getCredibilityIcon(q.credibility_score)}
                    {q.domain ? <span className="font-medium text-text">{q.domain}</span> : null}
                    {q.published_date ? <span>• {new Date(q.published_date).toLocaleDateString(locale)}</span> : null}
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
