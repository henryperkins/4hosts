import React from 'react'
import { FiShield, FiAlertTriangle, FiAlertCircle, FiCheckCircle, FiClock } from 'react-icons/fi'
import { getCredibilityBand } from './credibility'

export const getCredibilityIcon = (score: number): React.ReactNode => {
  const band = getCredibilityBand(score)
  if (band === 'high') return <FiShield className="h-4 w-4" aria-label="High credibility" />
  if (band === 'medium') return <FiAlertTriangle className="h-4 w-4" aria-label="Medium credibility" />
  return <FiAlertCircle className="h-4 w-4" aria-label="Low credibility" />
}

export const getPriorityIcon = (priority: string): React.ReactNode => {
  switch (priority) {
    case 'high':
      return <FiAlertCircle className="h-4 w-4 text-error" aria-label="High priority" />
    case 'medium':
      return <FiClock className="h-4 w-4 text-warning" aria-label="Medium priority" />
    default:
      return <FiCheckCircle className="h-4 w-4 text-success" aria-label="Low priority" />
  }
}

export const bottomLine = (text: string | undefined, maxWords = 20): string => {
  if (!text) return 'No summary available.'
  const firstSentence = text.split(/(?<=[.!?])\s+/)[0] || text
  const words = firstSentence.trim().split(/\s+/)
  if (words.length <= maxWords) return firstSentence.trim()
  return `${words.slice(0, maxWords).join(' ')}...`
}

export const parseExplanation = (
  expl?: string
): { bias?: string; fact?: string; cat?: string } => {
  if (!expl) return {}
  const out: { bias?: string; fact?: string; cat?: string } = {}
  const pairs = expl.split(',')
  for (const pair of pairs) {
    const [key, value] = pair.split('=').map(segment => (segment || '').trim())
    if (!key || !value) continue
    if (key.startsWith('bias')) out.bias = value
    if (key.startsWith('fact') || key.startsWith('factual')) out.fact = value
    if (key.startsWith('cat')) out.cat = value
  }
  return out
}
