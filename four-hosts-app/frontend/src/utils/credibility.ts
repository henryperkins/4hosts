/**
 * Standardized credibility thresholds and utilities
 * Ensures consistent credibility scoring across the application
 */

export const CREDIBILITY_THRESHOLDS = {
  HIGH: 0.7,   // ≥70% = High credibility
  MEDIUM: 0.4, // 40-69% = Medium credibility
  LOW: 0,      // <40% = Low credibility
} as const

export type CredibilityBand = 'high' | 'medium' | 'low'

export function getCredibilityBand(score: number): CredibilityBand {
  if (score >= CREDIBILITY_THRESHOLDS.HIGH) return 'high'
  if (score >= CREDIBILITY_THRESHOLDS.MEDIUM) return 'medium'
  return 'low'
}

export function getCredibilityLabel(score?: number): string {
  if (typeof score !== 'number') return 'Unknown'
  const band = getCredibilityBand(score)
  switch (band) {
    case 'high': return 'Strong'
    case 'medium': return 'Moderate'
    case 'low': return 'Weak'
  }
}

export function getCredibilityColor(score: number): string {
  const band = getCredibilityBand(score)
  switch (band) {
    case 'high': return 'text-green-600 dark:text-green-400'
    case 'medium': return 'text-yellow-600 dark:text-yellow-400'
    case 'low': return 'text-red-600 dark:text-red-400'
  }
}

export function getCredibilityEmoji(score: number): string {
  const band = getCredibilityBand(score)
  const icons = {
    high: '🛡️', // Shield for high credibility
    medium: '⚠️', // Warning for medium
    low: '❌' // X for low
  }
  return icons[band]
}