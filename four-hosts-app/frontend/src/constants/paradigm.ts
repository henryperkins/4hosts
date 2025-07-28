export type Paradigm = 'dolores' | 'teddy' | 'bernard' | 'maeve'

export const paradigmColors: Record<Paradigm | 'default', string> = {
  dolores: 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-200 dark:border-red-800',
  teddy: 'bg-orange-100 text-orange-800 border-orange-200 dark:bg-orange-900/20 dark:text-orange-200 dark:border-orange-800',
  bernard: 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/20 dark:text-blue-200 dark:border-blue-800',
  maeve: 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-200 dark:border-green-800',
  default: 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-800 dark:text-gray-200 dark:border-gray-700'
}

export const paradigmDescriptions: Record<Paradigm, string> = {
  dolores: 'Truth & Justice',
  teddy: 'Care & Support',
  bernard: 'Analysis & Logic',
  maeve: 'Strategy & Power'
}

// Hex colors for charts and visualizations (e.g., Recharts)
export const paradigmHexColors: Record<Paradigm, string> = {
  dolores: '#EF4444',
  teddy: '#F97316',
  bernard: '#3B82F6',
  maeve: '#10B981'
}

// Extended paradigm information used by ParadigmDisplay and ResearchHistory
export const paradigmInfo = {
  dolores: {
    name: 'Dolores (Revolutionary)',
    shortName: 'Dolores',
    color: 'bg-red-500',
    borderColor: 'border-red-500',
    textColor: 'text-red-700',
    bgLight: 'bg-red-50',
    description: 'Exposing systemic injustices and power imbalances',
    shortDescription: 'Truth & Justice',
    icon: '⚔️',
    focus: 'Revolutionary perspective focused on exposing hidden truths'
  },
  teddy: {
    name: 'Teddy (Devotion)',
    shortName: 'Teddy',
    color: 'bg-orange-500',
    borderColor: 'border-orange-500',
    textColor: 'text-orange-700',
    bgLight: 'bg-orange-50',
    description: 'Protecting and supporting vulnerable communities',
    shortDescription: 'Care & Support',
    icon: '🛡️',
    focus: 'Compassionate approach emphasizing community care'
  },
  bernard: {
    name: 'Bernard (Analytical)',
    shortName: 'Bernard',
    color: 'bg-blue-500',
    borderColor: 'border-blue-500',
    textColor: 'text-blue-700',
    bgLight: 'bg-blue-50',
    description: 'Providing objective analysis and empirical evidence',
    shortDescription: 'Analysis & Logic',
    icon: '🔬',
    focus: 'Data-driven analysis with empirical foundations'
  },
  maeve: {
    name: 'Maeve (Strategic)',
    shortName: 'Maeve',
    color: 'bg-green-500',
    borderColor: 'border-green-500',
    textColor: 'text-green-700',
    bgLight: 'bg-green-50',
    description: 'Delivering actionable strategies and competitive advantage',
    shortDescription: 'Strategy & Power',
    icon: '♟️',
    focus: 'Strategic planning for sustainable power dynamics'
  }
} as const

// Type guard for paradigm validation
export function isValidParadigm(value: string): value is Paradigm {
  return value === 'dolores' || value === 'teddy' || value === 'bernard' || value === 'maeve'
}

// Helper function to get paradigm class
export function getParadigmClass(paradigm: string): string {
  return paradigmColors[paradigm as Paradigm] || paradigmColors.default
}

// Helper function to get paradigm description
export function getParadigmDescription(paradigm: string): string {
  return paradigmDescriptions[paradigm as Paradigm] || paradigm
}

// Helper function to get paradigm hex color
export function getParadigmHexColor(paradigm: string): string {
  return paradigmHexColors[paradigm as Paradigm] || '#6B7280'
}

// Hover colors for navigation and interactive elements
export const paradigmHoverColors: Record<Paradigm, string> = {
  dolores: 'hover:text-red-600 dark:hover:text-red-400',
  teddy: 'hover:text-orange-600 dark:hover:text-orange-400',
  bernard: 'hover:text-blue-600 dark:hover:text-blue-400',
  maeve: 'hover:text-green-600 dark:hover:text-green-400'
}