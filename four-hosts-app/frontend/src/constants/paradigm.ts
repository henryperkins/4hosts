export type Paradigm = 'dolores' | 'teddy' | 'bernard' | 'maeve'

// Use CSS variables for colors to maintain single source of truth
export const paradigmColors: Record<Paradigm | 'default', string> = {
  dolores: 'bg-[--color-paradigm-dolores] text-white',
  teddy: 'bg-[--color-paradigm-teddy] text-white',
  bernard: 'bg-[--color-paradigm-bernard] text-white',
  maeve: 'bg-[--color-paradigm-maeve] text-white',
  default: 'bg-surface-muted text-text'
}

export const paradigmDescriptions: Record<Paradigm, string> = {
  dolores: 'Truth & Justice',
  teddy: 'Care & Support',
  bernard: 'Analysis & Logic',
  maeve: 'Strategy & Power'
}

// Helper to get color values from CSS variables for charts
export const getParadigmColorValue = (paradigm: Paradigm): string => {
  if (typeof window === 'undefined') {
    // Fallback for SSR
    const fallbacks: Record<Paradigm, string> = {
      dolores: '#ef4444',
      teddy: '#f97316',
      bernard: '#3b82f6',
      maeve: '#10b981'
    }
    return fallbacks[paradigm]
  }
  
  const computedStyle = getComputedStyle(document.documentElement)
  const cssVar = `--color-paradigm-${paradigm}`
  const value = computedStyle.getPropertyValue(cssVar).trim()
  
  // Convert oklch to hex if needed (for chart libraries)
  if (value.startsWith('oklch')) {
    // For now, return fallback - in production, use a proper color conversion library
    const fallbacks: Record<Paradigm, string> = {
      dolores: '#ef4444',
      teddy: '#f97316',
      bernard: '#3b82f6',
      maeve: '#10b981'
    }
    return fallbacks[paradigm]
  }
  
  return value || '#6B7280'
}

// Extended paradigm information with CSS variable references
export const paradigmInfo = {
  dolores: {
    name: 'Dolores (Revolutionary)',
    shortName: 'Dolores',
    color: 'bg-[--color-paradigm-dolores]',
    borderColor: 'border-[--color-paradigm-dolores]',
    textColor: 'text-[--color-paradigm-dolores]',
    bgLight: 'paradigm-bg-dolores',
    description: 'Exposing systemic injustices and power imbalances',
    shortDescription: 'Truth & Justice',
    icon: '⚔️',
    focus: 'Revolutionary perspective focused on exposing hidden truths',
    cssVar: '--color-paradigm-dolores'
  },
  teddy: {
    name: 'Teddy (Devotion)',
    shortName: 'Teddy',
    color: 'bg-[--color-paradigm-teddy]',
    borderColor: 'border-[--color-paradigm-teddy]',
    textColor: 'text-[--color-paradigm-teddy]',
    bgLight: 'paradigm-bg-teddy',
    description: 'Protecting and supporting vulnerable communities',
    shortDescription: 'Care & Support',
    icon: '🛡️',
    focus: 'Compassionate approach emphasizing community care',
    cssVar: '--color-paradigm-teddy'
  },
  bernard: {
    name: 'Bernard (Analytical)',
    shortName: 'Bernard',
    color: 'bg-[--color-paradigm-bernard]',
    borderColor: 'border-[--color-paradigm-bernard]',
    textColor: 'text-[--color-paradigm-bernard]',
    bgLight: 'paradigm-bg-bernard',
    description: 'Providing objective analysis and empirical evidence',
    shortDescription: 'Analysis & Logic',
    icon: '🔬',
    focus: 'Data-driven analysis with empirical foundations',
    cssVar: '--color-paradigm-bernard'
  },
  maeve: {
    name: 'Maeve (Strategic)',
    shortName: 'Maeve',
    color: 'bg-[--color-paradigm-maeve]',
    borderColor: 'border-[--color-paradigm-maeve]',
    textColor: 'text-[--color-paradigm-maeve]',
    bgLight: 'paradigm-bg-maeve',
    description: 'Delivering actionable strategies and competitive advantage',
    shortDescription: 'Strategy & Power',
    icon: '♟️',
    focus: 'Strategic planning for sustainable power dynamics',
    cssVar: '--color-paradigm-maeve'
  }
} as const

// Type guard for paradigm validation
export function isValidParadigm(value: string): value is Paradigm {
  return value === 'dolores' || value === 'teddy' || value === 'bernard' || value === 'maeve'
}

// Helper function to get paradigm class
export function getParadigmClass(paradigm: string, variant: 'full' | 'subtle' = 'full'): string {
  if (variant === 'subtle') {
    const subtleClasses: Record<Paradigm | 'default', string> = {
      dolores: 'paradigm-bg-dolores border border-[--color-paradigm-dolores]/20 text-[--color-paradigm-dolores]',
      teddy: 'paradigm-bg-teddy border border-[--color-paradigm-teddy]/20 text-[--color-paradigm-teddy]',
      bernard: 'paradigm-bg-bernard border border-[--color-paradigm-bernard]/20 text-[--color-paradigm-bernard]',
      maeve: 'paradigm-bg-maeve border border-[--color-paradigm-maeve]/20 text-[--color-paradigm-maeve]',
      default: 'bg-surface-muted text-text-muted border border-border'
    }
    return subtleClasses[paradigm as Paradigm] || subtleClasses.default
  }
  return paradigmColors[paradigm as Paradigm] || paradigmColors.default
}

// Helper function to get paradigm description
export function getParadigmDescription(paradigm: string): string {
  return paradigmDescriptions[paradigm as Paradigm] || paradigm
}

// Helper function to get paradigm hex color (for charts)
export function getParadigmHexColor(paradigm: string): string {
  return getParadigmColorValue(paradigm as Paradigm)
}

// Helper to get button class for paradigm
export function getParadigmButtonClass(paradigm: Paradigm): string {
  const buttonClasses: Record<Paradigm, string> = {
    dolores: 'btn-paradigm-dolores',
    teddy: 'btn-paradigm-teddy',
    bernard: 'btn-paradigm-bernard',
    maeve: 'btn-paradigm-maeve'
  }
  return buttonClasses[paradigm] || 'btn-primary'
}