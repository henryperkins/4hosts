export type Paradigm = 'dolores' | 'teddy' | 'bernard' | 'maeve'

// Paradigm color classes
export const paradigmColors: Record<Paradigm | 'default', string> = {
  dolores: 'bg-paradigm-dolores text-white',
  teddy: 'bg-paradigm-teddy text-white',
  bernard: 'bg-paradigm-bernard text-white',
  maeve: 'bg-paradigm-maeve text-white',
  default: 'bg-surface-muted text-text'
}

export const paradigmDescriptions: Record<Paradigm, string> = {
  dolores: 'Truth & Justice',
  teddy: 'Care & Support',
  bernard: 'Analysis & Logic',
  maeve: 'Strategy & Power'
}

// Helper to get color values for charts
export const getParadigmColorValue = (paradigm: Paradigm): string => {
  const colors: Record<Paradigm, string> = {
    dolores: '#dc5353',
    teddy: '#e79455',
    bernard: '#4a7adf',
    maeve: '#3fb57f'
  }
  return colors[paradigm] || '#6B7280'
}

// Extended paradigm information
import { FiShield, FiHeart, FiCpu, FiTrendingUp } from 'react-icons/fi'

export const paradigmInfo = {
  dolores: {
    name: 'Dolores (Revolutionary)',
    shortName: 'Dolores',
    color: 'bg-paradigm-dolores',
    borderColor: 'border-paradigm-dolores',
    textColor: 'text-paradigm-dolores',
    bgLight: 'paradigm-bg-dolores',
    description: 'Exposing systemic injustices and power imbalances',
    shortDescription: 'Truth & Justice',
    icon: FiShield,
    focus: 'Revolutionary perspective focused on exposing hidden truths'
  },
  teddy: {
    name: 'Teddy (Devotion)',
    shortName: 'Teddy',
    color: 'bg-paradigm-teddy',
    borderColor: 'border-paradigm-teddy',
    textColor: 'text-paradigm-teddy',
    bgLight: 'paradigm-bg-teddy',
    description: 'Protecting and supporting vulnerable communities',
    shortDescription: 'Care & Support',
    icon: FiHeart,
    focus: 'Compassionate approach emphasizing community care'
  },
  bernard: {
    name: 'Bernard (Analytical)',
    shortName: 'Bernard',
    color: 'bg-paradigm-bernard',
    borderColor: 'border-paradigm-bernard',
    textColor: 'text-paradigm-bernard',
    bgLight: 'paradigm-bg-bernard',
    description: 'Providing objective analysis and empirical evidence',
    shortDescription: 'Analysis & Logic',
    icon: FiCpu,
    focus: 'Data-driven analysis with empirical foundations'
  },
  maeve: {
    name: 'Maeve (Strategic)',
    shortName: 'Maeve',
    color: 'bg-paradigm-maeve',
    borderColor: 'border-paradigm-maeve',
    textColor: 'text-paradigm-maeve',
    bgLight: 'paradigm-bg-maeve',
    description: 'Delivering actionable strategies and competitive advantage',
    shortDescription: 'Strategy & Power',
    icon: FiTrendingUp,
    focus: 'Strategic planning for sustainable power dynamics'
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
      dolores: 'paradigm-bg-dolores text-paradigm-dolores',
      teddy: 'paradigm-bg-teddy text-paradigm-teddy',
      bernard: 'paradigm-bg-bernard text-paradigm-bernard',
      maeve: 'paradigm-bg-maeve text-paradigm-maeve',
      default: 'bg-surface-muted text-text-muted'
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
