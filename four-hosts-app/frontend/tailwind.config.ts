import type { Config } from 'tailwindcss'

// Helper function to create color with opacity support
function withOpacityValue(variable: string) {
  return ({ opacityValue }: { opacityValue?: string }) => {
    if (opacityValue !== undefined) {
      return `rgb(var(${variable}) / ${opacityValue})`
    }
    return `rgb(var(${variable}))`
  }
}

const config: Config = {
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
    './index.html',
  ],
  darkMode: 'class',
  safelist: [
    // Stagger delay animations
    'stagger-delay-50',
    'stagger-delay-100',
    'stagger-delay-150',
    'stagger-delay-200',
    'stagger-delay-250',
    'stagger-delay-300',
    // Paradigm hover text colors
    'hover:text-paradigm-dolores',
    'hover:text-paradigm-teddy',
    'hover:text-paradigm-bernard',
    'hover:text-paradigm-maeve',
    // Paradigm backgrounds
    'paradigm-bg-dolores',
    'paradigm-bg-teddy',
    'paradigm-bg-bernard',
    'paradigm-bg-maeve',
    // Paradigm borders
    'paradigm-border-dolores',
    'paradigm-border-teddy',
    'paradigm-border-bernard',
    'paradigm-border-maeve',
    // Opacity variations for colors
    'bg-primary/5',
    'bg-primary/10',
    'bg-primary/20',
    'text-primary/80',
    'border-primary/30',
    // Animation classes
    'animate-shimmer',
    'delay-300',
    'delay-500',
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['var(--font-display)', 'sans-serif'],
        sans: ['var(--font-body)', 'sans-serif'],
      },
      colors: {
        // Paradigm colors with opacity support
        'paradigm-dolores': withOpacityValue('--paradigm-dolores-rgb'),
        'paradigm-teddy': withOpacityValue('--paradigm-teddy-rgb'),
        'paradigm-bernard': withOpacityValue('--paradigm-bernard-rgb'),
        'paradigm-maeve': withOpacityValue('--paradigm-maeve-rgb'),
        // Base colors with opacity support
        'surface': withOpacityValue('--surface-rgb'),
        'surface-subtle': withOpacityValue('--surface-subtle-rgb'),
        'surface-muted': withOpacityValue('--surface-muted-rgb'),
        'border': withOpacityValue('--border-rgb'),
        'border-subtle': withOpacityValue('--border-subtle-rgb'),
        'text': withOpacityValue('--text-rgb'),
        'text-subtle': withOpacityValue('--text-subtle-rgb'),
        'text-muted': withOpacityValue('--text-muted-rgb'),
        // Semantic colors with opacity support
        'primary': withOpacityValue('--primary-rgb'),
        'success': withOpacityValue('--success-rgb'),
        'error': withOpacityValue('--error-rgb'),
        'warning': withOpacityValue('--warning-rgb'),
      },
      keyframes: {
        spin: {
          from: { transform: 'rotate(0deg)' },
          to: { transform: 'rotate(360deg)' }
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' }
        },
        'slide-up': {
          from: {
            opacity: '0',
            transform: 'translateY(10px)'
          },
          to: {
            opacity: '1',
            transform: 'translateY(0)'
          }
        },
        shake: {
          '0%, 100%': { transform: 'translateX(0)' },
          '10%, 30%, 50%, 70%, 90%': { transform: 'translateX(-2px)' },
          '20%, 40%, 60%, 80%': { transform: 'translateX(2px)' }
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' }
        }
      },
      animation: {
        'spin': 'spin 1s linear infinite',
        'fade-in': 'fade-in 0.3s ease-out',
        'slide-up': 'slide-up 0.3s ease-out',
        'shake': 'shake 0.5s ease-in-out',
        'shimmer': 'shimmer 2s linear infinite',
      },
      typography: {
        DEFAULT: {
          css: {
            color: 'var(--text)',
            a: {
              color: 'var(--primary)',
              '&:hover': {
                color: 'var(--primary)',
                opacity: '0.8',
              },
            },
            strong: {
              color: 'var(--text)',
              fontFamily: 'var(--font-display)',
            },
            h1: {
              color: 'var(--text)',
              fontFamily: 'var(--font-display)',
            },
            h2: {
              color: 'var(--text)',
              fontFamily: 'var(--font-display)',
            },
            h3: {
              color: 'var(--text)',
              fontFamily: 'var(--font-display)',
            },
            h4: {
              color: 'var(--text)',
              fontFamily: 'var(--font-display)',
            },
            code: {
              color: 'var(--text)',
            },
            blockquote: {
              color: 'var(--text-subtle)',
              borderLeftColor: 'var(--border)',
            },
            'code::before': {
              content: '""',
            },
            'code::after': {
              content: '""',
            },
          },
        },
      },
    },
  },
  plugins: [
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require('@tailwindcss/typography'),
  ],
}

export default config
