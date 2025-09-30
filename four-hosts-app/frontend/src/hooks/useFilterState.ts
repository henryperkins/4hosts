import { useEffect, useMemo, useState } from 'react'
import { getCredibilityBand } from '../utils/credibility'
import type { ResearchSource } from '../types/research-display'

type CredBand = 'high' | 'medium' | 'low'

export interface ResearchFilterState {
  categories: string[]
  selectedCategories: Set<string>
  toggleCategory: (category: string) => void
  selectedCredBands: Set<CredBand>
  toggleCredBand: (band: CredBand) => void
  resetFilters: () => void
  pageSize: number
  setPageSize: (size: number) => void
  page: number
  setPage: (page: number) => void
  filteredSources: ResearchSource[]
  paginatedSources: ResearchSource[]
  totalPages: number
}

const DEFAULT_BANDS: CredBand[] = ['high', 'medium', 'low']

export const useFilterState = (sources: ResearchSource[]): ResearchFilterState => {
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set(['all']))
  const [selectedCredBands, setSelectedCredBands] = useState<Set<CredBand>>(new Set(DEFAULT_BANDS))
  const [pageSize, setPageSize] = useState<number>(20)
  const [page, setPage] = useState<number>(1)

  const categories = useMemo(() => {
    const items = new Set<string>()
    for (const source of sources) {
      if (source.source_category) {
        items.add(source.source_category)
      }
    }
    return Array.from(items).sort()
  }, [sources])

  useEffect(() => {
    setPage(1)
  }, [selectedCategories, selectedCredBands, sources.length])

  const toggleCategory = (category: string) => {
    setSelectedCategories(prev => {
      const next = new Set(prev)
      if (category === 'all') {
        return new Set(['all'])
      }
      next.delete('all')
      if (next.has(category)) {
        next.delete(category)
        if (next.size === 0) {
          next.add('all')
        }
      } else {
        next.add(category)
      }
      return next
    })
  }

  const toggleCredBand = (band: CredBand) => {
    setSelectedCredBands(prev => {
      const next = new Set(prev)
      if (next.has(band)) {
        next.delete(band)
        if (next.size === 0) {
          DEFAULT_BANDS.forEach(defaultBand => next.add(defaultBand))
        }
      } else {
        next.add(band)
      }
      return next
    })
  }

  const resetFilters = () => {
    setSelectedCategories(new Set(['all']))
    setSelectedCredBands(new Set(DEFAULT_BANDS))
    setPage(1)
    setPageSize(20)
  }

  const filteredSources = useMemo(() => {
    return sources.filter(source => {
      const categoryMatches = selectedCategories.has('all') || (source.source_category && selectedCategories.has(source.source_category))
      const bandMatches = (() => {
        const score = typeof source.credibility_score === 'number' ? source.credibility_score : undefined
        if (typeof score !== 'number') return true
        const band = getCredibilityBand(score) as CredBand
        return selectedCredBands.has(band)
      })()
      return categoryMatches && bandMatches
    })
  }, [sources, selectedCategories, selectedCredBands])

  const totalPages = Math.max(1, Math.ceil(filteredSources.length / pageSize))

  const paginatedSources = useMemo(() => {
    const start = (page - 1) * pageSize
    return filteredSources.slice(start, start + pageSize)
  }, [filteredSources, page, pageSize])

  const handlePageSize = (size: number) => {
    setPageSize(size)
    setPage(1)
  }

  return {
    categories,
    selectedCategories,
    toggleCategory,
    selectedCredBands,
    toggleCredBand,
    resetFilters,
    pageSize,
    setPageSize: handlePageSize,
    page,
    setPage,
    filteredSources,
    paginatedSources,
    totalPages,
  }
}
