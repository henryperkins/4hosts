import { createContext } from 'react'
import type { ResearchDisplayData } from '../../hooks/useResearchData'
import type { ResearchFilterState } from '../../hooks/useFilterState'
import type { ExportManager } from '../../hooks/useExportManager'

export interface ResearchDisplayContextValue {
  data: ResearchDisplayData
  filters: ResearchFilterState
  exportManager: ExportManager
}

export const ResearchDisplayContext = createContext<ResearchDisplayContextValue | null>(null)
