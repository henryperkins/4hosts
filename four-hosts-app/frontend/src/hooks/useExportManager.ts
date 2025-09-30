import { useCallback, useMemo, useState } from 'react'
import toast from 'react-hot-toast'
import api from '../services/api'

type ExportFormat = 'json' | 'csv' | 'pdf' | 'markdown' | 'excel'

const DEFAULT_FORMATS: ExportFormat[] = ['json', 'csv', 'pdf', 'markdown', 'excel']

export interface ExportManager {
  isExporting: boolean
  currentFormat: ExportFormat | null
  availableFormats: ExportFormat[]
  exportResult: (format: ExportFormat) => Promise<void>
}

export const useExportManager = (
  researchId: string,
  exportFormats?: Record<string, string>
): ExportManager => {
  const [isExporting, setIsExporting] = useState(false)
  const [currentFormat, setCurrentFormat] = useState<ExportFormat | null>(null)

  const availableFormats = useMemo<ExportFormat[]>(() => {
    if (exportFormats && Object.keys(exportFormats).length > 0) {
      return DEFAULT_FORMATS.filter(fmt => fmt in exportFormats)
    }
    return DEFAULT_FORMATS
  }, [exportFormats])

  const exportResult = useCallback(
    async (format: ExportFormat) => {
      setIsExporting(true)
      setCurrentFormat(format)
      try {
        const blob = await api.exportResearch(researchId, format)
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `research-${researchId}.${format}`
        link.click()
        URL.revokeObjectURL(url)
        toast.success(`Exported as ${format.toUpperCase()}`)
      } catch (error) {
        const detail = error instanceof Error ? error.message : 'Export failed'
        toast.error(detail)
      } finally {
        setIsExporting(false)
        setCurrentFormat(null)
      }
    },
    [researchId]
  )

  return {
    isExporting,
    currentFormat,
    availableFormats,
    exportResult,
  }
}
