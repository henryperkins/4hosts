import { useContext } from 'react'
import { ResearchDisplayContext } from './ResearchDisplayContext'

export const useResearchDisplay = () => {
  const context = useContext(ResearchDisplayContext)
  if (!context) {
    throw new Error('useResearchDisplay must be used within a ResearchDisplayProvider')
  }
  return context
}
