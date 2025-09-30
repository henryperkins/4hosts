import React from 'react'
import { ResearchDisplayContainer } from './research-display/ResearchDisplayContainer'
import type { ResultsDisplayEnhancedProps } from '../types/research-display'

export const ResultsDisplayEnhanced: React.FC<ResultsDisplayEnhancedProps> = ({ results }) => {
  return <ResearchDisplayContainer results={results} />
}

export default ResultsDisplayEnhanced
