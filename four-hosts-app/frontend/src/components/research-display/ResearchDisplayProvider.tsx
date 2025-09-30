import React from 'react'
import { ResearchDisplayContext } from './ResearchDisplayContext'
import type { ResearchDisplayContextValue } from './ResearchDisplayContext'

export const ResearchDisplayProvider: React.FC<{ value: ResearchDisplayContextValue; children: React.ReactNode }> = ({
  value,
  children,
}) => <ResearchDisplayContext.Provider value={value}>{children}</ResearchDisplayContext.Provider>
