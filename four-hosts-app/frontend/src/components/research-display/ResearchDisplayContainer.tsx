import React from 'react'
import type { ResearchResult } from '../../types'
import { useResearchData } from '../../hooks/useResearchData'
import { useFilterState } from '../../hooks/useFilterState'
import { useExportManager } from '../../hooks/useExportManager'
import { ResearchDisplayProvider } from './ResearchDisplayProvider'
import { ResearchHeader } from './ResearchHeader'
import { ResearchSummary } from './ResearchSummary'
import { ActionItemsList } from './ActionItemsList'
import { AnswerSections } from './AnswerSections'
import { ResearchSources } from './ResearchSources'
import { ResearchMetrics } from './ResearchMetrics'
import { EvidencePanel } from './EvidencePanel'
import { ContextMetricsPanel } from './ContextMetricsPanel'
import { AgentTrace } from './AgentTrace'
import { MeshAnalysis } from './MeshAnalysis'
import { AnswerCitations } from './AnswerCitations'

interface ResearchDisplayContainerProps {
  results: ResearchResult
}

export const ResearchDisplayContainer: React.FC<ResearchDisplayContainerProps> = ({ results }) => {
  const data = useResearchData(results)
  const filters = useFilterState(data.sources)
  const exportManager = useExportManager(results.research_id, results.export_formats)

  return (
    <ResearchDisplayProvider value={{ data, filters, exportManager }}>
      <div className="mt-8 space-y-6 animate-fade-in">
        <ResearchHeader />
        <ResearchSummary />
        <ActionItemsList />
        <AnswerSections />
        <ResearchSources />
        <ResearchMetrics />
        <EvidencePanel quotes={data.evidenceQuotes} />
        <ContextMetricsPanel />
        <AgentTrace />
        <MeshAnalysis />
        <AnswerCitations />
      </div>
    </ResearchDisplayProvider>
  )
}
