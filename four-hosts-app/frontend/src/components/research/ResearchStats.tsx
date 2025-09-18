import { memo } from 'react'

type ResearchStatsProps = {
  sourcesFound: string
  searches: string
  analyzed: string
  highQuality: string
  qualityRate: string
  duplicates: string
  toolsExecuted: string
  elapsed: string
}

export const ResearchStats = memo(function ResearchStats(props: ResearchStatsProps) {
  const {
    sourcesFound, searches, analyzed, highQuality, qualityRate, duplicates, toolsExecuted, elapsed,
  } = props
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-8 gap-3 mb-4">
      <div className="bg-surface-subtle rounded-lg p-3">
        <div className="text-2xl font-bold text-text">{sourcesFound}</div>
        <div className="text-xs text-text-muted">Sources Found</div>
      </div>
      <div className="bg-surface-subtle rounded-lg p-3">
        <div className="text-2xl font-bold text-text">{searches}</div>
        <div className="text-xs text-text-muted">Searches</div>
      </div>
      <div className="bg-surface-subtle rounded-lg p-3">
        <div className="text-2xl font-bold text-text">{analyzed}</div>
        <div className="text-xs text-text-muted">Analyzed</div>
      </div>
      <div className="bg-surface-subtle rounded-lg p-3">
        <div className="text-2xl font-bold text-success">{highQuality}</div>
        <div className="text-xs text-text-muted">High Quality</div>
      </div>
      <div className="bg-surface-subtle rounded-lg p-3">
        <div className="text-2xl font-bold text-text">{qualityRate}</div>
        <div className="text-xs text-text-muted">Quality Rate</div>
      </div>
      <div className="bg-surface-subtle rounded-lg p-3">
        <div className="text-2xl font-bold text-error">{duplicates}</div>
        <div className="text-xs text-text-muted">Duplicates Removed</div>
      </div>
      <div className="bg-surface-subtle rounded-lg p-3">
        <div className="text-2xl font-bold text-primary">{toolsExecuted}</div>
        <div className="text-xs text-text-muted">Tools Executed</div>
      </div>
      <div className="bg-surface-subtle rounded-lg p-3">
        <div className="text-2xl font-bold text-text">{elapsed}</div>
        <div className="text-xs text-text-muted">Elapsed</div>
      </div>
    </div>
  )
})

export default ResearchStats

