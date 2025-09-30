import React, { useState } from 'react'
import { FiChevronDown, FiChevronUp } from 'react-icons/fi'
import { useResearchDisplay } from './useResearchDisplay'

export const AnswerSections: React.FC = () => {
  const {
    data: { baseAnswer, results },
  } = useResearchDisplay()

  const sections = baseAnswer?.sections || results.answer?.sections || []
  const [expanded, setExpanded] = useState<Set<number>>(new Set([0]))

  const toggleSection = (index: number) => {
    setExpanded(prev => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  if (!sections || sections.length === 0) {
    return (
      <section className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-text mb-2">Answer sections</h3>
        <p className="text-sm text-text-muted">No detailed sections were generated.</p>
      </section>
    )
  }

  return (
    <section className="bg-surface rounded-lg border border-border p-4 sm:p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-text mb-4">Answer sections</h3>
      <div className="space-y-3">
        {sections.map((section, index) => {
          const isOpen = expanded.has(index)
          return (
            <div key={`${section.title}-${index}`} className="border border-border rounded-lg overflow-hidden">
              <button
                type="button"
                onClick={() => toggleSection(index)}
                className="w-full flex items-center justify-between px-4 py-3 bg-surface-subtle hover:bg-surface transition"
                aria-expanded={isOpen}
              >
                <div>
                  <p className="text-sm font-semibold text-text">{section.title}</p>
                  <p className="text-xs text-text-muted">
                    {section.key_insights && section.key_insights.length > 0
                      ? `Key insights: ${section.key_insights.slice(0, 2).join('; ')}`
                      : `${section.citations?.length || 0} citations`}
                  </p>
                </div>
                {isOpen ? <FiChevronUp className="h-4 w-4" /> : <FiChevronDown className="h-4 w-4" />}
              </button>
              {isOpen ? (
                <div className="px-4 py-4 text-sm leading-relaxed text-text whitespace-pre-line">
                  {section.content}
                  {section.key_insights && section.key_insights.length ? (
                    <ul className="mt-3 list-disc pl-5 text-text-muted text-xs space-y-1">
                      {section.key_insights.map((insight, idx) => (
                        <li key={idx}>{insight}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              ) : null}
            </div>
          )
        })}
      </div>
    </section>
  )
}
