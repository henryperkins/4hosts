import { memo } from 'react'
import type { ParadigmClassification } from '../types'
import { paradigmInfo, type Paradigm } from '../constants/paradigm'

interface ParadigmDisplayProps {
  classification: ParadigmClassification
}

const ParadigmDisplayComponent = ({ classification }: ParadigmDisplayProps) => {
  // Validate that we have a valid primary paradigm
  if (!classification?.primary || !paradigmInfo[classification.primary as Paradigm]) {
    return (
      <div className="mt-6 bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">Paradigm Classification</h2>
        <p className="text-gray-600 dark:text-gray-400">Analyzing paradigm...</p>
      </div>
    )
  }

  const primary = paradigmInfo[classification.primary as Paradigm]
  const secondary = classification.secondary ? paradigmInfo[classification.secondary as Paradigm] : null
  
  return (
    <div className="mt-6 bg-white dark:bg-gray-800 shadow rounded-lg p-6">
      <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">Paradigm Classification</h2>
      
      <div className="space-y-4">
        <div className={`border-l-4 ${primary.borderColor} ${primary.bgLight} p-4 rounded-r-lg`}>
          <div className="flex items-center justify-between">
            <div>
              <h3 className={`font-semibold ${primary.textColor}`}>{primary.name}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{primary.description}</p>
            </div>
            <div className="text-right">
              <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${primary.color} text-white`}>
                {(classification.confidence * 100).toFixed(0)}% confidence
              </span>
            </div>
          </div>
        </div>
        
        {secondary && (
          <div className={`border-l-4 ${secondary.borderColor} ${secondary.bgLight} p-4 rounded-r-lg opacity-75`}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className={`font-semibold ${secondary.textColor} text-sm`}>Secondary: {secondary.name}</h3>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{secondary.description}</p>
              </div>
            </div>
          </div>
        )}
        
        {classification.distribution && Object.keys(classification.distribution).length > 0 && (
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Distribution</h4>
            <div className="space-y-2">
              {Object.entries(classification.distribution).map(([paradigm, score]) => {
              const info = paradigmInfo[paradigm as Paradigm]
              if (!info) return null
              return (
                <div key={paradigm} className="flex items-center">
                  <span className="text-xs text-gray-600 dark:text-gray-400 w-20">{info.name.split(' ')[0]}</span>
                  <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2 ml-2">
                    <div 
                      className={`${info.color} h-2 rounded-full`} 
                      style={{ width: `${score * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400 ml-2 w-10 text-right">
                    {(score * 100).toFixed(0)}%
                  </span>
                </div>
              )
            })}
            </div>
          </div>
        )}

        {/* Keyword signals (if provided) */}
        {classification.signals && (
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Signals</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {Object.entries(classification.signals).map(([p, detail]) => {
                const info = paradigmInfo[p as Paradigm]
                if (!info || !detail) return null
                const kws = (detail.keywords || []).slice(0, 3)
                const intents = (detail.intent_signals || []).slice(0, 3)
                if (kws.length === 0 && intents.length === 0) return null
                return (
                  <div key={p} className={`border rounded-md p-2 ${info.borderColor} ${info.bgLight}`}>
                    <div className={`text-xs font-semibold mb-1 ${info.textColor}`}>{info.name.split(' ')[0]}</div>
                    {kws.length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-1">
                        {kws.map((kw, idx) => (
                          <span key={idx} className="px-2 py-0.5 text-[10px] rounded bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-100">
                            {kw}
                          </span>
                        ))}
                      </div>
                    )}
                    {intents.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {intents.map((sig, idx) => (
                          <span key={idx} className="px-2 py-0.5 text-[10px] rounded bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200">
                            {sig}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

const ParadigmDisplay = memo(ParadigmDisplayComponent)
export default ParadigmDisplay
