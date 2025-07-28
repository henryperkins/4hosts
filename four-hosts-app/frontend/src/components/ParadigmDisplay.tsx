import type { ParadigmClassification } from '../types'
import { paradigmInfo, type Paradigm } from '../constants/paradigm'

interface ParadigmDisplayProps {
  classification: ParadigmClassification
}

function ParadigmDisplay({ classification }: ParadigmDisplayProps) {
  const primary = paradigmInfo[classification.primary as Paradigm]
  const secondary = classification.secondary ? paradigmInfo[classification.secondary as Paradigm] : null
  
  return (
    <div className="mt-6 bg-white shadow rounded-lg p-6">
      <h2 className="text-lg font-medium text-gray-900 mb-4">Paradigm Classification</h2>
      
      <div className="space-y-4">
        <div className={`border-l-4 ${primary.borderColor} ${primary.bgLight} p-4 rounded-r-lg`}>
          <div className="flex items-center justify-between">
            <div>
              <h3 className={`font-semibold ${primary.textColor}`}>{primary.name}</h3>
              <p className="text-sm text-gray-600 mt-1">{primary.description}</p>
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
                <p className="text-xs text-gray-600 mt-1">{secondary.description}</p>
              </div>
            </div>
          </div>
        )}
        
        <div className="mt-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Distribution</h4>
          <div className="space-y-2">
            {Object.entries(classification.distribution).map(([paradigm, score]) => {
              const info = paradigmInfo[paradigm as Paradigm]
              return (
                <div key={paradigm} className="flex items-center">
                  <span className="text-xs text-gray-600 w-20">{info.name.split(' ')[0]}</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-2 ml-2">
                    <div 
                      className={`${info.color} h-2 rounded-full`} 
                      style={{ width: `${score * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-600 ml-2 w-10 text-right">
                    {(score * 100).toFixed(0)}%
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ParadigmDisplay