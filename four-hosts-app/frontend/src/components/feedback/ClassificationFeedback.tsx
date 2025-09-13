import React, { useMemo, useState } from 'react'
import { FiCheck, FiEdit2, FiSend } from 'react-icons/fi'
import toast from 'react-hot-toast'
import { InputField } from '../ui/InputField'
import api from '../../services/api'
import type { ParadigmClassification, Paradigm } from '../../types'

type Props = {
  researchId?: string | null
  query: string
  classification: ParadigmClassification
}

export const ClassificationFeedback: React.FC<Props> = ({ researchId, query, classification }) => {
  const [agree, setAgree] = useState<boolean | null>(null)
  const [correction, setCorrection] = useState<Paradigm | ''>('')
  const [rationale, setRationale] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const options = useMemo<Paradigm[]>(() => ['dolores','teddy','bernard','maeve'], [])

  const submit = async () => {
    try {
      setSubmitting(true)
      await api.submitClassificationFeedback({
        research_id: researchId || undefined,
        query,
        original: {
          primary: classification.primary,
          secondary: classification.secondary || undefined,
          distribution: classification.distribution || {},
          confidence: classification.confidence || 0,
        },
        user_correction: agree === false ? correction || undefined : undefined,
        rationale: rationale || undefined,
      })
      toast.success('Thanks! Classification feedback recorded.')
      setAgree(null)
      setCorrection('')
      setRationale('')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to submit feedback'
      toast.error(message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="mt-3 p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800">
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-sm text-gray-700 dark:text-gray-300">Was this classification correct?</span>
        <div className="flex gap-2">
          <button
            type="button"
            className={`px-3 py-1 rounded border text-sm ${agree === true ? 'bg-green-600 text-white border-green-600' : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300'}`}
            onClick={() => setAgree(true)}
            aria-pressed={agree === true}
          >
            Yes
          </button>
          <button
            type="button"
            className={`px-3 py-1 rounded border text-sm ${agree === false ? 'bg-red-600 text-white border-red-600' : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300'}`}
            onClick={() => setAgree(false)}
            aria-pressed={agree === false}
          >
            No
          </button>
        </div>
      </div>

      {agree === false && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-sm text-gray-700 dark:text-gray-300 mb-1">Choose the correct paradigm</label>
            <div className="flex flex-wrap gap-2">
              {options.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setCorrection(p)}
                  className={`px-3 py-1 rounded-full text-xs border ${correction === p ? 'bg-blue-600 text-white border-blue-600' : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300'}`}
                >
                  {p.charAt(0).toUpperCase() + p.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <div>
            <InputField
              textarea
              label="Optional rationale"
              placeholder="Briefly explain why a different paradigm fits better"
              value={rationale}
              onChange={(e) => setRationale(e.target.value)}
            />
          </div>
        </div>
      )}

      <div className="mt-3 flex items-center gap-2">
        <button
          type="button"
          onClick={submit}
          disabled={submitting || (agree === false && !correction)}
          className={`inline-flex items-center gap-2 px-3 py-1.5 rounded text-sm border ${submitting ? 'opacity-60 cursor-not-allowed' : ''} ${agree === true || (agree === false && correction) ? 'bg-blue-600 text-white border-blue-600' : 'border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300'}`}
        >
          {submitting ? <FiEdit2 className="h-4 w-4 animate-pulse" /> : <FiSend className="h-4 w-4" />} Submit
        </button>
        {agree === true && (
          <span className="text-xs text-gray-500 dark:text-gray-400 inline-flex items-center gap-1"><FiCheck /> You’re confirming the classification.</span>
        )}
      </div>
    </div>
  )
}

export default ClassificationFeedback

