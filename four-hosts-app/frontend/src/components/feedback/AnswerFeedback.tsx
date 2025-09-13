import React, { useMemo, useState } from 'react'
import { FiSend, FiThumbsUp, FiThumbsDown, FiStar } from 'react-icons/fi'
import toast from 'react-hot-toast'
import { InputField } from '../ui/InputField'
import api from '../../services/api'

type Props = {
  researchId: string
}

export const AnswerFeedback: React.FC<Props> = ({ researchId }) => {
  const [rating, setRating] = useState<number>(0)
  const [hover, setHover] = useState<number>(0)
  const [helpful, setHelpful] = useState<boolean | null>(null)
  const [improvements, setImprovements] = useState('')
  const [reason, setReason] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const stars = useMemo(() => [1,2,3,4,5], [])

  const normalized = (rating: number) => Math.min(1, Math.max(0, (rating - 1) / 4))

  const handleSubmit = async () => {
    try {
      setSubmitting(true)
      const improvs = improvements
        .split(/\n|,|;/)
        .map(s => s.trim())
        .filter(Boolean)
      await api.submitAnswerFeedback({
        research_id: researchId,
        rating: normalized(rating),
        helpful: helpful === null ? undefined : helpful,
        improvements: improvs.length ? improvs : undefined,
        reason: reason || undefined,
      })
      toast.success('Thanks! Answer feedback recorded.')
      setRating(0)
      setHelpful(null)
      setImprovements('')
      setReason('')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to submit feedback'
      toast.error(message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-gray-700 dark:text-gray-300">Was this answer helpful?</span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className={`px-3 py-1 rounded border text-sm inline-flex items-center gap-1 ${helpful === true ? 'bg-green-600 text-white border-green-600' : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300'}`}
            onClick={() => setHelpful(true)}
            aria-pressed={helpful === true}
          >
            <FiThumbsUp className="h-4 w-4" /> Yes
          </button>
          <button
            type="button"
            className={`px-3 py-1 rounded border text-sm inline-flex items-center gap-1 ${helpful === false ? 'bg-red-600 text-white border-red-600' : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300'}`}
            onClick={() => setHelpful(false)}
            aria-pressed={helpful === false}
          >
            <FiThumbsDown className="h-4 w-4" /> No
          </button>
        </div>
      </div>

      <div className="mt-3">
        <label className="block text-sm text-gray-700 dark:text-gray-300 mb-1">Rate the answer</label>
        <div className="flex items-center gap-1">
          {stars.map((s) => (
            <button
              key={s}
              type="button"
              onMouseEnter={() => setHover(s)}
              onMouseLeave={() => setHover(0)}
              onClick={() => setRating(s)}
              className="p-1"
              aria-label={`${s} star${s > 1 ? 's' : ''}`}
            >
              <FiStar
                className={`h-6 w-6 ${ (hover || rating) >= s ? 'text-yellow-500' : 'text-gray-400 dark:text-gray-600' }`}
                aria-hidden="true"
              />
            </button>
          ))}
          <span className="ml-2 text-xs text-gray-600 dark:text-gray-400">{rating > 0 ? `${rating}/5 (${(normalized(rating) * 100).toFixed(0)}%)` : 'No rating yet'}</span>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
        <InputField
          textarea
          label="What could be improved?"
          hint="Separate suggestions by new lines, commas or semicolons"
          placeholder="e.g. Include more recent sources; Expand on limitations; Add step-by-step instructions"
          value={improvements}
          onChange={(e) => setImprovements(e.target.value)}
        />
        <InputField
          textarea
          label="Optional comments"
          placeholder="Share context or corrections"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
        />
      </div>

      <div className="mt-3">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={submitting || rating === 0}
          className={`inline-flex items-center gap-2 px-3 py-1.5 rounded text-sm border ${rating > 0 ? 'bg-blue-600 text-white border-blue-600' : 'border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300'} ${submitting ? 'opacity-60 cursor-not-allowed' : ''}`}
        >
          <FiSend className="h-4 w-4" /> Submit feedback
        </button>
      </div>
    </div>
  )
}

export default AnswerFeedback

