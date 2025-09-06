import React, { useState } from 'react'
import { Paradigm } from '../types'
import api from '../services/api'
import { Button } from './ui/Button'
import { Select } from './ui/Select'

interface Props {
  researchId: string
  currentParadigm: Paradigm
  onOverride?: (newParadigm: Paradigm) => void
}

/**
 * Lightweight selector that lets the user switch the paradigm
 * for an in-progress research task. This calls POST /paradigms/override
 * on success and optionally notifies parent via `onOverride`.
 */
const ParadigmOverride: React.FC<Props> = ({ researchId, currentParadigm, onOverride }) => {
  const [selected, setSelected] = useState<Paradigm>(currentParadigm)
  const [saving, setSaving] = useState(false)

  const paradigms: Paradigm[] = ['dolores', 'teddy', 'bernard', 'maeve']

  const handleSave = async () => {
    if (selected === currentParadigm) return
    setSaving(true)
    try {
      await api.overrideParadigm(researchId, selected)
      onOverride?.(selected)
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e)
      alert((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex items-center space-x-2">
      <Select
        value={selected}
        onChange={e => setSelected(e.target.value as Paradigm)}
        disabled={saving}
      >
        {paradigms.map(p => (
          <option key={p} value={p}>
            {p.charAt(0).toUpperCase() + p.slice(1)}
          </option>
        ))}
      </Select>
      <Button onClick={handleSave} disabled={saving || selected === currentParadigm}>
        Override
      </Button>
    </div>
  )
}

export default ParadigmOverride

