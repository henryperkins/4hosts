import React, { useEffect, useState } from 'react'
import api from '../services/api'
import { Button } from './ui/Button'
import { InputField } from './ui/InputField'

interface Webhook {
  id: string
  url: string
  event: string
}

/**
 * Very small admin-only page that lets a user list, create and delete
 * webhooks. In real use we would guard with auth role checks and place
 * in a proper router path.
 */
const WebhooksPage: React.FC = () => {
  const [webhooks, setWebhooks] = useState<Webhook[]>([])
  const [url, setUrl] = useState('')
  const [event, setEvent] = useState('research_completed')
  const [loading, setLoading] = useState(false)

  const fetchWebhooks = async () => {
    setLoading(true)
    try {
      const data = await api.listWebhooks()
      setWebhooks(data)
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e)
      alert('Failed to load webhooks')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchWebhooks()
  }, [])

  const handleCreate = async () => {
    if (!url.trim()) return
    setLoading(true)
    try {
      await api.createWebhook(url.trim(), event)
      setUrl('')
      await fetchWebhooks()
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e)
      alert((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this webhook?')) return
    setLoading(true)
    try {
      await api.deleteWebhook(id)
      await fetchWebhooks()
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e)
      alert('Failed to delete')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">Webhooks</h1>

      <div className="flex space-x-2 mb-4">
        <InputField
          placeholder="Webhook URL"
          value={url}
          onChange={e => setUrl(e.target.value)}
        />
        <InputField
          placeholder="Event (e.g. research_completed)"
          value={event}
          onChange={e => setEvent(e.target.value)}
        />
        <Button onClick={handleCreate} disabled={loading || !url.trim()}>
          Add
        </Button>
      </div>

      {loading && <p>Loading…</p>}

      <table className="w-full text-left border">
        <thead>
          <tr>
            <th className="border px-2 py-1">ID</th>
            <th className="border px-2 py-1">URL</th>
            <th className="border px-2 py-1">Event</th>
            <th className="border px-2 py-1">Actions</th>
          </tr>
        </thead>
        <tbody>
          {webhooks.map(w => (
            <tr key={w.id}>
              <td className="border px-2 py-1 text-xs">{w.id}</td>
              <td className="border px-2 py-1 break-all">{w.url}</td>
              <td className="border px-2 py-1">{w.event}</td>
              <td className="border px-2 py-1">
                <Button size="sm" variant="destructive" onClick={() => handleDelete(w.id)}>
                  Delete
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default WebhooksPage

