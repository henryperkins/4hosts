import { useEffect, useRef, useCallback } from 'react'
import type { WSMessage } from '../types'

interface UseWebSocketOptions {
  researchId: string
  onMessage: (message: WSMessage) => void
  enabled?: boolean
}

export function useWebSocket({ researchId, onMessage, enabled = true }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const onMessageRef = useRef(onMessage)
  const mountedRef = useRef(true)

  // Keep onMessage ref up-to-date without triggering reconnections
  useEffect(() => {
    onMessageRef.current = onMessage
  }, [onMessage])

  // Track mounted state to prevent reconnection after unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false
    }
  }, [])

  const connect = useCallback(() => {
    if (!enabled || wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    // Mark as mounted/ready for reconnection when establishing a new connection
    mountedRef.current = true

    // Keep naming consistent with the rest of the app (services/api.ts)
    const apiUrl = new URL(import.meta.env.VITE_API_URL || window.location.origin)
    const wsProtocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${wsProtocol}//${apiUrl.host}/ws/research/${researchId}`

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log(`WebSocket connected for research ${researchId}`)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        onMessageRef.current(message)
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    ws.onerror = (error) => {
      console.error(`WebSocket error for research ${researchId}:`, error)
    }

    ws.onclose = () => {
      console.log(`WebSocket disconnected for research ${researchId}`)
      wsRef.current = null

      // Attempt reconnection after 5 seconds if still enabled and mounted
      if (enabled && mountedRef.current) {
        reconnectTimeoutRef.current = setTimeout(() => {
          if (mountedRef.current) {
            connect()
          }
        }, 5000)
      }
    }
  }, [researchId, enabled])

  const disconnect = useCallback(() => {
    // Clear any pending reconnection attempts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    // Close the websocket connection
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    // Note: mountedRef is only set to false on component unmount (see useEffect above)
    // This allows reconnection to work across multiple research runs
  }, [])

  useEffect(() => {
    if (enabled) {
      connect()
    }
    
    // Cleanup on unmount or when dependencies change
    return () => {
      disconnect()
    }
  }, [connect, disconnect, enabled])

  return { disconnect, reconnect: connect }
}
