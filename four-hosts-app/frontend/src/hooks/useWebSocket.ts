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
  
  const connect = useCallback(() => {
    if (!enabled || wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const apiUrl = new URL(import.meta.env.VITE_API_BASE_URL || window.location.origin)
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
        onMessage(message)
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
      
      // Attempt reconnection after 5 seconds if still enabled
      if (enabled) {
        reconnectTimeoutRef.current = setTimeout(connect, 5000)
      }
    }
  }, [researchId, onMessage, enabled])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
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