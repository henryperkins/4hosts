/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  readonly VITE_ENABLE_METRICS?: string
  readonly VITE_ENABLE_EXPORT?: string
  readonly VITE_ENABLE_WEBSOCKET?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

declare const __APP_VERSION__: string