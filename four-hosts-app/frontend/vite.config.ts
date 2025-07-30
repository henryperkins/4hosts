import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
  ],

  // Resolve configuration
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },

  // Server configuration
  server: {
    port: 5173,
    host: true,
    allowedHosts: ['lakefrontdigital.io', 'localhost', '.localhost'],
    watch: {
      // Exclude directories that contain many files
      ignored: [
        '**/node_modules/**',
        '**/dist/**',
        '**/.git/**',
        '**/coverage/**',
        '**/build/**'
      ]
    },
    proxy: {
      // Proxy auth routes directly
      '/auth': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy paradigm routes directly
      '/paradigms': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy research routes directly
      '/research': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy system routes directly
      '/system': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy sources routes directly
      '/sources': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy health route directly
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy metrics route directly
      '/metrics': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy admin routes directly
      '/admin': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy API routes that actually use /api/v1
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy WebSocket connections
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },

  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: true,
    // Rollup options for better code splitting
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'chart-vendor': ['recharts'],
          'ui-vendor': ['lucide-react', 'react-hot-toast', 'date-fns'],
        },
      },
    },
  },

  // Define global constants
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
  },

  // Optimize dependencies
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-router-dom', 'recharts', 'lucide-react'],
  },
})
