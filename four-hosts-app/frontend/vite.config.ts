import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
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
      // Proxy versioned API root so relative calls like /v1/auth/login work
      '/v1': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        ws: false,
      },
      // Proxy auth routes directly
      '/auth': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Proxy paradigm routes directly
      '/paradigms': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Proxy research routes directly
      '/research': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Proxy system routes directly
      '/system': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Proxy sources routes directly
      '/sources': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Proxy health route directly
      '/health': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Proxy metrics route directly
      '/metrics': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Proxy admin routes directly
      '/admin': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Proxy API routes that actually use /api/v1
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Proxy WebSocket connections
      '/ws': {
        target: 'ws://localhost:8001',
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
          'ui-vendor': ['react-icons', 'react-hot-toast', 'date-fns'],
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
    include: ['react', 'react-dom', 'react-router-dom', 'recharts', 'react-icons'],
  },
})
