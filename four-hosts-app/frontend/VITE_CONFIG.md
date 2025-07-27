# Vite Configuration Summary

## Overview
The frontend is configured with Vite 7+ and Tailwind CSS v4.1, providing a modern development experience with hot module replacement and optimized builds.

## Key Configuration Features

### 1. Tailwind CSS v4 Integration
- Uses `@tailwindcss/vite` plugin for zero-configuration setup
- No need for `postcss.config.js` or `tailwind.config.js`
- Automatic content detection for all template files
- Import with simple `@import "tailwindcss"` in CSS

### 2. Path Aliases
- `@/*` resolves to `./src/*` for cleaner imports
- Example: `import api from '@/services/api'`

### 3. Proxy Configuration
- `/api` proxies to `http://localhost:8000` (backend)
- `/ws` proxies WebSocket connections
- Automatic handling of CORS in development

### 4. Build Optimization
- Code splitting for vendor libraries:
  - `react-vendor`: React core libraries
  - `chart-vendor`: Recharts for data visualization
  - `ui-vendor`: UI libraries (lucide-react, react-hot-toast, date-fns)
- Source maps enabled for debugging

### 5. Environment Variables
- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)
- `VITE_ENABLE_METRICS`: Enable/disable metrics feature
- `VITE_ENABLE_EXPORT`: Enable/disable export feature
- `VITE_ENABLE_WEBSOCKET`: Enable/disable WebSocket support

### 6. TypeScript Configuration
- Strict mode enabled
- Path mapping configured
- Support for `.tsx` extensions
- Proper type definitions for environment variables

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

   Or use the convenience script:
   ```bash
   ./start.sh
   ```

## Production Build

1. **Build the application:**
   ```bash
   npm run build
   ```

2. **Preview production build:**
   ```bash
   npm run preview
   ```

## Custom Styling

The project uses Tailwind CSS v4 with custom theme extensions:

- **Paradigm Colors:**
  - Dolores: `#ef4444` (red)
  - Teddy: `#3b82f6` (blue)
  - Bernard: `#10b981` (green)
  - Maeve: `#8b5cf6` (purple)

- **Custom Animations:**
  - `spin-slow`: 2s rotation
  - `pulse-slow`: 3s pulse effect

## Troubleshooting

### Common Issues

1. **Tailwind classes not working:**
   - Ensure `@import "tailwindcss"` is in `index.css`
   - Check that `@tailwindcss/vite` is in dependencies

2. **Path aliases not resolving:**
   - Verify `tsconfig.app.json` has correct path mappings
   - Restart the dev server after config changes

3. **API proxy not working:**
   - Check that backend is running on port 8000
   - Verify proxy configuration in `vite.config.ts`

4. **WebSocket connection failing:**
   - Ensure backend WebSocket support is enabled
   - Check browser console for connection errors

## Performance Tips

1. Use dynamic imports for large components
2. Leverage Vite's HMR for faster development
3. Keep bundle sizes small with code splitting
4. Use production builds for deployment

## Dependencies

### Core Dependencies
- `vite`: ^7.0.4
- `@tailwindcss/vite`: ^4.1.11
- `tailwindcss`: ^4.1.11
- `@vitejs/plugin-react`: ^4.6.0

### Development Tools
- TypeScript for type safety
- ESLint for code quality
- Vite's built-in dev server with HMR