# Four Hosts Research Frontend

A modern React TypeScript frontend for the Four Hosts paradigm-aware research system.

## Features

### Core Features
- **Paradigm-aware Research**: Submit queries analyzed through four distinct paradigms (Dolores, Teddy, Bernard, Maeve)
- **Real-time Updates**: WebSocket integration for live research progress
- **Authentication**: Secure user registration and login with JWT tokens
- **Research History**: Track and revisit all your previous research queries

### Enhanced Features
- **Export Functionality**: Export research results in JSON, Markdown, or PDF formats
- **Source Credibility**: Visual credibility scoring for all research sources
- **User Preferences**: Customize default paradigm, research depth, and feature toggles
- **Metrics Dashboard**: System-wide analytics and usage patterns
- **Dark Mode**: Theme customization support

## Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running on http://localhost:8000

## Installation

1. Clone the repository and navigate to frontend:
```bash
cd four-hosts-app/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file:
```bash
cp .env.example .env
```

4. Update `.env` with your configuration:
```env
VITE_API_URL=http://localhost:8000
```

## Development

Start the development server:
```bash
npm run dev
```

The app will be available at http://localhost:5173

## Building for Production

1. Build the application:
```bash
npm run build
```

2. Preview the production build:
```bash
npm run preview
```

## Project Structure

```
src/
├── components/          # React components
│   ├── auth/           # Authentication components
│   ├── ResearchFormEnhanced.tsx
│   ├── ResultsDisplayEnhanced.tsx
│   ├── ResearchProgress.tsx
│   ├── UserProfile.tsx
│   ├── ResearchHistory.tsx
│   └── MetricsDashboard.tsx
├── contexts/           # React contexts
│   └── AuthContext.tsx
├── services/           # API services
│   └── api.ts
├── types.ts            # TypeScript type definitions
├── App.tsx             # Main app component with routing
└── main.tsx           # Entry point
```

## Key Components

### Authentication
- **LoginForm**: User login with username/password
- **RegisterForm**: New user registration
- **ProtectedRoute**: Route wrapper for authenticated pages
- **AuthContext**: Global authentication state management

### Research Features
- **ResearchFormEnhanced**: Advanced research query submission with options
- **ResearchProgress**: Real-time WebSocket progress updates
- **ResultsDisplayEnhanced**: Rich results display with export and credibility
- **ParadigmDisplay**: Visual paradigm classification display

### User Features
- **UserProfile**: User settings and preferences management
- **ResearchHistory**: Browse and revisit past research queries
- **MetricsDashboard**: System analytics and usage visualization

## API Integration

The frontend integrates with all backend endpoints:

### Authentication
- POST `/auth/register` - User registration
- POST `/auth/login` - User login
- POST `/auth/logout` - User logout
- GET `/auth/me` - Current user info
- PUT `/auth/preferences` - Update preferences

### Research
- POST `/paradigms/classify` - Classify query paradigm
- POST `/research/query` - Submit research query
- GET `/research/status/{id}` - Check research status
- GET `/research/results/{id}` - Get research results
- GET `/research/history` - User research history

### Export & Analytics
- POST `/v1/export/research/{id}` - Export research results (PDF/JSON/CSV/Markdown/Excel)
- GET `/sources/credibility/{domain}` - Get source credibility
- GET `/system/stats` - System statistics
- GET `/metrics` - Prometheus metrics

### WebSocket
- WS `/ws/research` - Real-time research updates

## Customization

### Theme
Users can switch between light and dark themes in their profile settings.

### Research Options
- **Depth**: Quick, Standard, or Deep research
- **Secondary Paradigms**: Include secondary paradigm analysis
- **Real Search**: Enable live API searches
- **AI Classification**: Use advanced AI for paradigm detection

### Default Preferences
Users can set default values for:
- Preferred paradigm
- Research depth
- Feature toggles

## Performance Considerations

- **Code Splitting**: Routes are lazy-loaded for optimal performance
- **WebSocket Management**: Automatic connection management and cleanup
- **Caching**: API responses are cached where appropriate
- **Optimistic Updates**: UI updates before server confirmation

## Security

- Cookie-based auth: the backend issues httpOnly cookies for access/refresh tokens. The frontend never stores tokens in `localStorage`.
- CSRF protection: all state-changing requests include an `X-CSRF-Token` header fetched from `/api/csrf-token`.
- Credentials: all API requests use `credentials: 'include'` so cookies flow via the Vite dev proxy (`/v1`, `/auth`, `/api`, `/ws`).
- Automatic refresh: the frontend refreshes cookies via `/auth/refresh` when a `401` occurs.
- Protected routes require authentication.
- CORS/CSP: in production the app is served behind Nginx with a strict Content Security Policy. For cross-origin APIs, adjust `connect-src` or serve API at the same origin.

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure backend is running on configured port
   - Check CORS settings
   - Verify API URL in .env

2. **WebSocket Not Connecting**
   - Check if backend WebSocket is enabled
   - Verify authentication token
   - Check browser console for errors

3. **Build Errors**
   - Clear node_modules and reinstall
   - Check TypeScript errors with `npm run lint`
   - Ensure all dependencies are installed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is part of the Four Hosts Research System.
