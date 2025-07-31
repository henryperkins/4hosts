# Four Hosts Research Application - Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [API Documentation](#api-documentation)
6. [WebSocket Events](#websocket-events)
7. [Frontend Architecture](#frontend-architecture)
8. [Security Architecture](#security-architecture)
9. [Deployment Architecture](#deployment-architecture)
10. [Performance Considerations](#performance-considerations)

## System Overview

The Four Hosts Research Application is a paradigm-aware research platform that classifies queries into four distinct AI consciousness paradigms and executes specialized research strategies for each.

### The Four Paradigms
- **🔥 Dolores (Revolutionary)**: Investigative journalism, exposing systemic issues
- **💝 Teddy (Devotion)**: Compassionate support, community care
- **🔬 Bernard (Analytical)**: Data-driven research, empirical analysis
- **⚡ Maeve (Strategic)**: Business intelligence, actionable strategies

## Architecture Diagrams

### High-Level System Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                           Frontend (React + TypeScript)              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────┐│
│  │   Auth UI   │  │Research Form │  │  Results   │  │  Profile   ││
│  └─────────────┘  └──────────────┘  └────────────┘  └────────────┘│
└─────────────────────────────┬───────────────────────────────────────┘
                              │ HTTPS/WebSocket
┌─────────────────────────────┴───────────────────────────────────────┐
│                         API Gateway (FastAPI)                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────┐│
│  │Auth Handler │  │Research API  │  │WebSocket   │  │Admin API   ││
│  └─────────────┘  └──────────────┘  └────────────┘  └────────────┘│
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                         Core Services Layer                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────┐│
│  │Classification│  │Context Eng.  │  │Research    │  │Answer Gen  ││
│  │   Engine    │  │  Pipeline    │  │Orchestrator│  │  Service   ││
│  └─────────────┘  └──────────────┘  └────────────┘  └────────────┘│
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                      External Services Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────┐│
│  │  Azure/     │  │Google Search │  │Brave Search│  │ArXiv/PubMed││
│  │  OpenAI     │  │     API      │  │    API     │  │    APIs    ││
│  └─────────────┘  └──────────────┘  └────────────┘  └────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                         Data Layer                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐                │
│  │ PostgreSQL  │  │    Redis     │  │   File     │                │
│  │  Database   │  │   (Future)   │  │  Storage   │                │
│  └─────────────┘  └──────────────┘  └────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### Research Pipeline Flow
```
User Query
    │
    ▼
┌─────────────────┐
│ Classification  │──────► Paradigm Assignment
│     Engine      │        (Primary + Secondary)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Context Engineer │──────► W-S-C-I Pipeline
│   (W-S-C-I)    │        Paradigm-specific processing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Research     │──────► Multi-API Search
│  Orchestrator   │        Deduplication & Filtering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Answer Generator │──────► LLM Synthesis
│   (Paradigm)    │        Paradigm-aligned tone
└────────┬────────┘
         │
         ▼
    Final Result
```

### WebSocket Real-time Updates
```
┌─────────────┐     WebSocket      ┌─────────────┐
│   Client    │◄──────────────────►│   Server    │
└─────────────┘                    └─────────────┘
      │                                   │
      │  1. Connect (JWT)                │
      ├──────────────────────────────────►│
      │                                   │
      │  2. Subscribe to research        │
      ├──────────────────────────────────►│
      │                                   │
      │  3. Progress Updates             │
      │◄──────────────────────────────────┤
      │     {phase, progress, sources}   │
      │                                   │
      │  4. Cancel Request (optional)    │
      ├──────────────────────────────────►│
      │                                   │
      │  5. Final Result                 │
      │◄──────────────────────────────────┤
```

## Component Architecture

### Backend Components

#### 1. Main Application (`main.py`)
- **Responsibility**: API gateway, request routing, middleware management
- **Key Features**:
  - FastAPI with async support
  - JWT authentication middleware
  - CORS configuration
  - Error handling and monitoring
  - Graceful shutdown

#### 2. Classification Engine (`classification_engine.py`)
- **Responsibility**: Paradigm classification using hybrid approach
- **Methods**:
  - Rule-based classification (keywords, patterns)
  - LLM-based classification (GPT-4)
  - Confidence scoring
  - Secondary paradigm detection

#### 3. Context Engineering Pipeline (`context_engineering.py`)
- **W-S-C-I Layers**:
  - **Write**: Document paradigm focus
  - **Select**: Choose search methods
  - **Compress**: Optimize information density
  - **Isolate**: Extract key findings

#### 4. Research Orchestrator (`research_orchestrator.py`)
- **Responsibilities**:
  - Coordinate multi-API searches
  - Manage search budgets
  - Deduplicate results
  - Track progress via WebSocket
  - Handle cancellations

#### 5. Answer Generators (`answer_generator_*.py`)
- **Paradigm-specific generators**:
  - `answer_generator_dolores.py`: Revolutionary tone
  - `answer_generator_teddy.py`: Compassionate tone
  - `answer_generator_bernard.py`: Academic tone
  - `answer_generator_maeve.py`: Strategic tone

### Frontend Components

#### Component Hierarchy
```
App.tsx
├── AuthContext
├── ThemeContext
├── Router
│   ├── PublicRoute
│   │   ├── LandingPage
│   │   ├── Login
│   │   └── Register
│   └── ProtectedRoute
│       ├── Dashboard
│       │   ├── ResearchInterface
│       │   │   ├── ResearchFormEnhanced
│       │   │   ├── ResearchProgress
│       │   │   └── ResultsDisplay
│       │   └── Sidebar
│       ├── UserProfile
│       ├── ResearchHistory
│       └── Admin
│           ├── UserManagement
│           ├── SystemMonitoring
│           └── APIUsageStats
└── ErrorBoundary
```

## Data Flow

### 1. Authentication Flow
```
1. User Login
   → POST /api/auth/login
   → Validate credentials
   → Generate JWT tokens
   → Return access + refresh tokens

2. Token Refresh
   → POST /api/auth/refresh
   → Validate refresh token
   → Rotate refresh token
   → Return new token pair

3. Protected Request
   → Add Authorization header
   → Validate JWT
   → Extract user context
   → Process request
```

### 2. Research Query Flow
```
1. Submit Query
   → POST /api/research/query
   → Validate user permissions
   → Create research record

2. Classification
   → Analyze query features
   → Apply rule-based classification
   → LLM classification if needed
   → Return paradigm assignment

3. Context Engineering
   → Apply W-S-C-I pipeline
   → Generate search strategies
   → Optimize for paradigm

4. Search Execution
   → Execute parallel searches
   → Track API usage
   → Deduplicate results
   → Filter by relevance

5. Answer Generation
   → Select paradigm generator
   → Synthesize results
   → Apply paradigm tone
   → Return formatted answer
```

## API Documentation

### Authentication Endpoints

#### POST /api/auth/register
```json
Request:
{
  "username": "string",
  "email": "string",
  "password": "string",
  "full_name": "string"
}

Response:
{
  "message": "User registered successfully",
  "user_id": "uuid"
}
```

#### POST /api/auth/login
```json
Request:
{
  "username": "string", // or email
  "password": "string"
}

Response:
{
  "access_token": "jwt_token",
  "refresh_token": "jwt_token",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "string",
    "role": "string"
  }
}
```

### Research Endpoints

#### POST /api/research/query
```json
Request:
{
  "query": "string",
  "paradigm_override": "dolores|teddy|bernard|maeve", // optional
  "search_depth": "basic|deep", // optional
  "include_sources": true
}

Response:
{
  "query_id": "uuid",
  "status": "processing",
  "paradigm": {
    "primary": "string",
    "secondary": "string",
    "confidence": 0.95
  }
}
```

#### GET /api/research/query/{query_id}
```json
Response:
{
  "id": "uuid",
  "query": "string",
  "status": "completed",
  "paradigm": {...},
  "answer": "string",
  "sources": [...],
  "metadata": {...},
  "created_at": "datetime",
  "completed_at": "datetime"
}
```

### Admin Endpoints

#### GET /api/admin/users
- Requires ADMIN role
- Returns paginated user list

#### GET /api/admin/system/stats
```json
Response:
{
  "total_users": 1234,
  "active_queries": 45,
  "api_usage": {
    "google": {"used": 89, "limit": 100},
    "brave": {"used": 1234, "limit": 2000}
  },
  "system_health": "healthy"
}
```

## WebSocket Events

### Connection
```javascript
// Client connects with JWT
ws = new WebSocket('ws://localhost:8000/ws?token=JWT_TOKEN');
```

### Server Events

#### research_started
```json
{
  "type": "research_started",
  "query_id": "uuid",
  "paradigm": "dolores",
  "timestamp": "2024-01-31T12:00:00Z"
}
```

#### progress_update
```json
{
  "type": "progress_update",
  "query_id": "uuid",
  "phase": "searching|processing|generating",
  "progress": 0.75,
  "message": "Searching academic databases...",
  "sources_found": 12
}
```

#### source_discovered
```json
{
  "type": "source_discovered",
  "query_id": "uuid",
  "source": {
    "title": "string",
    "url": "string",
    "relevance_score": 0.92
  }
}
```

#### research_completed
```json
{
  "type": "research_completed",
  "query_id": "uuid",
  "status": "success",
  "answer_preview": "string"
}
```

### Client Events

#### cancel_research
```json
{
  "type": "cancel_research",
  "query_id": "uuid"
}
```

## Frontend Architecture

### State Management
- **Context API** for global state
  - AuthContext: User authentication state
  - ThemeContext: UI theme preferences
  - ResearchContext: Active research queries

### Component Design Patterns

#### 1. Container/Presenter Pattern
```typescript
// Container Component
const ResearchContainer: React.FC = () => {
  const [data, setData] = useState();
  const { user } = useAuth();
  
  useEffect(() => {
    fetchResearchData();
  }, []);
  
  return <ResearchPresenter data={data} />;
};

// Presenter Component
const ResearchPresenter: React.FC<Props> = ({ data }) => {
  return <div>{/* Pure rendering logic */}</div>;
};
```

#### 2. Custom Hooks
```typescript
// useWebSocket hook for real-time updates
const useWebSocket = (queryId: string) => {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    // WebSocket logic
    return () => ws.close();
  }, [queryId]);
  
  return { progress, status };
};
```

### Routing Structure
```typescript
<Routes>
  <Route path="/" element={<LandingPage />} />
  <Route path="/login" element={<Login />} />
  <Route path="/register" element={<Register />} />
  
  <Route element={<ProtectedRoute />}>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/research" element={<ResearchInterface />} />
    <Route path="/profile" element={<UserProfile />} />
    <Route path="/history" element={<ResearchHistory />} />
    
    <Route element={<AdminRoute />}>
      <Route path="/admin/*" element={<AdminPanel />} />
    </Route>
  </Route>
</Routes>
```

## Security Architecture

### Authentication & Authorization
1. **JWT-based authentication**
   - Access tokens (15 min expiry)
   - Refresh tokens (7 days expiry)
   - Token rotation on refresh

2. **Role-based access control**
   - Roles: FREE, BASIC, PRO, ENTERPRISE, ADMIN
   - Permission matrix for endpoints
   - Resource-level permissions

### Security Measures
1. **Input validation**: Pydantic models
2. **SQL injection prevention**: SQLAlchemy ORM
3. **XSS protection**: React auto-escaping
4. **CSRF protection**: SameSite cookies
5. **Rate limiting**: Per-user quotas
6. **API key rotation**: Scheduled rotation

## Deployment Architecture

### Current Setup (Development)
```
Single Server Deployment
├── Frontend (Vite dev server - port 5173)
├── Backend (Uvicorn - port 8000)
└── PostgreSQL (port 5432)
```

### Production Architecture
```
Load Balancer (NGINX/CloudFlare)
    │
    ├── Frontend CDN
    │   └── Static React App
    │
    ├── API Servers (Multiple instances)
    │   ├── Server 1 (with Redis)
    │   ├── Server 2 (with Redis)
    │   └── Server N (with Redis)
    │
    ├── PostgreSQL Cluster
    │   ├── Primary (Writes)
    │   └── Replicas (Reads)
    │
    └── Redis Cluster
        ├── Session Storage
        ├── Cache Layer
        └── Pub/Sub for WebSocket
```

### Scaling Considerations
1. **Horizontal scaling**: Add API server instances
2. **Database scaling**: Read replicas for queries
3. **Caching strategy**: Redis for hot data
4. **CDN integration**: Static asset delivery
5. **Queue system**: Background job processing

## Performance Considerations

### Current Optimizations
1. **Async I/O**: All database and API calls
2. **Connection pooling**: PostgreSQL connections
3. **Parallel searches**: Multi-API execution
4. **Result caching**: Paradigm-aware cache keys
5. **Early filtering**: Reduce processing load

### Bottlenecks & Solutions
1. **LLM API calls**
   - Solution: Implement response caching
   - Solution: Use smaller models for classification

2. **Database queries**
   - Solution: Add indexes on search columns
   - Solution: Implement query result caching

3. **WebSocket connections**
   - Solution: Implement connection pooling
   - Solution: Use Redis pub/sub for scaling

4. **Search API limits**
   - Solution: Implement fallback providers
   - Solution: Cache search results

### Monitoring & Observability
1. **Application metrics**
   - Request latency
   - Error rates
   - API usage

2. **Infrastructure metrics**
   - CPU/Memory usage
   - Database performance
   - Network latency

3. **Business metrics**
   - User engagement
   - Search quality scores
   - Paradigm classification accuracy