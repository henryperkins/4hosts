# Four Hosts Application - System Orchestration Guide

## Executive Summary

The Four Hosts Research Application orchestrates a complex multi-stage pipeline that transforms user queries into paradigm-aligned research results. This document provides a comprehensive guide to understanding and managing the system's orchestration.

## Core Orchestration Flow

### 1. Query Lifecycle

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│Query Input  │────►│Classification│────►│Context Engine │────►│Search Engine │
└─────────────┘     └──────────────┘     └───────────────┘     └──────┬───────┘
                                                                         │
┌─────────────┐     ┌──────────────┐     ┌───────────────┐            │
│Final Result │◄────│Answer Gen    │◄────│Result Process │◄───────────┘
└─────────────┘     └──────────────┘     └───────────────┘
```

### 2. Paradigm Classification Orchestration

The classification engine uses a two-tier approach:

#### Tier 1: Rule-Based Classification
```python
# Quick classification based on keywords and patterns
if contains_investigative_terms(query):
    return Paradigm.DOLORES
elif contains_care_terms(query):
    return Paradigm.TEDDY
elif contains_analytical_terms(query):
    return Paradigm.BERNARD
elif contains_business_terms(query):
    return Paradigm.MAEVE
```

#### Tier 2: LLM-Based Classification
```python
# When rule-based is uncertain (confidence < 0.7)
response = await llm_client.classify_paradigm(
    query=query,
    context=extracted_features
)
```

### 3. Context Engineering Pipeline (W-S-C-I)

Each layer transforms the query for optimal research:

#### Write Layer
- Documents the paradigm's narrative focus
- Example (Dolores): "Expose systemic issues in..."

#### Select Layer
- Chooses appropriate search methods
- Example (Bernard): Prioritize academic databases

#### Compress Layer
- Optimizes information density
- Different compression ratios per paradigm

#### Isolate Layer
- Extracts paradigm-specific insights
- Example (Maeve): Focus on actionable strategies

### 4. Search Orchestration

#### Parallel Search Execution
```python
async def execute_searches():
    tasks = []
    
    # Create search tasks for each API
    if google_enabled:
        tasks.append(search_google(query))
    if brave_enabled:
        tasks.append(search_brave(query))
    if paradigm == Paradigm.BERNARD:
        tasks.append(search_arxiv(query))
        tasks.append(search_pubmed(query))
    
    # Execute in parallel
    results = await asyncio.gather(*tasks)
    return merge_and_deduplicate(results)
```

#### Search Priority Matrix
| Paradigm | Primary Sources | Secondary Sources | Avoided Sources |
|----------|----------------|-------------------|-----------------|
| Dolores | Investigative media, WikiLeaks | Alternative news | Corporate PR |
| Teddy | Community forums, Support groups | Medical sites | Political sources |
| Bernard | Academic journals, ArXiv | Government data | Opinion pieces |
| Maeve | Business journals, Case studies | Market research | Academic theory |

### 5. Real-time Progress Tracking

#### WebSocket Event Flow
```
Client                    Server                    Research Pipeline
  │                         │                              │
  ├─Connect(JWT)───────────►│                              │
  │                         ├─Start Research──────────────►│
  │◄─research_started───────┤                              │
  │                         │◄─Classification Complete─────┤
  │◄─progress_update(20%)───┤                              │
  │                         │◄─Context Engineering Done────┤
  │◄─progress_update(40%)───┤                              │
  │                         │◄─Search Results──────────────┤
  │◄─source_discovered──────┤                              │
  │◄─progress_update(70%)───┤                              │
  │                         │◄─Answer Generated────────────┤
  │◄─research_completed─────┤                              │
```

### 6. Answer Generation Orchestration

Each paradigm has a specialized answer generator:

#### Paradigm-Specific Tone and Structure
- **Dolores**: Provocative opening → Evidence → Call to action
- **Teddy**: Empathetic acknowledgment → Support → Resources
- **Bernard**: Hypothesis → Data analysis → Conclusions
- **Maeve**: Executive summary → Strategy → Implementation

## System Components Integration

### Backend Service Dependencies
```
main.py (FastAPI Server)
    ├── auth.py (Authentication)
    ├── classification_engine.py
    │   └── llm_client.py
    ├── context_engineering.py
    ├── research_orchestrator.py
    │   ├── paradigm_search.py
    │   ├── search_apis.py
    │   │   ├── google_search()
    │   │   ├── brave_search()
    │   │   ├── arxiv_search()
    │   │   └── pubmed_search()
    │   └── websocket_service.py
    └── answer_generator_*.py
        └── llm_client.py
```

### Database Schema Relationships
```
Users
  │
  ├─1:N─► ResearchQueries
  │         ├── paradigm_classification
  │         ├── search_results
  │         └── generated_answer
  │
  ├─1:N─► RefreshTokens
  │
  └─1:N─► UserPreferences
           └── default_paradigm
```

## Operational Orchestration

### 1. Startup Sequence
```python
async def lifespan(app: FastAPI):
    # Startup
    await init_database()
    await verify_api_keys()
    await warm_up_llm_connections()
    await start_monitoring()
    
    yield  # Application runs
    
    # Shutdown
    await close_websocket_connections()
    await flush_pending_writes()
    await close_database()
```

### 2. Request Processing Pipeline

#### Authentication Layer
1. Extract JWT from header/cookie/query
2. Validate token signature and expiry
3. Check user permissions for endpoint
4. Inject user context into request

#### Research Processing
1. Validate query parameters
2. Check user quotas
3. Create research record
4. Queue for processing
5. Return query_id for tracking

#### Background Processing
1. Classification (1-3 seconds)
2. Context engineering (1-2 seconds)
3. Search execution (5-15 seconds)
4. Answer generation (10-20 seconds)
5. Result storage and notification

### 3. Error Handling Orchestration

#### Cascading Fallbacks
```
Primary Service Fails
    │
    ├─► Try Secondary Service
    │     │
    │     └─► Still Fails?
    │           │
    │           ├─► Use Cached Results
    │           │     │
    │           │     └─► No Cache?
    │           │           │
    │           │           └─► Return Graceful Error
    │           │
    │           └─► Log and Monitor
    │
    └─► Continue with Degraded Service
```

### 4. Resource Management

#### API Quota Management
```python
class QuotaManager:
    def __init__(self):
        self.quotas = {
            'google': {'daily': 100, 'used': 0},
            'brave': {'monthly': 2000, 'used': 0},
            'arxiv': {'per_second': 3},
            'pubmed': {'per_second': 3}
        }
    
    async def check_and_consume(self, service: str) -> bool:
        if self.has_quota(service):
            await self.consume_quota(service)
            return True
        return False
```

#### Database Connection Pooling
- Pool size: 20 connections
- Max overflow: 10 connections
- Timeout: 30 seconds
- Recycle time: 3600 seconds

## Monitoring and Observability

### 1. Key Metrics to Track

#### System Health
- API response times (p50, p90, p99)
- Error rates by endpoint
- Database query performance
- WebSocket connection count

#### Business Metrics
- Queries per paradigm
- Search API usage rates
- Answer generation quality scores
- User engagement metrics

### 2. Logging Strategy

#### Log Levels
- **ERROR**: System failures, API errors
- **WARNING**: Quota warnings, degraded service
- **INFO**: Request processing, paradigm classification
- **DEBUG**: Detailed search results, LLM prompts

#### Structured Logging
```json
{
  "timestamp": "2024-01-31T12:00:00Z",
  "level": "INFO",
  "service": "research_orchestrator",
  "user_id": "uuid",
  "query_id": "uuid",
  "paradigm": "dolores",
  "phase": "search_execution",
  "duration_ms": 5234,
  "api_calls": {
    "google": 1,
    "brave": 1
  }
}
```

## Deployment Orchestration

### 1. Development Environment
```bash
# Start all services
./scripts/dev-start.sh

# Runs:
# - PostgreSQL (Docker)
# - Backend (Uvicorn with reload)
# - Frontend (Vite dev server)
# - Redis (Docker) - if available
```

### 2. Production Deployment

#### Blue-Green Deployment Strategy
1. Deploy new version to staging
2. Run smoke tests
3. Switch load balancer to new version
4. Monitor for errors
5. Keep old version for quick rollback

#### Database Migrations
```bash
# Pre-deployment
alembic upgrade head

# Rollback if needed
alembic downgrade -1
```

### 3. Scaling Orchestration

#### Horizontal Scaling Requirements
1. **Enable Redis** for distributed state
2. **Configure load balancer** for API servers
3. **Set up read replicas** for database
4. **Implement queue system** for background jobs

#### Auto-scaling Rules
- Scale up: CPU > 70% for 5 minutes
- Scale up: Request queue > 100
- Scale down: CPU < 30% for 10 minutes
- Minimum instances: 2
- Maximum instances: 10

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Classification Failures
- **Symptom**: Queries stuck at classification
- **Check**: LLM API key validity
- **Solution**: Fallback to rule-based classification

#### 2. Search API Exhaustion
- **Symptom**: No search results returned
- **Check**: API quota status
- **Solution**: Switch to alternative providers

#### 3. WebSocket Disconnections
- **Symptom**: Progress updates stop
- **Check**: Network stability, server load
- **Solution**: Implement reconnection logic

#### 4. Slow Answer Generation
- **Symptom**: Timeouts on answer generation
- **Check**: LLM API response times
- **Solution**: Implement streaming responses

## Best Practices

### 1. Paradigm Consistency
- Maintain paradigm alignment throughout pipeline
- Test with paradigm-specific queries
- Monitor paradigm classification accuracy

### 2. Performance Optimization
- Cache frequently accessed data
- Batch API requests when possible
- Use connection pooling
- Implement request deduplication

### 3. Error Resilience
- Always have fallback options
- Log errors with context
- Implement circuit breakers
- Design for graceful degradation

### 4. Security Considerations
- Rotate API keys regularly
- Validate all user inputs
- Use parameterized queries
- Implement rate limiting
- Monitor for anomalous usage

## Future Orchestration Enhancements

### 1. Advanced Features
- [ ] Multi-language support
- [ ] Custom paradigm creation
- [ ] Collaborative research sessions
- [ ] Research result versioning

### 2. Infrastructure Improvements
- [ ] Kubernetes orchestration
- [ ] Service mesh implementation
- [ ] Advanced caching strategies
- [ ] Event-driven architecture

### 3. AI/ML Enhancements
- [ ] Paradigm classification fine-tuning
- [ ] Search result ranking ML
- [ ] Answer quality scoring
- [ ] User preference learning

## Conclusion

The Four Hosts Research Application represents a sophisticated orchestration of multiple services, APIs, and AI models to deliver paradigm-aware research capabilities. Success depends on:

1. **Careful coordination** of all pipeline stages
2. **Robust error handling** at every level
3. **Performance optimization** for user experience
4. **Continuous monitoring** for system health

This orchestration guide serves as the definitive reference for understanding, operating, and extending the Four Hosts system.