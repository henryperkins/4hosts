# Four Hosts Research App - MVP Definition

## Executive Summary

The MVP will be delivered at the end of **Month 6** (Phase 5 completion) with core functionality to validate the paradigm-based research concept with real users.

**MVP Goal**: Prove that paradigm-aware research provides superior answers compared to traditional search.

---

## MVP Scope Definition

### ✅ INCLUDED in MVP

#### 1. Core Classification Engine

- [x] Query analysis for paradigm classification
- [x] Support for all 4 paradigms (Dolores, Teddy, Bernard, Maeve)
- [x] Minimum 80% classification accuracy
- [x] Single paradigm per query (no mesh network yet)

#### 2. Basic Context Engineering

- [x] Simplified W-S-C-I pipeline
- [x] Paradigm-specific search query generation
- [x] Basic compression (fixed ratios)
- [x] Key finding extraction

#### 3. Essential Research Execution

- [x] Integration with 2 search APIs (Google + Bing)
- [x] Basic source credibility scoring
- [x] Simple deduplication
- [x] Result caching (24 hours)

#### 4. Answer Generation

- [x] Paradigm-specific answer templates
- [x] Basic citation system
- [x] Plain text and markdown output
- [x] Single-language support (English)

#### 5. Web Application

- [x] Simple, clean UI for query input
- [x] Real-time paradigm visualization
- [x] Progress indicator during research
- [x] Results display with citations
- [x] Basic mobile responsiveness

#### 6. API (Limited)

- [x] POST /research/query
- [x] GET /research/status/{id}
- [x] GET /research/results/{id}
- [x] Basic authentication

#### 7. Infrastructure

- [x] Single-region deployment (US-East)
- [x] Basic monitoring (uptime, errors)
- [x] Simple caching layer
- [x] Manual scaling capability

### ❌ NOT INCLUDED in MVP (Post-MVP Features)

#### Advanced Features

- [ ] Self-healing paradigm switching
- [ ] Mesh network (multi-paradigm integration)
- [ ] User preference learning
- [ ] A/B testing framework
- [ ] Advanced analytics dashboard

#### Enhanced Capabilities

- [ ] Academic database integration
- [ ] PDF/image analysis
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Real-time collaboration

#### Enterprise Features

- [ ] Team accounts
- [ ] Custom paradigm training
- [ ] API rate limit customization
- [ ] White-label options
- [ ] SLA guarantees

#### Optimizations

- [ ] Multi-region deployment
- [ ] Advanced caching strategies
- [ ] Parallel paradigm processing
- [ ] Cost optimization algorithms
- [ ] Predictive pre-fetching

---

## MVP User Journey

### Primary Use Case: Individual Researcher

1. **User arrives at landing page**
    
    - Clear value proposition
    - Example queries shown
    - "Try it free" CTA
2. **User enters research query**
    
    - Auto-suggestions (optional)
    - Character limit: 500
    - Language: English only
3. **System shows paradigm classification**
    
    - Visual indicator (icon + color)
    - Brief explanation why
    - Processing begins
4. **Progress displayed**
    
    - Step indicators
    - Estimated time remaining
    - Cancel option
5. **Results presented**
    
    - Paradigm-appropriate answer
    - 5-10 key sources cited
    - Export as markdown
    - Feedback request

---

## MVP Success Metrics

### Technical Metrics

|Metric|Target|Measurement|
|---|---|---|
|Classification Accuracy|>80%|Manual validation (200 queries)|
|Response Time|<15 seconds|P95 latency|
|System Uptime|>98%|Monitoring dashboard|
|Error Rate|<2%|Failed queries / total|
|Search Coverage|>50 sources/query|Average analyzed|

### User Metrics

|Metric|Target|Measurement|
|---|---|---|
|User Satisfaction|>75%|Post-research survey|
|Return Users|>40%|Week 2 retention|
|Completion Rate|>70%|Queries completed|
|Paradigm Acceptance|>80%|Users who don't override|
|Export Usage|>30%|Results exported|

### Business Metrics

|Metric|Target|Measurement|
|---|---|---|
|Beta Users|500|Registered accounts|
|Daily Active Users|100|Unique users/day|
|Queries/Day|1,000|Total processed|
|Cost per Query|<$0.25|All infrastructure|
|NPS Score|>50|Standard survey|

---

## MVP Technical Architecture (Simplified)

```
┌─────────────────────┐     ┌─────────────────────┐
│   React Frontend    │────▶│   FastAPI Backend   │
│  - Query Input      │     │  - Classification   │
│  - Paradigm Display │     │  - Context Pipeline │
│  - Results View     │     │  - Search Orchestra │
└─────────────────────┘     └─────────────────────┘
                                      │
                            ┌─────────┴─────────┐
                            ▼                   ▼
                    ┌──────────────┐   ┌──────────────┐
                    │ Search APIs  │   │  PostgreSQL  │
                    │ - Google     │   │  - Results   │
                    │ - Bing       │   │  - Sessions  │
                    └──────────────┘   └──────────────┘
```

### Technology Stack (MVP)

- **Frontend**: React + TypeScript (basic setup)
- **Backend**: FastAPI (Python 3.11)
- **Database**: PostgreSQL 15
- **Cache**: Redis
- **Search**: Google Custom Search + Bing API
- **AI/ML**: OpenAI GPT-4 (classification + synthesis)
- **Hosting**: AWS (EC2 + RDS + ElastiCache)
- **Monitoring**: CloudWatch + Sentry

---

## MVP Development Priorities

### Phase-by-Phase Focus

#### Phase 1-2 (Classification + Context)

**Goal**: Accurate paradigm classification

```python
# MVP Classification (simplified)
def classify_query(query: str) -> Paradigm:
    # Rule-based + GPT-4 hybrid
    # No complex ML models yet
    # Focus on accuracy over speed
```

#### Phase 3-4 (Research + Synthesis)

**Goal**: Quality answers with real sources

```python
# MVP Research (basic)
async def research(query: str, paradigm: Paradigm):
    # 2 search APIs only
    # Simple credibility scoring
    # Basic caching
    # No advanced NLP
```

#### Phase 5 (Web App)

**Goal**: Clean, usable interface

```javascript
// MVP Frontend (essential features)
- Simple query form
- Loading states  
- Results display
- Basic error handling
- Mobile responsive
```

---

## MVP Constraints & Tradeoffs

### Performance Constraints

- Single-threaded search execution (no parallel)
- Basic caching only (24-hour TTL)
- No query optimization
- Simple rate limiting (10 req/min)

### Feature Constraints

- English only
- No user accounts (session-based)
- No saved searches
- No collaboration features
- Limited customization

### Scale Constraints

- 1,000 concurrent users max
- 10,000 queries/day limit
- Single region deployment
- Manual scaling required
- No auto-failover

---

## Post-MVP Roadmap

### MVP+1 (Month 7-8)

- User accounts & preferences
- Search history
- Basic learning from feedback
- Performance optimizations

### MVP+2 (Month 9-10)

- Self-healing mechanisms
- Mesh network (multi-paradigm)
- Advanced caching
- Multi-language support

### MVP+3 (Month 11-12)

- Enterprise features
- API v2 with full endpoints
- Analytics dashboard
- Scale to 10K users

---

## MVP Risk Mitigation

### Technical Risks

1. **Classification accuracy <80%**
    
    - Mitigation: Manual override option
    - Fallback: Default to Bernard (analytical)
2. **Search API costs exceed budget**
    
    - Mitigation: Aggressive caching
    - Fallback: Reduce search depth
3. **Response time >15 seconds**
    
    - Mitigation: Progressive loading
    - Fallback: Email results option

### User Risks

1. **Users don't understand paradigms**
    
    - Mitigation: Clear explanations
    - Fallback: Hide paradigm details
2. **Results not better than Google**
    
    - Mitigation: Focus on complex queries
    - Fallback: Pivot to niche markets

---

## MVP Launch Plan

### Week 1-2: Internal Testing

- Team dogfooding
- Bug fixes
- Performance baseline

### Week 3: Closed Beta (50 users)

- Hand-picked researchers
- Daily feedback sessions
- Rapid iteration

### Week 4: Open Beta (500 users)

- Public signup
- Feedback surveys
- Monitor all metrics

### Month 7: MVP Assessment

- Go/No-Go decision
- Pivot recommendations
- Scale-up planning

---

## Success Criteria for MVP

The MVP is considered **successful** if:

1. **Technical Success**
    
    - ✓ 80%+ classification accuracy achieved
    - ✓ <15 second response time maintained
    - ✓ <2% error rate under load
    - ✓ 98%+ uptime during beta
2. **User Success**
    
    - ✓ 75%+ user satisfaction rating
    - ✓ 40%+ week 2 retention
    - ✓ Clear preference over traditional search
    - ✓ Positive qualitative feedback
3. **Business Success**
    
    - ✓ 500 beta users acquired
    - ✓ Cost per query <$0.25
    - ✓ Clear path to monetization
    - ✓ Investor interest generated

---

## Conclusion

This MVP focuses on proving the core concept: **paradigm-aware research produces better answers**. By constraining scope to essential features, we can launch in 6 months and gather real user feedback to guide future development.

The key is to resist feature creep and maintain laser focus on the core value proposition. Everything else can wait for post-MVP iterations.