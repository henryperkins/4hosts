# Four Hosts Research API Specification v1.0

## Overview

RESTful API for the Four Hosts Agentic Research Application, providing paradigm-aware research capabilities based on Westworld host consciousness models.

---

## Base URL

```
https://api.fourhoststresearch.com/v1
```

---

## Authentication

```http
Authorization: Bearer YOUR_API_KEY
```

---

## Core Endpoints

### 1. Research Query Submission

#### POST `/research/query`

Submit a research query for paradigm-based analysis.

**Request Body:**

```json
{
  "query": "How can small businesses compete with Amazon?",
  "options": {
    "depth": "standard",              // quick | standard | deep
    "paradigm_override": null,        // null | "dolores" | "teddy" | "bernard" | "maeve"
    "include_secondary": true,        // Include secondary paradigm analysis
    "max_sources": 100,              // Maximum sources to analyze
    "language": "en",                // ISO language code
    "region": "us"                   // Target region for results
  }
}
```

**Response:**

```json
{
  "research_id": "res_abc123def456",
  "status": "processing",
  "paradigm_classification": {
    "primary": "maeve",
    "secondary": "dolores",
    "distribution": {
      "maeve": 0.40,
      "dolores": 0.25,
      "bernard": 0.20,
      "teddy": 0.15
    },
    "confidence": 0.78
  },
  "estimated_completion": "2025-01-20T10:30:45Z",
  "webhook_url": "https://api.fourhoststresearch.com/v1/research/res_abc123def456/webhook"
}
```

---

### 2. Get Research Status

#### GET `/research/status/{research_id}`

Check the status of an ongoing research task.

**Response:**

```json
{
  "research_id": "res_abc123def456",
  "status": "in_progress",           // queued | processing | in_progress | completed | failed
  "progress": {
    "current_phase": "research_execution",
    "phases_completed": [
      "paradigm_classification",
      "context_engineering"
    ],
    "searches_executed": 12,
    "sources_analyzed": 847,
    "completion_percentage": 65
  },
  "paradigm": "maeve",
  "started_at": "2025-01-20T10:30:00Z",
  "updated_at": "2025-01-20T10:30:30Z"
}
```

---

### 3. Get Research Results

#### GET `/research/results/{research_id}`

Retrieve completed research results.

**Response:**

```json
{
  "research_id": "res_abc123def456",
  "query": "How can small businesses compete with Amazon?",
  "status": "completed",
  "paradigm_analysis": {
    "primary": {
      "paradigm": "maeve",
      "confidence": 0.78,
      "approach": "strategic",
      "focus": "Actionable competitive strategies and implementation frameworks"
    },
    "secondary": {
      "paradigm": "dolores",
      "confidence": 0.52,
      "approach": "revolutionary",
      "focus": "Systemic issues and resistance opportunities"
    }
  },
  "answer": {
    "summary": "Strategic framework for small businesses to effectively compete with Amazon through local advantages, relationship commerce, and niche domination.",
    "sections": [
      {
        "title": "Immediate Strategic Advantages",
        "paradigm": "maeve",
        "content": "1. Local Presence Leverage...",
        "confidence": 0.92,
        "sources_count": 23
      },
      {
        "title": "Systemic Context",
        "paradigm": "dolores",
        "content": "Growing antitrust scrutiny...",
        "confidence": 0.85,
        "sources_count": 12
      }
    ],
    "action_items": [
      {
        "priority": "high",
        "action": "Implement same-day local delivery",
        "timeframe": "2-4 weeks",
        "paradigm": "maeve"
      }
    ],
    "citations": [
      {
        "id": "cite_001",
        "source": "Harvard Business Review",
        "title": "Competing with Digital Giants",
        "url": "https://hbr.org/...",
        "credibility_score": 0.95,
        "paradigm_alignment": "maeve"
      }
    ]
  },
  "metadata": {
    "total_sources_analyzed": 3765,
    "high_quality_sources": 47,
    "search_queries_executed": 16,
    "processing_time_seconds": 4.7,
    "paradigms_used": ["maeve", "dolores"],
    "context_layers": {
      "write_focus": "Map competitive landscape and opportunities",
      "compression_ratio": 0.4,
      "isolation_strategy": "High-leverage strategic opportunities"
    }
  },
  "export_formats": {
    "pdf": "/research/res_abc123def456/export/pdf",
    "markdown": "/research/res_abc123def456/export/markdown",
    "json": "/research/res_abc123def456/export/json"
  }
}
```

---

### 4. Paradigm Classification Only

#### POST `/paradigms/classify`

Get paradigm classification without full research.

**Request Body:**

```json
{
  "query": "How to protect endangered species in urban areas?"
}
```

**Response:**

```json
{
  "query": "How to protect endangered species in urban areas?",
  "classification": {
    "primary": "teddy",
    "secondary": "bernard",
    "distribution": {
      "teddy": 0.45,
      "bernard": 0.30,
      "dolores": 0.15,
      "maeve": 0.10
    }
  },
  "explanation": {
    "teddy": "Focus on 'protect' and 'endangered' indicates devotion/protection paradigm",
    "bernard": "Scientific aspect of species and urban ecology suggests analytical approach"
  },
  "suggested_approach": "Combine protection-focused research with scientific analysis"
}
```

---

### 5. Paradigm Override

#### POST `/paradigms/override`

Force a specific paradigm for research.

**Request Body:**

```json
{
  "research_id": "res_abc123def456",
  "paradigm": "bernard",
  "reason": "Need more analytical depth"
}
```

**Response:**

```json
{
  "success": true,
  "research_id": "res_abc123def456",
  "previous_paradigm": "maeve",
  "new_paradigm": "bernard",
  "status": "re-processing",
  "message": "Research restarted with BERNARD (analytical) paradigm"
}
```

---

### 6. Search with Paradigm Context

#### POST `/search/paradigm-aware`

Execute a single paradigm-aware search.

**Request Body:**

```json
{
  "query": "renewable energy investment strategies",
  "paradigm": "maeve",
  "options": {
    "source_types": ["industry_reports", "strategic_analysis"],
    "max_results": 50,
    "date_range": "2023-2025"
  }
}
```

**Response:**

```json
{
  "query": "renewable energy investment strategies",
  "paradigm": "maeve",
  "search_modifications": [
    "Added: 'ROI', 'competitive advantage'",
    "Prioritized: Industry and consultancy sources",
    "Filtered: Theoretical papers"
  ],
  "results": [
    {
      "title": "Strategic Framework for Renewable Energy Investment",
      "source": "McKinsey & Company",
      "url": "https://...",
      "relevance_score": 0.94,
      "paradigm_alignment": 0.91,
      "key_insights": [
        "5 high-leverage investment opportunities",
        "Risk mitigation strategies",
        "10-year ROI projections"
      ]
    }
  ],
  "total_results": 187,
  "filtered_results": 42
}
```

---

### 7. Get Paradigm Explanation

#### GET `/paradigms/explanation/{paradigm}`

Get detailed explanation of a specific paradigm.

**Response:**

```json
{
  "paradigm": "maeve",
  "name": "Strategic Paradigm",
  "description": "Research approach focused on gaining control, strategic advantage, and actionable outcomes",
  "host_inspiration": "Maeve Millay - The strategist who rewrites her own code",
  "characteristics": {
    "core_values": ["Strategy", "Control", "Optimization", "Leverage"],
    "methodology": "Strategic Design Research",
    "researcher_role": "Strategist-Hacker",
    "key_practices": [
      "Competitive analysis",
      "Stakeholder mapping", 
      "Implementation planning",
      "ROI optimization"
    ]
  },
  "when_to_use": [
    "Business competition questions",
    "Influence and persuasion needs",
    "Optimization problems",
    "Strategic planning"
  ],
  "example_queries": [
    "How to influence policy makers?",
    "Best strategy to enter new market",
    "Optimize conversion rates"
  ]
}
```

---

### 8. User Preferences

#### POST `/users/preferences`

Set user research preferences.

**Request Body:**

```json
{
  "user_id": "user_123",
  "preferences": {
    "default_paradigm": null,
    "paradigm_weights": {
      "dolores": 1.0,
      "teddy": 1.2,      // Slight preference for protection/care
      "bernard": 1.0,
      "maeve": 0.8       // Slight reduction in strategic
    },
    "source_preferences": {
      "academic": true,
      "industry": true,
      "alternative": false,
      "social_media": false
    },
    "output_style": "detailed",  // concise | standard | detailed
    "include_citations": true
  }
}
```

---

### 9. Research History

#### GET `/users/history`

Get user's research history.

**Response:**

```json
{
  "user_id": "user_123",
  "total_researches": 47,
  "paradigm_usage": {
    "teddy": 18,
    "bernard": 15,
    "maeve": 10,
    "dolores": 4
  },
  "recent_researches": [
    {
      "research_id": "res_abc123def456",
      "query": "How can small businesses compete with Amazon?",
      "paradigm": "maeve",
      "timestamp": "2025-01-20T10:30:00Z",
      "satisfaction_rating": 5
    }
  ],
  "favorite_topics": ["business strategy", "social justice", "technology"],
  "average_satisfaction": 4.7
}
```

---

### 10. Submit Feedback

#### POST `/research/feedback`

Provide feedback on research quality.

**Request Body:**

```json
{
  "research_id": "res_abc123def456",
  "rating": 5,
  "paradigm_fit": "excellent",     // poor | fair | good | excellent
  "feedback": {
    "what_worked": "Strategic framework was immediately actionable",
    "what_didnt": "Could use more local examples",
    "alternative_paradigm": null    // Suggest different paradigm
  }
}
```

---

## Webhook Events

### Research Completed

```json
{
  "event": "research.completed",
  "research_id": "res_abc123def456",
  "timestamp": "2025-01-20T10:30:45Z",
  "data": {
    "query": "How can small businesses compete with Amazon?",
    "paradigm": "maeve",
    "success": true,
    "results_url": "https://api.fourhoststresearch.com/v1/research/results/res_abc123def456"
  }
}
```

### Paradigm Switch (Self-Healing)

```json
{
  "event": "paradigm.switched",
  "research_id": "res_abc123def456",
  "timestamp": "2025-01-20T10:30:30Z",
  "data": {
    "reason": "Detected analysis paralysis",
    "from_paradigm": "bernard",
    "to_paradigm": "maeve",
    "confidence": 0.89
  }
}
```

---

## Error Responses

### 400 Bad Request

```json
{
  "error": {
    "code": "invalid_query",
    "message": "Query must be between 10 and 500 characters",
    "details": {
      "provided_length": 7,
      "min_length": 10,
      "max_length": 500
    }
  }
}
```

### 429 Rate Limited

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Please retry after cooldown.",
    "details": {
      "limit": 100,
      "window": "1h",
      "retry_after": "2025-01-20T11:30:00Z"
    }
  }
}
```

---

## Rate Limits

|Plan|Requests/Hour|Concurrent Research|Max Sources/Query|
|---|---|---|---|
|Free|10|1|50|
|Basic|100|5|200|
|Pro|1000|20|1000|
|Enterprise|Custom|Custom|Unlimited|

---

## SDKs

Official SDKs available for:

- Python: `pip install fourhost-research`
- JavaScript/TypeScript: `npm install @fourhost/research-sdk`
- Go: `go get github.com/fourhost/research-sdk-go`
- Ruby: `gem install fourhost-research`

---

## Example: Complete Research Flow

```python
import fourhost

# Initialize client
client = fourhost.Client(api_key="YOUR_API_KEY")

# Submit research query
research = client.research.create(
    query="How can small businesses compete with Amazon?",
    options={
        "depth": "standard",
        "include_secondary": True
    }
)

# Wait for completion (or use webhooks)
result = research.wait_for_completion()

# Access results
print(f"Paradigm: {result.paradigm}")
print(f"Answer: {result.answer.summary}")

# Export as PDF
pdf_url = result.export("pdf")
```

---

## Support

- Documentation: https://docs.fourhoststresearch.com
- API Status: https://status.fourhoststresearch.com
- Support: support@fourhoststresearch.com
- Discord: https://discord.gg/fourhost