# Idea Browser Design System Applied to Four Hosts

## Design Principles from Idea Browser

### 1. Multi-Dimensional Scoring & Metrics
- **Opportunity Score** (0-10)
- **Problem Clarity** (0-10)  
- **Feasibility Rating** (0-10)
- **Growth Metrics** (+84%, 8.1K volume)
- **Revenue Projections** ($1M-$10M ARR)

### 2. Information Layers
- **Quick Scan**: Title + key metrics
- **Medium Depth**: Categories, tags, growth data
- **Deep Dive**: Full analysis with strategies
- **Contextual**: Related trends and signals

### 3. Visual Hierarchy
- Card-based layouts
- Data-driven visualizations
- Concise descriptive text
- Progressive disclosure
- Color-coded categories

### 4. Presentation Perspectives
- Business opportunity lens
- Market trend analysis
- Execution complexity
- Competitive landscape
- Target audience specificity

## Applied to Four Hosts Research

### Research Form Enhancement
```tsx
// Multi-dimensional input scoring
const researchDimensions = {
  urgency: 0-10,
  complexity: 0-10,
  business_impact: 0-10,
  paradigm_confidence: 0-100%
}

// Visual paradigm selection with metrics
const paradigmCards = {
  dolores: { score: 8.5, trending: "+12%", activeResearchers: "2.3K" },
  bernard: { score: 9.2, trending: "+5%", activeResearchers: "4.1K" },
  teddy: { score: 7.8, trending: "+18%", activeResearchers: "1.8K" },
  maeve: { score: 8.9, trending: "+23%", activeResearchers: "3.2K" }
}
```

### Progress Tracking Enhancements
```tsx
// Real-time metrics dashboard
const researchMetrics = {
  sources_quality_score: 0-10,
  paradigm_alignment: 0-100%,
  search_effectiveness: 0-100%,
  answer_confidence: 0-100%
}

// Phase-based scoring
const phaseScores = {
  classification: { accuracy: 95%, time: "0.8s" },
  context_engineering: { compression: 84%, relevance: 92% },
  search: { coverage: 87%, quality: 91% },
  synthesis: { coherence: 94%, paradigm_fit: 96% }
}
```

### Results Display Perspectives
```tsx
// Multiple view modes
const resultViews = {
  executive_summary: "Quick scan with key findings",
  paradigm_analysis: "Deep dive by consciousness type",
  source_credibility: "Quality-scored source ranking",
  action_roadmap: "Prioritized next steps",
  trend_alignment: "Market signals correlation"
}

// Scoring overlays
const resultScoring = {
  answer_quality: 0-10,
  source_diversity: 0-10,
  paradigm_coherence: 0-10,
  actionability: 0-10
}
```

## Implementation Examples

### 1. Enhanced Research Form
- Urgency/Impact matrix selector
- Paradigm confidence slider
- Related trends suggestions
- Budget/depth calculator
- Visual complexity indicator

### 2. Progress Dashboard
- Live quality metrics
- Source discovery feed
- Paradigm alignment gauge
- Search effectiveness chart
- Time/cost tracking

### 3. Results Presentation
- Expandable insight cards
- Quality score badges
- Trend correlation indicators
- Visual source mapping
- Paradigm strength meters

## Design Tokens

```css
/* Metrics colors */
--metric-excellent: #10b981; /* 8-10 score */
--metric-good: #3b82f6;      /* 6-8 score */
--metric-fair: #f59e0b;      /* 4-6 score */
--metric-poor: #ef4444;      /* 0-4 score */

/* Trend indicators */
--trend-up: #10b981;
--trend-stable: #6b7280;
--trend-down: #ef4444;

/* Paradigm colors (existing) */
--paradigm-dolores: #dc2626;
--paradigm-bernard: #3b82f6;
--paradigm-teddy: #10b981;
--paradigm-maeve: #a855f7;
```