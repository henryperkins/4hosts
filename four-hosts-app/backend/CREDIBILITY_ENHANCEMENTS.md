# Credibility Assessment System Enhancements

## Overview
The credibility assessment system in `services/credibility.py` has been significantly enhanced with comprehensive features for more accurate and nuanced source evaluation.

## New Features Implemented

### 1. **Comprehensive Credibility Features**
- **Recency Score**: Exponential decay modeling based on publication date
- **Cross-Source Agreement**: Measures consensus among multiple sources
- **Controversy Score**: Identifies and quantifies controversial content
- **Update Frequency**: Tracks how often sources publish new content
- **Source Category**: Classifies sources (news, academic, blog, social, etc.)
- **Topic-Specific Credibility**: Domain expertise scoring

### 2. **Enhanced Source Database**
- Expanded from ~20 to **50+ sources** with detailed metadata
- Categories include:
  - News (left/right/center leaning)
  - Academic/Scientific
  - Reference/Fact-checking
  - Tech/Blog
  - Social Media
- Each source includes:
  - Bias rating and score
  - Factual accuracy rating
  - Category classification
  - Update frequency

### 3. **Controversy Detection System**
```python
class ControversyDetector:
    - Identifies controversial topics (politics, social issues, health, etc.)
    - Calculates controversy scores based on:
      - Topic keywords
      - Polarizing sources
      - Conflicting viewpoints
      - Source disagreement
```

### 4. **Temporal Credibility Modeling**
```python
class RecencyModeler:
    - Domain-specific decay rates:
      - News: 7 days to 50% credibility
      - Academic: 365 days
      - Social Media: 1 day
    - Breaking news boost (< 24 hours)
    - Update frequency detection
```

### 5. **Credibility Cards**
New `generate_credibility_card()` method provides:
- Visual trust level (Very High → Very Low)
- Key credibility factors
- Strengths and concerns
- Actionable recommendations
- Paradigm alignment scores

## Enhanced Scoring Algorithm

### Old Algorithm (40% Authority, 30% Bias, 30% Factual)
```python
score = (domain_authority * 0.4) + (bias_score * 0.3) + (factual * 0.3)
```

### New Algorithm (Comprehensive)
```python
score = (
    domain_authority * 0.2 +     # 20% - Traditional authority
    bias_score * 0.15 +          # 15% - Political neutrality
    factual_accuracy * 0.25 +    # 25% - Fact checking
    recency_score * 0.15 +       # 15% - Temporal relevance
    (1 - controversy) * 0.15 +   # 15% - Controversy penalty
    cross_agreement * 0.10       # 10% - Source consensus
)
```

## API Changes

### Enhanced `get_source_credibility()` Function
```python
async def get_source_credibility(
    domain: str,
    paradigm: str = "bernard",
    content: Optional[str] = None,              # NEW: Analyze content
    search_terms: Optional[List[str]] = None,   # NEW: Context terms
    published_date: Optional[datetime] = None,   # NEW: Temporal factor
    other_sources: Optional[List[Dict]] = None   # NEW: Cross-reference
) -> CredibilityScore
```

## Usage Examples

### Basic Credibility Check
```python
credibility = await get_source_credibility("nytimes.com")
```

### Controversy Detection
```python
credibility = await get_source_credibility(
    domain="foxnews.com",
    search_terms=["vaccine", "mandate", "freedom"]
)
print(f"Controversy: {credibility.controversy_score}")
```

### Cross-Source Agreement
```python
other_sources = [
    {"domain": "reuters.com", "bias_score": 0.9},
    {"domain": "apnews.com", "bias_score": 0.9}
]
credibility = await get_source_credibility(
    domain="bbc.com",
    other_sources=other_sources
)
print(f"Agreement: {credibility.cross_source_agreement}")
```

### Generate Credibility Card
```python
card = credibility.generate_credibility_card()
# Returns structured assessment with recommendations
```

## Benefits

1. **More Accurate Assessment**: Multiple factors provide nuanced evaluation
2. **Temporal Awareness**: Recognizes when information may be outdated
3. **Controversy Detection**: Warns users about polarizing content
4. **Paradigm Alignment**: Tailored scoring for each Host paradigm
5. **Actionable Insights**: Specific recommendations for users

## Integration Points

- **Search Results**: Enhanced credibility scores in search results
- **Answer Generation**: Weight sources by credibility
- **Research Reports**: Include credibility cards
- **WebSocket Updates**: Real-time credibility alerts
- **Export Service**: Include credibility assessments in exports

## Future Enhancements

1. Machine learning for bias detection
2. Real-time fact-checking integration
3. User feedback loop for credibility
4. Social proof metrics from APIs
5. Historical credibility tracking