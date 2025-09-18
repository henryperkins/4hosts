# Search Relevance and Content Filtering Improvements

## Overview
This document details the improvements made to enhance search term relevance and implement early-stage content filtering in the Four Hosts research pipeline.

## Key Improvements

### 1. Query Optimization (search_apis.py)
- **QueryOptimizer Class**: Intelligently processes search queries before sending to APIs
  - Removes common "fluff" words that reduce search relevance
  - Extracts key terms and meaningful phrases
  - Preserves quoted phrases and important multi-word terms
  - Automatically optimizes long queries by focusing on most important terms

### 2. Content Relevance Scoring (search_apis.py)
- **ContentRelevanceFilter Class**: Calculates relevance scores for all search results
  - **Term Frequency Analysis (40% weight)**: Measures how many key terms appear in results
  - **Title Relevance (30% weight)**: Higher scores for key terms in titles
  - **Content Freshness (10% weight)**: Newer content gets higher scores
  - **Source Type Scoring (10% weight)**: Academic and primary sources get bonuses
  - **Phrase Matching (10% weight)**: Exact phrase matches are rewarded
  - Identifies primary sources based on domain and content indicators

### 3. Paradigm-Specific Query Enhancement (paradigm_search.py)
Enhanced all four paradigm search strategies:

#### Dolores (Revolutionary)
- Smart modifier selection based on query content (e.g., "corporation" → "corrupt", "scandal")
- Strategic modifier insertion for long queries
- Context-aware investigative patterns (different for companies vs. government queries)

#### Teddy (Devotion)
- Removes unhelpful emotional terms that don't aid in finding resources
- Detects need type and suggests appropriate modifiers (mental health, food, housing, etc.)
- Location-aware search patterns when location is mentioned
- Emergency-specific patterns for urgent queries

#### Bernard (Analytical)
- Converts colloquial language to academic terms
- Field-specific modifiers based on detected research domain
- Adds citation-focused queries for established research topics
- Includes recent publication date ranges

#### Maeve (Strategic)
- Removes business jargon that doesn't help search
- Industry-specific modifiers (tech, retail, finance, healthcare, etc.)
- ROI and implementation-focused patterns
- Current year industry reports

### 4. Early-Stage Content Filtering (research_orchestrator.py)
- **EarlyRelevanceFilter Class**: Removes obviously irrelevant results before expensive processing
  - **Spam Detection**: Filters common spam keywords
  - **Low-Quality Domain Filtering**: Blocks known content farms
  - **Content Validation**: Ensures minimum title/snippet length
  - **Language Detection**: Basic non-English content filtering
  - **Query Term Presence**: Ensures at least one query term appears
  - **Duplicate Site Detection**: Identifies mirror/cache sites
  - **Paradigm-Specific Filters**: E.g., Bernard requires academic indicators for web results

### 5. Multi-Stage Filtering Pipeline
The improved pipeline now follows this flow:
1. **Query Optimization** → Clean and optimize search terms
2. **API Search** → Execute searches with optimized queries
3. **Initial Relevance Filtering** → Apply relevance scoring at API level
4. **Deduplication** → Remove duplicate results
5. **Early Content Filtering** → Remove spam and irrelevant content
6. **Paradigm-Specific Filtering** → Apply paradigm-specific ranking
7. **Credibility Assessment** → Final credibility scoring

## Benefits
- **Higher Quality Results**: Irrelevant and spam content filtered early
- **Better Search Terms**: Optimized queries lead to more relevant API results
- **Resource Efficiency**: Early filtering reduces processing of bad content
- **Paradigm Alignment**: Search terms better match paradigm perspectives
- **Primary Source Identification**: Automatic detection of authoritative sources

## Technical Details
- Uses NLTK for natural language processing
- Regex patterns for spam and duplicate detection
- Weighted scoring system for relevance calculation
- Configurable minimum relevance thresholds per search API