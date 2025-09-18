# Brave Search API Integration

This document describes the Brave Search API integration in the Four Hosts Research Application.

## Overview

The Brave Search API has been integrated as an additional search provider alongside Google Custom Search, Bing, ArXiv, and PubMed. It provides comprehensive web search capabilities with support for news, FAQ, discussions, and standard web results.

## Features

- **Multiple Result Types**: Web, News, FAQ, and Discussion results
- **Content Filtering**: Safe search with off/moderate/strict settings  
- **Date Filtering**: Filter by freshness (past day/week/month/year)
- **Rate Limiting**: Built-in rate limit handling and monitoring
- **Paradigm Integration**: Works with the Four Hosts classification system

## Setup

1. **Get an API Key**
   - Visit https://api-dashboard.search.brave.com/
   - Sign up for a free account
   - Navigate to the API Keys section
   - Create a new API key

2. **Configure Environment**
   ```bash
   # Add to your .env file
   BRAVE_SEARCH_API_KEY=your_brave_search_api_key_here
   ```

3. **Test the Integration**
   ```bash
   cd /home/azureuser/4hosts/four-hosts-app/backend
   python test_brave_search.py
   ```

## Usage

### Basic Search
```python
from services.search_apis import BraveSearchAPI, SearchConfig

# Initialize API
brave_api = BraveSearchAPI(api_key="your_key")

# Configure search
config = SearchConfig(
    max_results=20,
    language="en", 
    region="us",
    safe_search="moderate"
)

# Perform search
async with brave_api:
    results = await brave_api.search("your query", config)
```

### With Search Manager
```python
from services.search_apis import create_search_manager

# Create manager (automatically loads from environment)
manager = create_search_manager()

# Search with automatic fallback
results = await manager.search_with_fallback("your query", config)
```

### Paradigm-Aware Search
See `examples/brave_search_example.py` for a complete example of using Brave Search with the Four Hosts paradigm classification system.

## API Limits

- **Free Tier**: 2,000 searches per month
- **Rate Limits**: 1 request per second
- **Results per Request**: Maximum 20 results

## Result Types

### Web Results
Standard web page results with title, URL, snippet, and domain.

### News Results  
Recent news articles with publication dates and source information.

### FAQ Results
Question and answer pairs extracted from web content.

### Discussion Results
Forum and discussion board content with top comments.

## Integration Points

1. **SearchAPIManager**: Brave is automatically added when `BRAVE_SEARCH_API_KEY` is set
2. **Fallback Priority**: Can be primary if Google is not configured
3. **Research Orchestrator**: Used for paradigm-specific searches
4. **Result Aggregation**: Combined with other API results

## Testing

Run the comprehensive test suite:
```bash
python test_brave_search.py
```

This will test:
- Basic web search
- News search with date filtering
- Academic-style search (FAQ & discussions)
- Safe search filtering
- SearchAPIManager integration
- Rate limit handling

## Troubleshooting

### Authentication Errors
- Verify your API key is correct
- Check that the key is properly set in `.env`
- Ensure the key has not expired

### Rate Limit Errors
- The API allows 1 request per second
- Monthly quota is 2,000 searches on free tier
- Check response headers for remaining quota

### No Results
- Try broader search terms
- Check safe search settings
- Verify region/language parameters

## Examples

See the following files for usage examples:
- `test_brave_search.py` - Comprehensive test suite
- `examples/brave_search_example.py` - Paradigm-aware search examples