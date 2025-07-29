# Complete Brave Search API Guide

---

## Table of Contents

1. [Introduction & Overview](#introduction--overview)
2. [API Endpoints](#api-endpoints)
3. [Authentication & Setup](#authentication--setup)
4. [Request Parameters](#request-parameters)
5. [Request Headers](#request-headers)
6. [Response Headers](#response-headers)
7. [Response Objects & Models](#response-objects--models)
8. [Complete Implementation Examples](#complete-implementation-examples)
9. [Best Practices & Optimization](#best-practices--optimization)
10. [Rate Limiting & Error Handling](#rate-limiting--error-handling)

---

## Introduction & Overview

The Brave Web Search API is a REST API that provides programmatic access to Brave Search's web search results. It offers multiple endpoints for different types of data and supports both general web search and location-specific queries.

### Key Features
- **Web Search**: General web search with comprehensive result types
- **Local Search**: Location-specific results with enhanced POI data
- **Multiple Result Types**: Web results, news, videos, FAQ, discussions, infoboxes, locations
- **AI Features**: Summarization capabilities and AI-generated location descriptions
- **Custom Ranking**: Support for Goggles (custom re-ranking)
- **Rich Metadata**: Extensive structured data and enrichment

### Subscription Requirements
- **Free Plan**: Basic access (requires subscription but no charges)
- **Pro Plans**: Required for Local Search API access
- **API Keys**: Available in the [API Keys section](https://api-dashboard.search.brave.com/app/keys)

---

## API Endpoints

### Web Search API
```
https://api.search.brave.com/res/v1/web/search
```
Primary endpoint for web search queries with comprehensive result types.

### Local Search API
```
https://api.search.brave.com/res/v1/local/pois
```
Endpoint for retrieving extra information about locations (up to 20 locations per request).

```
https://api.search.brave.com/res/v1/local/descriptions
```
Endpoint for AI-generated location descriptions.

---

## Authentication & Setup

### API Key
All requests require authentication via the `X-Subscription-Token` header:

```bash
-H "X-Subscription-Token: <YOUR_API_KEY>"
```

### Basic Request Structure
```bash
curl -s --compressed "https://api.search.brave.com/res/v1/web/search?q=your+query" \
  -H "Accept: application/json" \
  -H "Accept-Encoding: gzip" \
  -H "X-Subscription-Token: <YOUR_API_KEY>"
```

---

## Request Parameters

### Web Search API Parameters

#### Required Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | string | Search query (max 400 chars, 50 words) |

#### Core Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `country` | string | `US` | 2-character country code |
| `search_lang` | string | `en` | 2+ character language code |
| `ui_lang` | string | `en-US` | UI language (format: `<lang>-<country>`) |
| `count` | int | `20` | Results per page (max 20, applies to web results only) |
| `offset` | int | `0` | Results offset for pagination (max 9) |

#### Content Filtering
| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `safesearch` | string | `off`, `moderate`, `strict` | Adult content filtering |
| `freshness` | string | `pd`, `pw`, `pm`, `py`, `YYYY-MM-DDtoYYYY-MM-DD` | Time-based filtering |

#### Result Customization
| Parameter | Type | Description |
|-----------|------|-------------|
| `result_filter` | string | Comma-separated result types: `discussions`, `faq`, `infobox`, `news`, `query`, `summarizer`, `videos`, `web`, `locations` |
| `text_decorations` | bool | `true` - Include highlighting in snippets |
| `spellcheck` | bool | `true` - Enable spell correction |
| `extra_snippets` | bool | Get up to 5 additional excerpts |
| `summary` | bool | Enable summary key generation |

#### Advanced Features
| Parameter | Type | Description |
|-----------|------|-------------|
| `goggles` | list[string] | Custom re-ranking (URL or definition) |
| `goggles_id` | string | Deprecated - use `goggles` instead |
| `units` | string | `metric` or `imperial` measurement units |

### Local Search API Parameters

#### Required Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `ids` | list[string] | Location IDs (max 20 per request) |

#### Optional Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_lang` | string | `en` | Search language |
| `ui_lang` | string | `en-US` | UI language |
| `units` | string | - | Measurement units |

---

## Request Headers

### Web Search API Headers

#### Required Headers
| Header | Description |
|--------|-------------|
| `X-Subscription-Token` | API authentication token |

#### Recommended Headers
| Header | Example | Description |
|--------|---------|-------------|
| `Accept` | `application/json` | Response format |
| `Accept-Encoding` | `gzip` | Compression support |
| `User-Agent` | Platform-specific string | Client identification |

#### Location Headers (Optional but Recommended)
| Header | Format | Range | Description |
|--------|--------|--------|-------------|
| `X-Loc-Lat` | float | -90.0 to +90.0 | Latitude in degrees |
| `X-Loc-Long` | float | -180.0 to +180.0 | Longitude in degrees |
| `X-Loc-Timezone` | string | IANA format | e.g., `America/New_York` |
| `X-Loc-City` | string | - | City name |
| `X-Loc-State` | string | 3 chars max | State/region code |
| `X-Loc-State-Name` | string | - | State/region name |
| `X-Loc-Country` | string | ISO 3166-1 alpha-2 | 2-letter country code |
| `X-Loc-Postal-Code` | string | - | Postal code |

#### Control Headers
| Header | Values | Description |
|--------|--------|-------------|
| `Api-Version` | `YYYY-MM-DD` | API version specification |
| `Cache-Control` | `no-cache` | Bypass cached results |

#### User Agent Examples by Platform
```
Android: Mozilla/5.0 (Linux; Android 13; Pixel 7 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Mobile Safari/537.36

iOS: Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1

macOS: Mozilla/5.0 (Macintosh; Intel Mac OS X 12_0_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/

Windows: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/
```

### Local Search API Headers
Similar to Web Search API but with simplified requirements:
- `X-Subscription-Token` (required)
- Standard headers (`Accept`, `Accept-Encoding`, `Api-Version`, `Cache-Control`, `User-Agent`)
- No location-specific headers needed

---

## Response Headers

### Rate Limiting Headers
All API responses include rate limiting information:

| Header | Format | Description |
|--------|--------|-------------|
| `X-RateLimit-Limit` | `1, 15000` | Current limits (requests/sec, requests/month) |
| `X-RateLimit-Policy` | `1;w=1, 15000;w=2592000` | Limit policies with time windows (seconds) |
| `X-RateLimit-Remaining` | `1, 1000` | Remaining quota for current windows |
| `X-RateLimit-Reset` | `1, 1419704` | Seconds until quota resets |

### Rate Limit Example
```
X-RateLimit-Limit: 1, 15000
X-RateLimit-Policy: 1;w=1, 15000;w=2592000
X-RateLimit-Remaining: 1, 1000
X-RateLimit-Reset: 1, 1419704
```
This indicates: 1 request per second, 15,000 per month; currently 1 request available this second, 1,000 requests remaining this month.

---

## Response Objects & Models

### Top-Level Response Models

#### WebSearchApiResponse
```json
{
  "type": "search",
  "query": { /* Query object */ },
  "web": { /* Web results */ },
  "news": { /* News results */ },
  "videos": { /* Video results */ },
  "locations": { /* Location results */ },
  "infobox": { /* Infobox data */ },
  "discussions": { /* Forum discussions */ },
  "faq": { /* FAQ results */ },
  "mixed": { /* Result ranking */ },
  "summarizer": { /* Summary key */ },
  "rich": { /* Rich callback info */ }
}
```

#### LocalPoiSearchApiResponse
```json
{
  "type": "local_pois",
  "results": [ /* LocationResult objects */ ]
}
```

#### LocalDescriptionsSearchApiResponse
```json
{
  "type": "local_descriptions",
  "results": [ /* LocationDescription objects */ ]
}
```

### Core Result Types

#### Search Results (Web)
```json
{
  "type": "search",
  "results": [
    {
      "type": "search_result",
      "title": "Page Title",
      "url": "https://example.com",
      "description": "Page description...",
      "age": "2024-01-15T10:30:00Z",
      "meta_url": {
        "scheme": "https",
        "netloc": "example.com",
        "hostname": "example.com",
        "favicon": "https://example.com/favicon.ico"
      }
    }
  ]
}
```

#### News Results
```json
{
  "type": "news",
  "results": [
    {
      "title": "Breaking News Title",
      "url": "https://news.example.com/article",
      "description": "Article summary...",
      "age": "2024-01-15T10:30:00Z",
      "source": "News Source",
      "breaking": true,
      "thumbnail": {
        "src": "https://example.com/thumb.jpg"
      }
    }
  ]
}
```

#### Video Results
```json
{
  "type": "videos",
  "results": [
    {
      "type": "video_result",
      "title": "Video Title",
      "url": "https://video.example.com",
      "video": {
        "duration": "05:42",
        "views": "1.2M views",
        "creator": "Creator Name",
        "thumbnail": {
          "src": "https://example.com/video-thumb.jpg"
        }
      }
    }
  ]
}
```

#### Location Results
```json
{
  "type": "locations",
  "results": [
    {
      "type": "location_result",
      "id": "temp_location_id_12345",
      "title": "Business Name",
      "coordinates": [37.7749, -122.4194],
      "postal_address": {
        "streetAddress": "123 Main St",
        "addressLocality": "San Francisco",
        "addressRegion": "CA",
        "postalCode": "94105",
        "country": "US"
      },
      "rating": {
        "ratingValue": 4.5,
        "reviewCount": 250
      },
      "opening_hours": {
        "current_day": [
          {
            "opens": "09:00",
            "closes": "18:00"
          }
        ]
      }
    }
  ]
}
```

### Query Object
```json
{
  "original": "user search query",
  "altered": "corrected search query",
  "safesearch": true,
  "is_navigational": false,
  "is_geolocal": true,
  "country": "US",
  "language": {
    "main": "en"
  }
}
```

### Infobox Types

#### Generic Infobox
```json
{
  "type": "graph",
  "results": {
    "type": "infobox",
    "subtype": "generic",
    "title": "Entity Name",
    "description": "Entity description...",
    "thumbnail": {
      "src": "https://example.com/entity-image.jpg"
    },
    "attributes": [
      ["Founded", "2003"],
      ["Headquarters", "San Francisco, CA"]
    ]
  }
}
```

#### Location Infobox
```json
{
  "type": "graph",
  "results": {
    "subtype": "location",
    "coordinates": [37.7749, -122.4194],
    "zoom_level": 10,
    "location": { /* LocationResult object */ }
  }
}
```

### Specialized Result Types

#### FAQ Results
```json
{
  "type": "faq",
  "results": [
    {
      "question": "What is...?",
      "answer": "The answer is...",
      "title": "Source Page Title",
      "url": "https://source.example.com",
      "meta_url": { /* MetaUrl object */ }
    }
  ]
}
```

#### Discussion Results
```json
{
  "type": "search",
  "results": [
    {
      "type": "discussion",
      "data": {
        "forum_name": "Reddit",
        "num_answers": 42,
        "score": "15",
        "title": "Discussion Title",
        "question": "Original question...",
        "top_comment": "Top voted response..."
      }
    }
  ]
}
```

### Rich Media Objects

#### Product Information
```json
{
  "type": "Product",
  "name": "Product Name",
  "category": "Electronics",
  "price": "$299.99",
  "description": "Product description...",
  "thumbnail": {
    "src": "https://example.com/product.jpg"
  },
  "rating": {
    "ratingValue": 4.2,
    "reviewCount": 150
  }
}
```

#### Recipe Data
```json
{
  "title": "Chocolate Chip Cookies",
  "description": "Delicious homemade cookies...",
  "time": "30 minutes",
  "servings": 24,
  "ingredients": "2 cups flour, 1 cup sugar...",
  "instructions": [
    {
      "text": "Preheat oven to 350°F",
      "name": "Step 1"
    }
  ],
  "rating": {
    "ratingValue": 4.8,
    "reviewCount": 89
  }
}
```

### Mixed Response (Result Ranking)
```json
{
  "type": "mixed",
  "main": [
    {"type": "web", "index": 0, "all": false},
    {"type": "news", "index": 0, "all": true},
    {"type": "videos", "index": 2, "all": false}
  ],
  "top": [
    {"type": "infobox", "index": 0, "all": true}
  ],
  "side": [
    {"type": "locations", "index": 0, "all": true}
  ]
}
```

---

## Complete Implementation Examples

### Basic Web Search
```bash
curl -s --compressed "https://api.search.brave.com/res/v1/web/search?q=artificial+intelligence" \
  -H "Accept: application/json" \
  -H "Accept-Encoding: gzip" \
  -H "X-Subscription-Token: YOUR_API_KEY"
```

### Advanced Web Search with Location
```bash
curl -s --compressed "https://api.search.brave.com/res/v1/web/search" \
  -G \
  -d "q=best+restaurants" \
  -d "country=US" \
  -d "search_lang=en" \
  -d "count=10" \
  -d "safesearch=moderate" \
  -d "result_filter=web,locations,reviews" \
  -H "Accept: application/json" \
  -H "Accept-Encoding: gzip" \
  -H "X-Loc-Lat: 37.7749" \
  -H "X-Loc-Long: -122.4194" \
  -H "X-Loc-City: San Francisco" \
  -H "X-Loc-State: CA" \
  -H "X-Loc-Country: US" \
  -H "X-Subscription-Token: YOUR_API_KEY"
```

### Local POI Search Workflow
```bash
# Step 1: Get location IDs from web search
curl -s --compressed "https://api.search.brave.com/res/v1/web/search?q=greek+restaurants+in+san+francisco&result_filter=locations" \
  -H "X-Subscription-Token: YOUR_API_KEY"

# Step 2: Get detailed location information
curl -s --compressed "https://api.search.brave.com/res/v1/local/pois?ids=location_id_1&ids=location_id_2" \
  -H "X-Subscription-Token: YOUR_API_KEY"

# Step 3: Get AI descriptions
curl -s --compressed "https://api.search.brave.com/res/v1/local/descriptions?ids=location_id_1&ids=location_id_2" \
  -H "X-Subscription-Token: YOUR_API_KEY"
```

### JavaScript/Node.js Implementation
```javascript
const axios = require('axios');

class BraveSearchAPI {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.baseURL = 'https://api.search.brave.com/res/v1';
  }

  async webSearch(query, options = {}) {
    const params = {
      q: query,
      count: options.count || 20,
      offset: options.offset || 0,
      country: options.country || 'US',
      search_lang: options.searchLang || 'en',
      safesearch: options.safesearch || 'moderate',
      ...options.additionalParams
    };

    const headers = {
      'Accept': 'application/json',
      'Accept-Encoding': 'gzip',
      'X-Subscription-Token': this.apiKey,
      ...options.additionalHeaders
    };

    try {
      const response = await axios.get(`${this.baseURL}/web/search`, {
        params,
        headers
      });
      return response.data;
    } catch (error) {
      throw new Error(`Search failed: ${error.message}`);
    }
  }

  async localPOIs(ids, options = {}) {
    const params = {
      ids: Array.isArray(ids) ? ids : [ids],
      search_lang: options.searchLang || 'en',
      units: options.units
    };

    const headers = {
      'Accept': 'application/json',
      'X-Subscription-Token': this.apiKey
    };

    try {
      const response = await axios.get(`${this.baseURL}/local/pois`, {
        params,
        headers,
        paramsSerializer: {
          indexes: null // This handles multiple ids parameters correctly
        }
      });
      return response.data;
    } catch (error) {
      throw new Error(`Local POI search failed: ${error.message}`);
    }
  }

  async localDescriptions(ids) {
    const params = {
      ids: Array.isArray(ids) ? ids : [ids]
    };

    const headers = {
      'Accept': 'application/json',
      'X-Subscription-Token': this.apiKey
    };

    try {
      const response = await axios.get(`${this.baseURL}/local/descriptions`, {
        params,
        headers,
        paramsSerializer: {
          indexes: null
        }
      });
      return response.data;
    } catch (error) {
      throw new Error(`Local descriptions search failed: ${error.message}`);
    }
  }
}

// Usage example
async function example() {
  const api = new BraveSearchAPI('your_api_key');

  // Basic search
  const results = await api.webSearch('machine learning', {
    count: 10,
    result_filter: 'web,news'
  });

  // Search with location
  const restaurantResults = await api.webSearch('italian restaurants', {
    country: 'US',
    additionalHeaders: {
      'X-Loc-City': 'New York',
      'X-Loc-State': 'NY'
    }
  });

  // Get location details
  if (restaurantResults.locations) {
    const locationIds = restaurantResults.locations.results.map(loc => loc.id);
    const detailedLocations = await api.localPOIs(locationIds);
    const descriptions = await api.localDescriptions(locationIds);
  }
}
```

### Python Implementation
```python
import requests
from typing import List, Dict, Optional

class BraveSearchAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1"

    def _get_headers(self, additional_headers: Dict[str, str] = None) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        if additional_headers:
            headers.update(additional_headers)
        return headers

    def web_search(self, query: str, **kwargs) -> Dict:
        """Perform a web search using Brave Search API"""
        params = {
            "q": query,
            "count": kwargs.get("count", 20),
            "offset": kwargs.get("offset", 0),
            "country": kwargs.get("country", "US"),
            "search_lang": kwargs.get("search_lang", "en"),
            "safesearch": kwargs.get("safesearch", "moderate")
        }

        # Add optional parameters
        optional_params = ["ui_lang", "freshness", "text_decorations",
                          "spellcheck", "result_filter", "goggles", "units"]
        for param in optional_params:
            if param in kwargs:
                params[param] = kwargs[param]

        headers = self._get_headers(kwargs.get("additional_headers", {}))

        response = requests.get(
            f"{self.base_url}/web/search",
            params=params,
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    def local_pois(self, location_ids: List[str], **kwargs) -> Dict:
        """Get detailed location information"""
        params = {"ids": location_ids}

        optional_params = ["search_lang", "ui_lang", "units"]
        for param in optional_params:
            if param in kwargs:
                params[param] = kwargs[param]

        headers = self._get_headers()

        response = requests.get(
            f"{self.base_url}/local/pois",
            params=params,
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    def local_descriptions(self, location_ids: List[str]) -> Dict:
        """Get AI-generated descriptions for locations"""
        params = {"ids": location_ids}
        headers = self._get_headers()

        response = requests.get(
            f"{self.base_url}/local/descriptions",
            params=params,
            headers=headers
        )
        response.raise_for_status()
        return response.json()

# Usage example
def main():
    api = BraveSearchAPI("your_api_key")

    # Basic search
    results = api.web_search(
        "climate change",
        count=15,
        result_filter="web,news,faq"
    )

    # Location-aware search
    restaurant_results = api.web_search(
        "sushi restaurants",
        country="US",
        additional_headers={
            "X-Loc-Lat": "40.7128",
            "X-Loc-Long": "-74.0060",
            "X-Loc-City": "New York",
            "X-Loc-State": "NY"
        }
    )

    # Process location results
    if "locations" in restaurant_results:
        location_ids = [loc["id"] for loc in restaurant_results["locations"]["results"]]

        # Get detailed information
        detailed_locations = api.local_pois(location_ids)
        descriptions = api.local_descriptions(location_ids)

        print(f"Found {len(detailed_locations['results'])} detailed locations")

if __name__ == "__main__":
    main()
```

---

## Best Practices & Optimization

### Query Optimization
1. **Use specific queries**: More specific queries yield better results
2. **Leverage search operators**: Use quotes, site:, filetype: operators
3. **Set appropriate result filters**: Only request needed result types
4. **Use location headers**: Include location data for geo-relevant queries

### Performance Optimization
1. **Enable compression**: Always use `Accept-Encoding: gzip`
2. **Implement caching**: Cache results appropriately based on your use case
3. **Use pagination wisely**: Implement proper pagination with count/offset
4. **Batch local requests**: Use up to 20 location IDs per request

### Location-Based Searches
1. **Include coordinate precision**: Use appropriate precision for coordinates
2. **Set timezone correctly**: Use IANA timezone identifiers
3. **Provide multiple location signals**: City, state, country, postal code
4. **Use local search workflow**: Web search → Extract location IDs → Local POI details

### Response Processing
1. **Handle missing fields**: Not all results contain all possible fields
2. **Check result types**: Use the `type` field to determine result structure
3. **Process mixed responses**: Use the `mixed` object for proper result ordering
4. **Cache location IDs**: Location IDs are valid for 8 hours

---

## Rate Limiting & Error Handling

### Rate Limit Management
Monitor these response headers to manage your rate limits:

```javascript
function checkRateLimit(response) {
  const headers = response.headers;

  const limits = headers['x-ratelimit-limit']?.split(', ');
  const remaining = headers['x-ratelimit-remaining']?.split(', ');
  const reset = headers['x-ratelimit-reset']?.split(', ');

  console.log(`Rate Limits: ${limits}`);
  console.log(`Remaining: ${remaining}`);
  console.log(`Reset in: ${reset} seconds`);

  // Check if approaching limits
  if (remaining && remaining[0] < 5) {
    console.warn('Approaching per-second rate limit');
  }

  if (remaining && remaining[1] < 1000) {
    console.warn('Approaching monthly rate limit');
  }
}
```

### Error Response Handling
```javascript
function handleAPIError(error) {
  if (error.response) {
    const status = error.response.status;
    const data = error.response.data;

    switch (status) {
      case 400:
        console.error('Bad Request:', data.message);
        break;
      case 401:
        console.error('Unauthorized: Check your API key');
        break;
      case 403:
        console.error('Forbidden: Insufficient plan or quota exceeded');
        break;
      case 429:
        console.error('Rate Limited: Too many requests');
        const retryAfter = error.response.headers['retry-after'];
        console.log(`Retry after: ${retryAfter} seconds`);
        break;
      case 500:
        console.error('Server Error: Try again later');
        break;
      default:
        console.error(`HTTP ${status}:`, data);
    }
  } else {
    console.error('Network Error:', error.message);
  }
}
```

### Retry Strategy
```javascript
async function searchWithRetry(api, query, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const results = await api.webSearch(query);
      return results;
    } catch (error) {
      if (error.response?.status === 429) {
        const retryAfter = parseInt(error.response.headers['retry-after'] || '60');
        if (attempt < maxRetries) {
          console.log(`Rate limited. Retrying in ${retryAfter} seconds...`);
          await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
          continue;
        }
      }

      if (attempt === maxRetries) {
        throw error;
      }

      // Exponential backoff for other errors
      const backoffTime = Math.pow(2, attempt) * 1000;
      await new Promise(resolve => setTimeout(resolve, backoffTime));
    }
  }
}
```

---

## Advanced Features

### Goggle Support (Custom Re-ranking)
```bash
# Using a Goggle URL
curl "https://api.search.brave.com/res/v1/web/search?q=search+query&goggles=https://raw.githubusercontent.com/brave/goggles-quickstart/main/goggles/no_pinterest.goggle" \
  -H "X-Subscription-Token: YOUR_API_KEY"

# Using inline Goggle definition
curl "https://api.search.brave.com/res/v1/web/search?q=search+query&goggles=%21site%3Apinterest.com" \
  -H "X-Subscription-Token: YOUR_API_KEY"
```

### Summary Generation
```bash
# Enable summarizer
curl "https://api.search.brave.com/res/v1/web/search?q=climate+change&summary=true" \
  -H "X-Subscription-Token: YOUR_API_KEY"

# The response will include a summarizer key that can be used for further processing
```

### Time-based Filtering
```bash
# Recent results (last 24 hours)
curl "https://api.search.brave.com/res/v1/web/search?q=news&freshness=pd" \
  -H "X-Subscription-Token: YOUR_API_KEY"

# Custom date range
curl "https://api.search.brave.com/res/v1/web/search?q=events&freshness=2024-01-01to2024-01-31" \
  -H "X-Subscription-Token: YOUR_API_KEY"
```

---

This comprehensive guide covers all aspects of the Brave Search API based on the documentation from your notes [[Brave Search - API]], [[Brave Search - API 1]], [[Brave Search - API 2]], [[Brave Search - API 3]], and [[Brave Search - API 4]]. The API provides powerful search capabilities with extensive customization options and rich result types suitable for a wide variety of applications.
