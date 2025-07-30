# Ethical Content Access Improvements for Four Hosts Research System

This document outlines the ethical and legal approaches implemented to handle content access restrictions in the Four Hosts research system.

## Key Improvements Made

### 1. **Proper User-Agent Headers**
- Added a descriptive User-Agent that identifies the bot and provides contact information
- Headers now include standard browser headers to ensure compatibility
- Implementation in `fetch_and_parse_url()` function

### 2. **Robots.txt Compliance**
- Created `RespectfulFetcher` class that checks robots.txt before fetching any URL
- Caches robots.txt parsers to minimize requests
- Falls back gracefully if robots.txt cannot be fetched

### 3. **Rate Limiting**
- Implemented per-domain rate limiting (1 second between requests to same domain)
- Prevents overwhelming servers with rapid requests
- Tracks last fetch time per domain

### 4. **Enhanced Error Handling**
- Specific handling for 403 (Forbidden) responses - logs and uses snippets instead
- Specific handling for 429 (Rate Limited) responses - warns about rate limiting
- Timeout handling for slow or blocking sites

### 5. **Content Fallback Strategy**
The `fetch_with_fallback()` method implements a tiered approach:
1. **Direct Fetch**: Try respectful fetching with proper headers and robots.txt compliance
2. **Academic APIs**: Extract DOIs and arXiv IDs to use official APIs
3. **Search Snippets**: Use search result snippets as fallback content

### 6. **New Free Academic APIs**
Added two new academic search APIs that provide free, legal access:
- **Semantic Scholar API**: Excellent for academic papers, no API key required
- **CrossRef API**: For DOI metadata and open access papers

### 7. **Enhanced Search Result Processing**
- `enhance_results_with_snippets()` method uses search snippets when full content unavailable
- Marks content source (direct_fetch, snippet_only, etc.) in result metadata
- Preserves transparency about content origin

## Ethical Principles Followed

1. **Respect robots.txt**: Always check and comply with robots.txt directives
2. **Identify the bot**: Use clear User-Agent with contact information
3. **Rate limit requests**: Avoid overwhelming servers
4. **Use official APIs**: Prefer official APIs over web scraping
5. **Graceful degradation**: Use snippets when full content unavailable
6. **Transparency**: Mark content source and type clearly

## Usage Examples

```python
# The system now automatically:
# 1. Checks robots.txt before fetching
# 2. Uses proper headers
# 3. Rate limits requests
# 4. Falls back to snippets if access denied

# Results include metadata about content source:
result.raw_data['content_source'] # 'direct_fetch', 'snippet_only', 'arxiv_api', etc.
result.raw_data['content_type'] # 'full_content', 'snippet_only'
```

## Benefits

1. **Legal Compliance**: Respects website terms of service and robots.txt
2. **Better Reliability**: Multiple fallback options ensure some content is always available
3. **Server Friendly**: Rate limiting and proper identification reduce server load
4. **Transparency**: Users know the source and type of content they're getting
5. **Academic Focus**: Enhanced support for academic content through dedicated APIs

## Future Considerations

For production use, consider:
- Academic institutional subscriptions
- Business agreements with content providers
- Integration with Unpaywall API for legal open-access versions
- Content partnerships with major publishers
- OAuth integration for user-authenticated access

These improvements ensure the Four Hosts research system operates ethically while maximizing access to legitimate content sources.