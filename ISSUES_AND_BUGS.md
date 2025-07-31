# Four Hosts Application - Issues and Bug Report

## Overview
This document catalogs all identified issues, bugs, and potential problems in the Four Hosts Research Application codebase, organized by severity and category.

## Critical Security Issues

### 1. JWT Secret Key Vulnerability
- **Location**: `backend/services/auth.py:40-45`
- **Issue**: No validation of JWT_SECRET_KEY strength
- **Risk**: Weak keys could enable token forgery
- **Fix**: Implement key length and entropy validation

### 2. Default Database Credentials
- **Location**: `backend/database/connection.py:30-34`
- **Issue**: Hardcoded fallback credentials "user"/"password"
- **Risk**: Production deployments may inadvertently use weak defaults
```python
pguser = os.getenv("PGUSER", "user")
pgpassword = os.getenv("PGPASSWORD", "password")
```
- **Fix**: Remove defaults, require explicit configuration

### 3. JWT Token Storage in localStorage
- **Location**: `frontend/src/services/api.ts:51`
- **Issue**: Tokens vulnerable to XSS attacks
- **Fix**: Use httpOnly cookies or secure storage

### 4. Empty Error Logging in ErrorBoundary
- **Location**: `frontend/src/components/ErrorBoundary.tsx:24-26`
- **Issue**: Errors not sent to monitoring service
```typescript
componentDidCatch() {
  // Error caught by boundary - log to monitoring service in production
}
```
- **Fix**: Implement proper error logging

## High Priority Issues

### 5. TODO Markers in Production Code
Multiple incomplete implementations:
- `backend/services/export_service.py:845` - "TODO: Fetch research data from database"
- `frontend/src/App.tsx:233` - "TODO: Add UI for search context size"
- `backend/brave-search-mcp-server/src/BraveAPI/index.ts:24,27,36,103` - Multiple TODOs

### 6. Silent Error Suppression
- **Location**: `backend/main.py:2409-2410`
```python
except Exception:
    pass  # Don't let monitoring errors break the response
```
- **Fix**: Log errors before suppressing

### 7. Missing ErrorBoundary Implementation
- **Location**: `frontend/src/App.tsx`
- **Issue**: Component exists but not wrapped around App
- **Risk**: Unhandled errors crash entire application

### 8. SQL Injection Risk (Low)
- **Location**: `backend/database/connection.py:205,208`
```python
await conn.execute(text(f"VACUUM ANALYZE {table_name}"))
```
- **Fix**: Use parameterized queries even for internal operations

## Medium Priority Issues

### 9. Hardcoded Configuration
Multiple hardcoded URLs and values:
- `frontend/test-auth.html:27` - `http://localhost:8000`
- `frontend/vite.config.ts:36,41,46,51,56` - Multiple localhost references
- **Fix**: Use environment variables

### 10. Debug Settings in Production
- **Location**: `backend/main.py:2443`
- **Issue**: Debug logging enabled
```python
log_level="debug"
```

### 11. WebSocket Connection Management
- **Location**: `backend/services/websocket_service.py:86-95`
- **Issues**:
  - No connection timeout
  - No maximum connection limit
  - Message history grows indefinitely

### 12. Missing Input Validation
- **Location**: `backend/database/connection.py:250-272`
- **Issue**: Search query builder lacks input sanitization

### 13. Rate Limiting Implementation
- **Location**: `backend/services/search_apis.py:85-92`
- **Issue**: Basic 1-second delay instead of proper rate limiter
- **Risk**: API quota exhaustion

## Low Priority Issues

### 14. Debug Code in Production
- `backend/test_db.py:11` - Debug environment variable printing
- `backend/database/connection.py:41` - Database URL logging

### 15. Incomplete Test Coverage
- **Location**: `backend/tests/`
- **Issue**: Only 14 test files for complex application
- **Missing**: Integration tests, frontend tests, API contract tests

### 16. Missing Environment Variable Validation
- **Location**: `backend/services/llm_client.py:140`
- **Issue**: Runtime failures instead of startup validation

## Performance Issues

### 17. Single Worker Limitation
- **Location**: `backend/main.py:2433`
```python
workers=1,  # Changed from 4 to 1
```
- **Issue**: Can't scale horizontally without Redis

### 18. Memory Leak Risk
- **Location**: `backend/services/websocket_service.py:94-95`
- **Issue**: Unbounded message history storage

### 19. Inefficient PDF Processing
- **Location**: `backend/services/search_apis.py:29`
- **Issue**: No error handling for large PDF files

### 20. No Connection Pool Limits
- **Location**: `backend/database/connection.py:314-323`
- **Issue**: Pool monitoring without automatic recovery

## Configuration Issues

### 21. Missing SSL Configuration
- No HTTPS enforcement
- No SSL redirect configuration

### 22. Exposed Server Information
- **Location**: `backend/main.py:2437-2438`
- Headers disabled but other info may leak

### 23. No Health Check Endpoints
- **Location**: `backend/database/connection.py:362-393`
- Health check functions exist but not exposed

## Accessibility Issues

### 24. Missing ARIA Labels
- Frontend components lack screen reader support
- No keyboard navigation optimization

### 25. No Loading States
- Some async operations lack loading indicators

## Monitoring Gaps

### 26. Limited Error Context
- **Location**: `backend/main.py:2400-2408`
- Missing user context in error tracking

### 27. No Automated Alerts
- No monitoring for API quota usage
- No alerts for system health issues

## Action Items

### Immediate (Critical/High)
1. [ ] Validate JWT secret key strength
2. [ ] Remove hardcoded database credentials
3. [ ] Implement ErrorBoundary wrapping
4. [ ] Complete TODO implementations
5. [ ] Add error logging to ErrorBoundary

### Short-term (Medium)
1. [ ] Implement proper rate limiting
2. [ ] Add WebSocket connection management
3. [ ] Add comprehensive input validation
4. [ ] Create health check endpoints
5. [ ] Remove debug code

### Long-term (Low)
1. [ ] Implement Redis for scaling
2. [ ] Increase test coverage to 80%+
3. [ ] Add accessibility features
4. [ ] Implement monitoring and alerting
5. [ ] Add SSL/TLS configuration

## Summary

The Four Hosts application has **27 identified issues** across various categories:
- **4 Critical** security vulnerabilities
- **8 High priority** bugs and TODOs
- **5 Medium priority** issues
- **10 Low priority** improvements

The most urgent items requiring immediate attention are:
1. JWT security vulnerabilities
2. Hardcoded credentials
3. Incomplete error handling
4. Missing ErrorBoundary implementation

These issues should be addressed before any production deployment.