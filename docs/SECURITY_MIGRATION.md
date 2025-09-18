# Security Migration Guide

## Phase 1: Backend Changes Required (Week 1)

### 1.1 Cookie-Based Authentication
```python
# backend/main.py - Add to login endpoint
@app.post("/auth/login")
async def login(credentials: LoginCredentials, response: Response):
    # ... validate credentials ...
    
    # Set httpOnly cookies instead of returning tokens
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=3600  # 1 hour
    )
    
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=604800  # 7 days
    )
    
    return {"success": True, "user": user_data}
```

### 1.2 CSRF Token Endpoint
```python
# backend/main.py - Add CSRF token generation
from secrets import token_urlsafe

@app.get("/api/csrf-token")
async def get_csrf_token(request: Request):
    token = token_urlsafe(32)
    # Store token in session or cache with user identifier
    return {"token": token}

# Add CSRF validation middleware
@app.middleware("http")
async def validate_csrf(request: Request, call_next):
    if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
        csrf_token = request.headers.get("X-CSRF-Token")
        # Validate token against stored value
        if not validate_csrf_token(csrf_token, request):
            return JSONResponse(status_code=403, content={"error": "Invalid CSRF token"})
    
    response = await call_next(request)
    return response
```

## Phase 2: Frontend Migration (Week 2)

### 2.1 Replace localStorage Usage
```typescript
// OLD - api-auth.ts
localStorage.setItem('auth_token', accessToken)

// NEW - Use secure API client
import { SecureAPIClient } from './secure-api-client'
const api = new SecureAPIClient(API_BASE_URL)
// Tokens handled automatically via cookies
```

### 2.2 Update API Calls
```typescript
// OLD - api.ts
const response = await fetch('/api/research', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
})

// NEW - secure-api-client.ts
const data = await api.post('/research', researchData)
// CSRF and auth handled automatically
```

### 2.3 Replace WebSocket Connections
```typescript
// OLD - Direct WebSocket usage
const ws = new WebSocket(wsUrl)

// NEW - Use hook with cleanup
import { useWebSocket } from '../hooks/useWebSocket'

function ResearchComponent({ researchId }) {
  const { disconnect } = useWebSocket({
    researchId,
    onMessage: handleMessage
  })
  
  // Cleanup happens automatically
}
```

## Phase 3: Testing & Validation (Week 3)

### 3.1 Security Tests
```typescript
// tests/security.test.ts
describe('Security', () => {
  it('should not expose tokens in localStorage', () => {
    expect(localStorage.getItem('auth_token')).toBeNull()
    expect(localStorage.getItem('refresh_token')).toBeNull()
  })
  
  it('should include CSRF token in requests', async () => {
    const mockFetch = jest.spyOn(global, 'fetch')
    await api.post('/test', {})
    
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-CSRF-Token': expect.any(String)
        })
      })
    )
  })
})
```

### 3.2 Memory Leak Tests
```typescript
describe('WebSocket cleanup', () => {
  it('should close connections on unmount', () => {
    const { unmount } = renderHook(() => 
      useWebSocket({ researchId: 'test', onMessage: jest.fn() })
    )
    
    // Verify connection is open
    expect(WebSocket.prototype.close).not.toHaveBeenCalled()
    
    unmount()
    
    // Verify connection is closed
    expect(WebSocket.prototype.close).toHaveBeenCalled()
  })
})
```

## Phase 4: Deployment (Week 4)

### 4.1 Update nginx.conf
```nginx
# Remove unsafe-inline and unsafe-eval
add_header Content-Security-Policy "
  default-src 'self';
  script-src 'self';
  style-src 'self';
  img-src 'self' data: https:;
  font-src 'self';
  connect-src 'self' ws: wss:;
  frame-ancestors 'none';
  base-uri 'self';
  form-action 'self';
  upgrade-insecure-requests;
" always;

# Add security headers
add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
```

### 4.2 Environment Variables
```bash
# .env.production
VITE_ENABLE_CSRF=true
VITE_SECURE_COOKIES=true
VITE_SESSION_TIMEOUT=3600000
```

## Rollback Plan

If issues arise during migration:

1. **Feature flags**: Use environment variables to toggle between old/new auth
2. **Gradual rollout**: Deploy to staging first, then % of production users
3. **Monitoring**: Track auth failures, CSRF errors, WebSocket disconnections
4. **Quick revert**: Keep old auth code commented but available for 30 days

## Success Metrics

- [ ] Zero tokens in localStorage
- [ ] All state-changing requests include CSRF token
- [ ] WebSocket connections properly cleaned up
- [ ] No memory leaks in production monitoring
- [ ] Security headers score A+ on securityheaders.com