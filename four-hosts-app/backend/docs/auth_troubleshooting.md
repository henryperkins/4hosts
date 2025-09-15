# Authentication Troubleshooting Guide

This guide helps diagnose and resolve common authentication issues in the Four Hosts Research API.

## Quick Diagnostic Steps

### 1. Run the Authentication Diagnostic Script
```bash
cd four-hosts-app/backend
python3 test_auth_diagnostic.py
```

### 2. Manual CSRF + Login Test
```bash
# Get CSRF token and store cookies
curl -i -c /tmp/auth-cookies.txt http://localhost:8000/api/csrf-token

# Extract CSRF token
CSRF_TOKEN=$(grep csrf_token /tmp/auth-cookies.txt | awk '{print $7}')

# Login with proper CSRF headers
curl -i -b /tmp/auth-cookies.txt -c /tmp/auth-cookies.txt \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $CSRF_TOKEN" \
  -X POST http://localhost:8000/v1/auth/login \
  -d '{"email": "test@example.com", "password": "password123"}'

# Test authenticated endpoint
curl -i -b /tmp/auth-cookies.txt http://localhost:8000/v1/auth/user
```

### 3. Check Auth Debug Status (Development Only)
```bash
# Get current auth state
curl -i -b /tmp/auth-cookies.txt \
  -H "X-CSRF-Token: $CSRF_TOKEN" \
  http://localhost:8000/v1/auth/debug/status
```

## Common Issues and Solutions

### 401 Unauthorized on Login

#### **Issue: CSRF Token Mismatch**
- **Symptoms**: 403 Forbidden with "CSRF token mismatch" message
- **Cause**: Missing or incorrect CSRF token in request
- **Solution**:
  1. Always get CSRF token first: `GET /api/csrf-token`
  2. Include token in both cookie AND header: `X-CSRF-Token: <token>`
  3. Use the same cookie jar for all requests

#### **Issue: Invalid Credentials**
- **Symptoms**: 401 with "Invalid credentials" message
- **Causes**:
  - User doesn't exist in database
  - Wrong password
  - User account is inactive
- **Solution**:
  1. Check if user exists: `python3 list_users.py`
  2. Create test user: `python3 test_auth_diagnostic.py`
  3. Verify password hash in database

#### **Issue: Missing Authentication Token**
- **Symptoms**: 401 with "Missing authentication token"
- **Cause**: No access token in cookies or Authorization header
- **Solution**:
  1. Ensure login response sets cookies properly
  2. Check cookie security flags (Secure over HTTP)
  3. Verify curl uses `-b cookies.txt` for subsequent requests

### Cookie Issues

#### **Issue: Cookies Not Set After Login**
- **Symptoms**: Login returns 200 but no Set-Cookie headers
- **Causes**:
  - Secure flag set while using HTTP
  - SameSite restrictions
  - Path mismatches
- **Solutions**:
  1. Use HTTPS for production
  2. Set `COOKIE_SECURE=false` for development
  3. Check cookie attributes in response headers

#### **Issue: httpOnly Cookies in Browser**
- **Symptoms**: JavaScript can't read access tokens
- **Cause**: This is by design for security
- **Solution**: Use cookie-based authentication, not header-based

### Route Path Confusion

#### **Issue: 404 Not Found on Auth Endpoints**
- **Symptoms**: Auth endpoints return 404
- **Causes**:
  - Using wrong path (`/auth/login` vs `/v1/auth/login`)
  - Server not mounting routes correctly
- **Solutions**:
  1. Use correct versioned paths: `/v1/auth/*`
  2. Check FastAPI route mounting in `core/app.py`

### Development vs Production

#### **Issue: Different Behavior in Production**
- **Symptoms**: Auth works locally but fails in production
- **Causes**:
  - HTTPS enforcement
  - Different environment variables
  - Proxy headers not forwarded
- **Solutions**:
  1. Check `x-forwarded-proto` headers
  2. Verify SSL certificate configuration
  3. Ensure environment variables are set

## Environment Variables

### Required for Authentication
```bash
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Development-Specific
```bash
ENVIRONMENT=development
COOKIE_SECURE=false  # Allow insecure cookies over HTTP
```

### Production-Specific
```bash
ENVIRONMENT=production
COOKIE_SECURE=true   # Enforce secure cookies
TRUSTED_ORIGINS=https://yourdomain.com
```

## Debugging Tools

### 1. Enhanced Logging
The system now includes structured logging for auth failures:
- Login attempts with request IDs
- Specific failure reasons (user not found, password mismatch)
- CSRF validation failures

### 2. Debug Endpoint (Development Only)
`GET /v1/auth/debug/status` provides:
- Cookie presence check
- CSRF token validation
- HTTPS detection status
- Request correlation info

### 3. Health Checks
- `GET /health` - Basic system health
- `GET /ready` - Readiness including auth service

## Security Best Practices

### 1. CSRF Protection
- ✅ All state-changing operations require CSRF tokens
- ✅ Tokens are bound to sessions
- ✅ Double-submit cookie pattern implemented

### 2. Token Security
- ✅ Access tokens are short-lived (30 minutes)
- ✅ Refresh tokens use secure rotation
- ✅ httpOnly cookies prevent XSS theft
- ✅ Secure flags enforce HTTPS in production

### 3. Input Validation
- ✅ Email format validation
- ✅ Password strength requirements
- ✅ SQL injection prevention via ORM
- ✅ Request size limits

## Log Analysis

### Finding Auth Failures
```bash
# Search for login failures
grep "Login failed" backend.log

# Search for CSRF mismatches
grep "CSRF token mismatch" backend.log

# Search by request ID
grep "req_id: abc123" backend.log
```

### Understanding Log Entries
```
[INFO] Login attempt for email: user@example.com [req_id: abc123]
[WARNING] Login failed for user@example.com: user_not_found_or_invalid_password [req_id: abc123]
[WARNING] CSRF token mismatch on /v1/auth/login: cookie=token1, header=token2
```

## Getting Help

If you're still experiencing issues:

1. **Run the diagnostic script**: `python3 test_auth_diagnostic.py`
2. **Check server logs** for specific error messages
3. **Verify environment variables** are set correctly
4. **Test with curl** using the examples in this guide
5. **Check browser network tab** for cookie and header details

Remember: Authentication is a multi-step process involving CSRF tokens, cookies, and proper HTTP headers. Each step must work correctly for the full flow to succeed.
