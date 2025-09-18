# Four Hosts - Localhost Access Guide

This guide helps you troubleshoot authentication, CORS, and localhost access issues.

## Quick Start for Localhost Development

### 1. Backend Setup
```bash
cd four-hosts-app/backend
source venv/bin/activate  # or create venv: python -m venv venv
pip install -r requirements.txt
python main.py
```
The backend will run on `http://localhost:8000`

### 2. Frontend Setup
```bash
cd four-hosts-app/frontend
npm install
npm run dev
```
The frontend will run on `http://localhost:5173`

## Common Issues and Solutions

### CORS Errors
**Problem:** "CORS policy: The value of the 'Access-Control-Allow-Credentials' header is 'true' which cannot be used with origin '*'"

**Solution:** Fixed in the codebase - the wildcard origin has been removed from CORS configuration.

### Authentication Failures
**Problem:** "Cannot connect to backend server"

**Solutions:**
1. Ensure backend is running on port 8000
2. Check that the backend `.env` file has `JWT_SECRET_KEY` set
3. Verify PostgreSQL database is accessible with credentials in `.env`

### Frontend API Configuration
The frontend uses Vite's proxy feature for development:
- API calls to `/auth/*`, `/research/*`, etc. are proxied to `http://localhost:8000`
- No need to set `VITE_API_URL` for localhost development
- WebSocket connections are also proxied

### Testing Authentication Flow

1. **Register a new user:**
   - Navigate to `http://localhost:5173`
   - Click "Get Started" or "Sign Up"
   - Fill in username, email, and password
   - Password requirements: 8+ chars, uppercase, lowercase, number, special char

2. **Login:**
   - Use the email and password from registration
   - Token is stored in localStorage as `auth_token`

3. **Verify authentication:**
   - Open browser DevTools > Application > Local Storage
   - Look for `auth_token` key
   - Network tab should show Authorization header in API requests

## Environment Variables

### Backend (.env)
```env
JWT_SECRET_KEY=<your-secret-key>  # Required for authentication
PGHOST=<database-host>
PGUSER=<database-user>
PGPASSWORD=<database-password>
PGDATABASE=fourhosts
```

### Frontend (.env)
```env
VITE_API_URL=  # Leave empty for localhost development
```

## Debugging Tips

1. **Check backend logs:**
   - Authentication errors appear in terminal running `python main.py`
   - Look for JWT validation errors or database connection issues

2. **Browser DevTools:**
   - Network tab: Check for 401/403 errors
   - Console: Look for CORS or connection errors
   - Application > Local Storage: Verify auth_token exists

3. **Test backend directly:**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Test registration
   curl -X POST http://localhost:8000/auth/register \
     -H "Content-Type: application/json" \
     -d '{"username":"test","email":"test@example.com","password":"Test123!@#"}'
   ```

## Production Deployment Notes

For production deployment:
1. Set `VITE_API_URL` to your backend URL (e.g., `https://api.4hosts.ai`)
2. Update CORS allowed origins in `main.py` to include your production domain
3. Use environment-specific `.env` files
4. Enable HTTPS for both frontend and backend

## Additional Resources

- [FastAPI CORS Documentation](https://fastapi.tiangolo.com/tutorial/cors/)
- [Vite Proxy Configuration](https://vitejs.dev/config/server-options.html#server-proxy)
- [JWT Authentication Best Practices](https://jwt.io/introduction/)