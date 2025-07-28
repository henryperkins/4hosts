#!/bin/bash

echo "Four Hosts Frontend-Backend Integration Test"
echo "==========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if backend is running
echo -e "\n${YELLOW}Checking backend health...${NC}"
BACKEND_HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Backend is running${NC}"
    echo "Response: $BACKEND_HEALTH" | jq . 2>/dev/null || echo "$BACKEND_HEALTH"
else
    echo -e "${RED}✗ Backend is not running on http://localhost:8000${NC}"
    echo "Please start the backend first with: cd backend && python3 main.py"
    exit 1
fi

# Check API endpoints
echo -e "\n${YELLOW}Testing API endpoints...${NC}"

# Test root endpoint
echo -e "\n1. Testing root endpoint (/)"
ROOT_RESPONSE=$(curl -s http://localhost:8000/)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Root endpoint accessible${NC}"
else
    echo -e "${RED}✗ Root endpoint failed${NC}"
fi

# Test auth endpoints (without credentials)
echo -e "\n2. Testing auth endpoints"
AUTH_TEST=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/auth/user)
if [ "$AUTH_TEST" = "401" ]; then
    echo -e "${GREEN}✓ Auth endpoint returns 401 (as expected without token)${NC}"
else
    echo -e "${RED}✗ Auth endpoint returned unexpected status: $AUTH_TEST${NC}"
fi

# Check frontend dependencies
echo -e "\n${YELLOW}Checking frontend setup...${NC}"
cd four-hosts-app/frontend

if [ -d "node_modules" ]; then
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${RED}✗ Frontend dependencies not installed${NC}"
    echo "Run: cd frontend && npm install"
fi

# Check if frontend build exists
if [ -d "dist" ]; then
    echo -e "${GREEN}✓ Frontend build exists${NC}"
else
    echo -e "${YELLOW}! Frontend not built (run: npm run build)${NC}"
fi

echo -e "\n${YELLOW}Summary:${NC}"
echo "1. Backend API is accessible at http://localhost:8000"
echo "2. Frontend dev server should run on http://localhost:5173"
echo "3. Frontend proxies /api requests to backend"
echo ""
echo "To start the full application:"
echo "  Terminal 1: cd backend && python3 main.py"
echo "  Terminal 2: cd frontend && npm run dev"
echo ""
echo "Then access the application at http://localhost:5173"