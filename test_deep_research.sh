#!/bin/bash

# Test the deep research endpoint

echo "=== Testing Deep Research Endpoint ==="

# Test with a valid request (replace YOUR_TOKEN with an actual token)
echo -e "\n1. Valid request:"
curl -X POST "http://localhost:8000/research/deep" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "What are the latest breakthroughs in quantum computing research?",
    "paradigm": "bernard",
    "search_context_size": "medium"
  }' | jq .

echo -e "\n2. Missing query field (should return 422):"
curl -X POST "http://localhost:8000/research/deep" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "paradigm": "bernard"
  }' | jq .

echo -e "\n3. Query too short (should return 422):"
curl -X POST "http://localhost:8000/research/deep" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "AI?",
    "paradigm": "bernard"
  }' | jq .

echo -e "\nNote: Replace YOUR_TOKEN with an actual authentication token"
echo "To get a token, first register/login a user with PRO role"