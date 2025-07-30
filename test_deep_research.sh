#!/bin/bash

# Test the deep research endpoint

echo "=== Testing Deep Research Endpoint ==="

# Test with a valid request (replace eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZmM5YjFlNDktOWQ3Yi00OTNmLTlhNDItOWM3YTk5NGFhNzZhIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwicm9sZSI6InBybyIsImV4cCI6MTc1Mzg4NjYyMSwiaWF0IjoxNzUzODg0ODIxLCJqdGkiOiJTV2NEZmtoR1BPMnYwX3VJa1hPNkhnIn0.aOuTq11jLMnlrEB6VjppHy0bismmtZPv6JP0qKePIEM with an actual token)
echo -e "\n1. Valid request:"
curl -X POST "http://localhost:8000/research/deep" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZmM5YjFlNDktOWQ3Yi00OTNmLTlhNDItOWM3YTk5NGFhNzZhIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwicm9sZSI6InBybyIsImV4cCI6MTc1Mzg4NjYyMSwiaWF0IjoxNzUzODg0ODIxLCJqdGkiOiJTV2NEZmtoR1BPMnYwX3VJa1hPNkhnIn0.aOuTq11jLMnlrEB6VjppHy0bismmtZPv6JP0qKePIEM" \
  -d '{
    "query": "What are the latest breakthroughs in quantum computing research?",
    "paradigm": "bernard",
    "search_context_size": "medium"
  }' | jq .

echo -e "\n2. Missing query field (should return 422):"
curl -X POST "http://localhost:8000/research/deep" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZmM5YjFlNDktOWQ3Yi00OTNmLTlhNDItOWM3YTk5NGFhNzZhIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwicm9sZSI6InBybyIsImV4cCI6MTc1Mzg4NjYyMSwiaWF0IjoxNzUzODg0ODIxLCJqdGkiOiJTV2NEZmtoR1BPMnYwX3VJa1hPNkhnIn0.aOuTq11jLMnlrEB6VjppHy0bismmtZPv6JP0qKePIEM" \
  -d '{
    "paradigm": "bernard"
  }' | jq .

echo -e "\n3. Query too short (should return 422):"
curl -X POST "http://localhost:8000/research/deep" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZmM5YjFlNDktOWQ3Yi00OTNmLTlhNDItOWM3YTk5NGFhNzZhIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwicm9sZSI6InBybyIsImV4cCI6MTc1Mzg4NjYyMSwiaWF0IjoxNzUzODg0ODIxLCJqdGkiOiJTV2NEZmtoR1BPMnYwX3VJa1hPNkhnIn0.aOuTq11jLMnlrEB6VjppHy0bismmtZPv6JP0qKePIEM" \
  -d '{
    "query": "AI?",
    "paradigm": "bernard"
  }' | jq .

echo -e "\nNote: Replace eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZmM5YjFlNDktOWQ3Yi00OTNmLTlhNDItOWM3YTk5NGFhNzZhIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwicm9sZSI6InBybyIsImV4cCI6MTc1Mzg4NjYyMSwiaWF0IjoxNzUzODg0ODIxLCJqdGkiOiJTV2NEZmtoR1BPMnYwX3VJa1hPNkhnIn0.aOuTq11jLMnlrEB6VjppHy0bismmtZPv6JP0qKePIEM with an actual authentication token"
echo "To get a token, first register/login a user with PRO role"