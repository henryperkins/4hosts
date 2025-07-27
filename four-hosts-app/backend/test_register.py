import requests
import json

url = "http://localhost:8000/auth/register"
data = {
    "email": "test@example.com",
    "username": "testuser",
    "password": "Test123!Pass",
    "role": "free"
}

response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")