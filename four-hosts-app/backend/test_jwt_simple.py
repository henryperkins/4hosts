#!/usr/bin/env python3
"""
Simple JWT test to isolate the issue
"""

import os
import sys
import jwt
from pathlib import Path

# Load env vars
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, _, value = line.partition('=')
                if key and value:
                    os.environ[key.strip()] = value.strip()

# Test JWT directly
jwt_secret = os.getenv("JWT_SECRET_KEY")
print(f"JWT_SECRET_KEY: {jwt_secret[:10]}... (length: {len(jwt_secret)})")

# Create a test token
test_payload = {
    "user_id": "test-123",
    "email": "test@example.com",
    "role": "free"
}

algorithm = "HS256"

# Generate token
token = jwt.encode(test_payload, jwt_secret, algorithm=algorithm)
print(f"Generated token: {token[:50]}...")

# Try to decode it
try:
    decoded = jwt.decode(token, jwt_secret, algorithms=[algorithm])
    print(f"✓ Token decoded successfully: {decoded}")
except jwt.InvalidSignatureError as e:
    print(f"✗ Signature verification failed: {e}")
except Exception as e:
    print(f"✗ Error: {e}")

# Now test with actual auth module
print("\n--- Testing with auth module ---")
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import just the constants first
    from core.config import ALGORITHM
    print(f"Algorithm from config: {ALGORITHM}")

    # Now import the secret key
    from services.auth.tokens import SECRET_KEY
    print(f"SECRET_KEY from auth.tokens: {SECRET_KEY[:10]}... (length: {len(SECRET_KEY)})")
    print(f"Keys match: {SECRET_KEY == jwt_secret}")

    # Try to decode with auth module's key
    decoded2 = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    print(f"✓ Token decoded with auth module key: {decoded2}")

except Exception as e:
    print(f"✗ Error with auth module: {e}")
    import traceback
    traceback.print_exc()