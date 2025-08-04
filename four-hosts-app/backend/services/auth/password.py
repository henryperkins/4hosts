"""
Password utilities for authentication
"""

import re
import bcrypt
from utils.async_utils import run_in_thread


async def hash_password(password: str) -> str:
    """Hash password using bcrypt in a worker thread."""
    return await run_in_thread(
        lambda: bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode(
            "utf-8"
        )
    )


async def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain_password against hashed_password without blocking the loop."""
    return await run_in_thread(
        lambda: bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    )


def validate_password_strength(password: str) -> bool:
    """Validate password meets security requirements"""
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    return True
