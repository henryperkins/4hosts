"""Centralized security utilities for input validation and sanitization.

This module provides reusable security functions to prevent injection attacks,
validate inputs, and sanitize user data across all services.
"""

import re
import html
import bleach
import hashlib
import secrets
from typing import Optional, List, Pattern
import logging
from .url_utils import MAX_URL_LENGTH

logger = logging.getLogger(__name__)

# Configuration constants
MAX_QUERY_LENGTH = 5000
MAX_API_KEY_LENGTH = 128
MAX_TOKEN_LENGTH = 2048

# Validation patterns
IP_PATTERN = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
FORWARDED_PATTERN = re.compile(r'^[0-9\., ]+$')
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Dangerous pattern detection for ReDoS prevention
REDOS_PATTERNS = [
    re.compile(r'(\+|\*|\{[0-9,]+\})\s*[^\)]*?(\+|\*|\{[0-9,]+\})'),  # Nested quantifiers
    re.compile(r'\(\?P<'),  # Named groups that can be exploited
    re.compile(r'\\[0-9]{4,}'),  # Excessive backreferences
]


def validate_and_sanitize_url(url: str) -> str:
    """Validate and sanitize a URL.

    Args:
        url: The URL string to validate and sanitize

    Returns:
        The sanitized URL string
    """
    if not url:
        return ""

    # Remove any dangerous characters and truncate to max length
    url = url.strip()[:MAX_URL_LENGTH]

    # Basic URL validation - ensure it starts with http/https
    if not url.startswith(('http://', 'https://')):
        return ""

    # Remove any control characters or non-printable characters
    url = ''.join(char for char in url if ord(char) >= 32)

    return url


class InputSanitizer:
    """Centralized input sanitization to prevent injection attacks."""

    def __init__(self):
        self.allowed_tags = []
        self.allowed_attributes = {}
        self.allowed_protocols = []

    def sanitize_text(self, text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
        """Sanitize plain text input, removing HTML and normalizing whitespace."""
        if not text:
            return ""

        # Truncate to reasonable length
        text = text[:max_length]

        # Clean HTML/script tags using bleach
        text = bleach.clean(
            text,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            protocols=self.allowed_protocols,
            strip=True,
            strip_comments=True
        )

        # Escape HTML entities for additional safety
        text = html.escape(text)

        # Normalize whitespace
        text = " ".join(text.split())

        return text


    def sanitize_sql_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifiers (table names, column names)."""
        # Only allow alphanumeric and underscore
        return re.sub(r'[^a-zA-Z0-9_]', '', identifier)[:64]

    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        if not email or len(email) > 254:
            return False
        return bool(EMAIL_PATTERN.match(email))


class PatternValidator:
    """Validate regex patterns to prevent ReDoS attacks."""

    @staticmethod
    def is_safe_pattern(pattern: str, max_length: int = 200) -> bool:
        """Check if a regex pattern is safe from ReDoS attacks."""
        if not pattern or len(pattern) > max_length:
            return False

        # Check for dangerous patterns
        for dangerous in REDOS_PATTERNS:
            if dangerous.search(pattern):
                logger.warning(f"Potentially dangerous regex pattern detected: {pattern[:50]}...")
                return False

        # Try to compile with timeout (would need subprocess for true timeout)
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    @staticmethod
    def safe_compile(pattern: str, flags: int = re.IGNORECASE) -> Optional[Pattern]:
        """Safely compile a regex pattern with validation."""
        if not PatternValidator.is_safe_pattern(pattern):
            return None

        try:
            return re.compile(pattern, flags)
        except re.error as e:
            logger.error(f"Failed to compile pattern: {e}")
            return None


class IPValidator:
    """Validate and extract IP addresses from request headers."""

    @staticmethod
    def validate_ip(ip: str) -> bool:
        """Check if string is a valid IPv4 address."""
        if not ip or len(ip) > 45:  # Max IPv6 length
            return False
        return bool(IP_PATTERN.match(ip))

    @staticmethod
    def extract_from_forwarded(forwarded: str) -> Optional[str]:
        """Safely extract IP from X-Forwarded-For header."""
        if not forwarded or len(forwarded) > 256:
            return None

        # Validate format to prevent header injection
        if not FORWARDED_PATTERN.match(forwarded):
            return None

        # Take first IP in chain
        first_ip = forwarded.split(',')[0].strip()

        if IPValidator.validate_ip(first_ip):
            return first_ip

        return None


class APIKeyManager:
    """Manage API key generation and indexing."""

    @staticmethod
    def generate_api_key(prefix: str = "fh", length: int = 32) -> str:
        """Generate a secure API key with prefix."""
        return f"{prefix}_{secrets.token_urlsafe(length)}"

    @staticmethod
    def compute_key_index(api_key: str) -> Optional[str]:
        """Compute a searchable index for API key lookups."""
        if not api_key or not api_key.startswith("fh_"):
            return None

        # Use first 12 chars after prefix for index
        if len(api_key) < 15:
            return None

        key_portion = api_key[3:15]
        return hashlib.sha256(key_portion.encode()).hexdigest()[:16]

    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """Validate API key format and length."""
        if not api_key or not api_key.startswith("fh_"):
            return False

        if len(api_key) > MAX_API_KEY_LENGTH:
            return False

        # Check for valid base64url characters after prefix
        key_part = api_key[3:]
        if not re.match(r'^[A-Za-z0-9_-]+$', key_part):
            return False

        return True


class TokenValidator:
    """Validate JWT tokens and extract claims safely."""

    @staticmethod
    def validate_bearer_header(auth_header: str) -> Optional[str]:
        """Extract and validate token from Authorization header."""
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        parts = auth_header.split(" ", 1)
        if len(parts) != 2:
            return None

        token = parts[1].strip()

        if not token or len(token) > MAX_TOKEN_LENGTH:
            return None

        # Basic JWT format validation (three base64url parts)
        if token.count('.') != 2:
            return None

        return token


# Singleton instances for convenience
input_sanitizer = InputSanitizer()
pattern_validator = PatternValidator()
ip_validator = IPValidator()
api_key_manager = APIKeyManager()
token_validator = TokenValidator()


# Convenience functions
def sanitize_user_input(text: str) -> str:
    """Quick sanitization for user text input."""
    return input_sanitizer.sanitize_text(text)




def is_valid_email(email: str) -> bool:
    """Check if email is valid."""
    return input_sanitizer.validate_email(email)


def safe_regex_compile(pattern: str) -> Optional[Pattern]:
    """Safely compile regex pattern."""
    return pattern_validator.safe_compile(pattern)