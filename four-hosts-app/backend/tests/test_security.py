"""
Comprehensive test suite for security utilities and patches.

Tests input sanitization, API key validation, rate limiting,
circuit breakers, and other security measures.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from utils.security import (
    InputSanitizer,
    PatternValidator,
    IPValidator,
    APIKeyManager,
    TokenValidator,
    sanitize_user_input
)
from utils.circuit_breaker import CircuitBreaker, CircuitOpenError


class TestInputSanitizer:
    """Test input sanitization functionality."""

    def setup_method(self):
        self.sanitizer = InputSanitizer()

    def test_sanitize_html_tags(self):
        """Test HTML tag removal."""
        malicious = '<script>alert("XSS")</script>Hello'
        clean = self.sanitizer.sanitize_text(malicious)
        assert '<script>' not in clean
        assert 'alert' not in clean
        assert 'Hello' in clean

    def test_sanitize_sql_injection(self):
        """Test SQL injection prevention."""
        malicious = "'; DROP TABLE users; --"
        clean = self.sanitizer.sanitize_text(malicious)
        assert "DROP TABLE" not in clean
        # HTML entities should be escaped
        assert "&" in clean or ";" not in clean

    def test_max_length_truncation(self):
        """Test input length limits."""
        long_input = "a" * 10000
        clean = self.sanitizer.sanitize_text(long_input, max_length=100)
        assert len(clean) <= 100

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        messy = "  Hello   \n\t  World  \r\n  "
        clean = self.sanitizer.sanitize_text(messy)
        assert clean == "Hello World"

    def test_url_validation(self):
        """Test URL validation and sanitization."""
        valid_url = "https://example.com/path"
        invalid_url = "javascript:alert('xss')"

        assert self.sanitizer.sanitize_url(valid_url) == valid_url
        assert self.sanitizer.sanitize_url(invalid_url) is None

    def test_sql_identifier_sanitization(self):
        """Test SQL identifier sanitization."""
        dangerous = "users; DROP TABLE--"
        safe = self.sanitizer.sanitize_sql_identifier(dangerous)
        assert ";" not in safe
        assert "DROP" not in safe
        assert safe == "usersDROPTABLE"


class TestPatternValidator:
    """Test regex pattern validation for ReDoS prevention."""

    def test_safe_patterns(self):
        """Test acceptance of safe patterns."""
        safe_patterns = [
            r"hello world",
            r"\d{3}-\d{4}",
            r"[a-zA-Z]+",
            r"^start.*end$"
        ]
        for pattern in safe_patterns:
            assert PatternValidator.is_safe_pattern(pattern)

    def test_dangerous_patterns(self):
        """Test rejection of ReDoS patterns."""
        dangerous_patterns = [
            r"(a+)+b",  # Nested quantifiers
            r"(x*)*y",  # Catastrophic backtracking
            r"(?P<name>.*)",  # Named groups (potential exploit)
            r"a" * 500  # Excessive length
        ]
        for pattern in dangerous_patterns:
            assert not PatternValidator.is_safe_pattern(pattern)

    def test_safe_compile(self):
        """Test safe pattern compilation."""
        safe = PatternValidator.safe_compile(r"\d+")
        assert safe is not None

        dangerous = PatternValidator.safe_compile(r"(a+)+b")
        assert dangerous is None


class TestIPValidator:
    """Test IP address validation."""

    def test_valid_ipv4(self):
        """Test valid IPv4 addresses."""
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "8.8.8.8",
            "127.0.0.1"
        ]
        for ip in valid_ips:
            assert IPValidator.validate_ip(ip)

    def test_invalid_ips(self):
        """Test invalid IP addresses."""
        invalid_ips = [
            "256.256.256.256",
            "192.168.1",
            "not.an.ip.address",
            "::1",  # IPv6 not supported yet
            ""
        ]
        for ip in invalid_ips:
            assert not IPValidator.validate_ip(ip)

    def test_extract_from_forwarded(self):
        """Test IP extraction from X-Forwarded-For header."""
        # Valid forwarded header
        forwarded = "203.0.113.195, 70.41.3.18, 150.172.238.178"
        extracted = IPValidator.extract_from_forwarded(forwarded)
        assert extracted == "203.0.113.195"

        # Invalid header (injection attempt)
        malicious = "192.168.1.1; DROP TABLE users"
        assert IPValidator.extract_from_forwarded(malicious) is None


class TestAPIKeyManager:
    """Test API key generation and management."""

    def test_generate_api_key(self):
        """Test secure API key generation."""
        key1 = APIKeyManager.generate_api_key()
        key2 = APIKeyManager.generate_api_key()

        assert key1.startswith("fh_")
        assert key2.startswith("fh_")
        assert key1 != key2  # Should be unique
        assert len(key1) > 20  # Sufficient entropy

    def test_compute_key_index(self):
        """Test API key indexing for database lookups."""
        key = "fh_test123456789abcdef"
        index = APIKeyManager.compute_key_index(key)

        assert index is not None
        assert len(index) == 16  # Consistent length
        # Same key should produce same index
        assert index == APIKeyManager.compute_key_index(key)

    def test_validate_api_key_format(self):
        """Test API key format validation."""
        valid_keys = [
            "fh_abc123DEF456",
            "fh_" + "a" * 40
        ]
        for key in valid_keys:
            assert APIKeyManager.validate_api_key_format(key)

        invalid_keys = [
            "wrong_prefix_123",
            "fh_",  # Too short
            "fh_" + "a" * 200,  # Too long
            "fh_invalid!@#chars"
        ]
        for key in invalid_keys:
            assert not APIKeyManager.validate_api_key_format(key)


class TestTokenValidator:
    """Test JWT token validation."""

    def test_validate_bearer_header(self):
        """Test Bearer token extraction."""
        # Valid Bearer token
        valid_header = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        token = TokenValidator.validate_bearer_header(valid_header)
        assert token is not None
        assert token.count('.') == 2  # JWT format

        # Invalid headers
        invalid_headers = [
            "Basic dXNlcjpwYXNz",
            "Bearer",
            "Bearer invalid.token",
            "",
            "Bearer " + "a" * 3000  # Too long
        ]
        for header in invalid_headers:
            assert TokenValidator.validate_bearer_header(header) is None


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_closed_success(self):
        """Test normal operation when circuit is closed."""
        breaker = CircuitBreaker("test", failure_threshold=3)

        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state.value == "closed"

    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreaker("test", failure_threshold=3)

        async def failing_func():
            raise Exception("Failed")

        # First 3 failures should open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state.value == "open"

        # Next call should be rejected immediately
        with pytest.raises(CircuitOpenError):
            await breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_recovery(self):
        """Test circuit recovery after timeout."""
        breaker = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

        async def failing_func():
            raise Exception("Failed")

        async def success_func():
            return "success"

        # Open the circuit
        for i in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state.value == "open"

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Circuit should enter half-open state and allow test
        result = await breaker.call(success_func)
        assert result == "success"
        # After success in half-open, should close
        assert breaker.state.value == "closed"

    @pytest.mark.asyncio
    async def test_circuit_statistics(self):
        """Test circuit breaker statistics tracking."""
        breaker = CircuitBreaker("test", failure_threshold=5)

        async def mixed_func(should_fail):
            if should_fail:
                raise Exception("Failed")
            return "success"

        # Mix of successes and failures
        await breaker.call(mixed_func, False)  # Success
        await breaker.call(mixed_func, False)  # Success

        with pytest.raises(Exception):
            await breaker.call(mixed_func, True)  # Failure

        stats = breaker.get_stats()
        assert stats["successful_calls"] == 2
        assert stats["failed_calls"] == 1
        assert stats["total_calls"] == 3
        assert stats["failure_rate"] == pytest.approx(33.33, 0.1)


@pytest.mark.asyncio
class TestIntegrationSecurity:
    """Integration tests for security features."""

    async def test_classification_with_sanitization(self):
        """Test classification engine with input sanitization."""
        from services.classification_engine import ClassificationEngine

        engine = ClassificationEngine()

        # Malicious input
        malicious_query = "<script>alert('xss')</script>How to improve security?"

        # Should sanitize and classify without errors
        result = await engine.classify_query(malicious_query)
        assert result is not None
        assert '<script>' not in result.query

    async def test_rate_limiting_with_validation(self):
        """Test rate limiting with IP validation."""
        from services.rate_limiter import RateLimitMiddleware, RateLimiter

        limiter = RateLimiter()
        middleware = RateLimitMiddleware(limiter)

        # Mock request with invalid IP
        mock_request = Mock()
        mock_request.headers = {"X-Forwarded-For": "999.999.999.999"}
        mock_request.client = None
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        # Should extract IP safely or reject
        identifier = await middleware._extract_identifier(mock_request)
        if identifier:
            # If identifier extracted, should be safe
            assert "999.999.999.999" not in identifier

    async def test_api_key_validation_timing(self):
        """Test constant-time API key validation."""
        from services.auth_service import get_api_key_info

        # Mock database session
        mock_db = AsyncMock()

        # Time multiple invalid keys - should take similar time
        times = []
        for i in range(5):
            start = time.perf_counter()
            result = await get_api_key_info(f"fh_invalid_{i}", mock_db)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            assert result is None

        # Times should be relatively consistent (not exact due to async)
        avg_time = sum(times) / len(times)
        for t in times:
            # Allow 50% variance due to async scheduling
            assert abs(t - avg_time) < avg_time * 0.5


@pytest.mark.asyncio
class TestSearchAPICircuitBreakers:
    """Test circuit breakers on search APIs."""

    async def test_search_api_circuit_breaker(self):
        """Test that search APIs have circuit breakers."""
        from services.search_apis import BraveSearchAPI

        # Create API with mock key
        api = BraveSearchAPI("test_key")

        # Check that search method has circuit breaker
        assert hasattr(api.search, 'circuit_breaker')
        breaker = api.search.circuit_breaker
        assert breaker.name == "brave_search"
        assert breaker.failure_threshold == 3


def test_centralized_utilities_exist():
    """Verify all centralized security utilities are available."""
    from utils import security, circuit_breaker

    # Check security module exports
    assert hasattr(security, 'InputSanitizer')
    assert hasattr(security, 'PatternValidator')
    assert hasattr(security, 'IPValidator')
    assert hasattr(security, 'APIKeyManager')
    assert hasattr(security, 'TokenValidator')

    # Check convenience functions
    assert callable(security.sanitize_user_input)
    assert callable(security.validate_and_sanitize_url)
    assert callable(security.is_valid_email)
    assert callable(security.safe_regex_compile)

    # Check circuit breaker exports
    assert hasattr(circuit_breaker, 'CircuitBreaker')
    assert hasattr(circuit_breaker, 'CircuitOpenError')
    assert hasattr(circuit_breaker, 'with_circuit_breaker')
    assert hasattr(circuit_breaker, 'circuit_manager')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])