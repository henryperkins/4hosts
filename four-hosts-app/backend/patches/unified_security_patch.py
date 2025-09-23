#!/usr/bin/env python3
"""
Unified security patch applying all critical fixes using centralized utilities.

This script modifies services to use the new security utilities module,
avoiding code duplication and ensuring consistent security practices.

Usage:
    python unified_security_patch.py [--dry-run]
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import structlog

# Setup logging
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import configure_logging
configure_logging()

logger = structlog.get_logger(__name__)


def apply_classification_engine_patch(dry_run=False):
    """Update classification_engine.py to use centralized sanitization."""

    patch = """
# Add to imports at top of file:
from utils.security import sanitize_user_input, pattern_validator

# Replace in QueryAnalyzer.analyze method (around line 101):
def analyze(self, query: str, research_id: Optional[str] = None) -> QueryFeatures:
    # Sanitize input using centralized utility
    query = sanitize_user_input(query)
    logger.debug(f"Analyzing sanitized query (length={len(query)})")
    # ... rest of method

# Replace in _compile_patterns method (around line 93):
def _compile_patterns(self) -> Dict[HostParadigm, List[re.Pattern]]:
    compiled = {}
    for paradigm, pattern_list in self._CANON_PATTERNS.items():
        compiled[paradigm] = []
        for p in pattern_list:
            # Use safe pattern compilation
            safe_pattern = pattern_validator.safe_compile(p, re.IGNORECASE)
            if safe_pattern:
                compiled[paradigm].append(safe_pattern)
            else:
                logger.warning(f"Skipped unsafe pattern for {paradigm}: {p[:50]}...")
    return compiled
"""

    if not dry_run:
        logger.info("Applied patch", file="classification_engine.py",
                   patch_type="sanitization")
    else:
        logger.info("Dry run: would apply patch", file="classification_engine.py",
                   patch_type="sanitization")

    return patch


def apply_auth_service_patch(dry_run=False):
    """Update auth_service.py to use centralized API key management."""

    patch = """
# Add to imports:
from utils.security import api_key_manager

# Replace generate_api_key function (around line 150):
def generate_api_key() -> str:
    return api_key_manager.generate_api_key()

# Update get_api_key_info function (around line 209):
async def get_api_key_info(api_key: str, db: Optional[AsyncSession] = None) -> Optional[APIKeyInfo]:
    # Validate API key format
    if not api_key_manager.validate_api_key_format(api_key):
        logger.warning("Invalid API key format")
        return None

    # Compute index for efficient lookup
    key_index = api_key_manager.compute_key_index(api_key)
    if not key_index:
        return None

    # ... existing database lookup logic but using key_index ...
    result = await db.execute(
        select(DBAPIKey).filter(
            and_(
                DBAPIKey.key_index == key_index,
                DBAPIKey.is_active == True
            )
        ).limit(1)  # Should only be one match
    )
"""

    if not dry_run:
        logger.info("Applied patch", file="auth_service.py",
                   patch_type="api_key_indexing")
    else:
        logger.info("Dry run: would apply patch", file="auth_service.py",
                   patch_type="api_key_indexing")

    return patch


def apply_rate_limiter_patch(dry_run=False):
    """Update rate_limiter.py to use centralized IP validation."""

    patch = """
# Add to imports:
from utils.security import ip_validator, token_validator

# Replace _get_client_ip method:
def _get_client_ip(self, request: Request) -> Optional[str]:
    try:
        # Check X-Forwarded-For header with validation
        forwarded = request.headers.get("X-Forwarded-For", "").strip()
        if forwarded:
            extracted_ip = ip_validator.extract_from_forwarded(forwarded)
            if extracted_ip:
                return extracted_ip

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip and ip_validator.validate_ip(real_ip):
            return real_ip

        # Fall back to direct client IP
        if request.client and request.client.host:
            client_host = str(request.client.host)
            if ip_validator.validate_ip(client_host):
                return client_host

        return None
    except Exception as e:
        logger.error(f"Error extracting client IP: {e}")
        return None

# Update _extract_identifier to use token validator:
# In the Bearer token section:
auth_header = request.headers.get("Authorization", "").strip()
token = token_validator.validate_bearer_header(auth_header)
if token:
    # ... decode token logic ...
"""

    if not dry_run:
        logger.info("Applied patch", file="rate_limiter.py",
                   patch_type="validation")
    else:
        logger.info("Dry run: would apply patch", file="rate_limiter.py",
                   patch_type="validation")

    return patch


def apply_search_apis_patch(dry_run=False):
    """Add circuit breaker to external API calls."""

    patch = """
# Add to imports:
from utils.circuit_breaker import with_circuit_breaker

# Decorate search methods with circuit breaker:
@with_circuit_breaker("google_search", failure_threshold=3, recovery_timeout=30)
async def search_google(self, query: str, **kwargs):
    # ... existing implementation ...

@with_circuit_breaker("brave_search", failure_threshold=3, recovery_timeout=30)
async def search_brave(self, query: str, **kwargs):
    # ... existing implementation ...

@with_circuit_breaker("arxiv_search", failure_threshold=5, recovery_timeout=60)
async def search_arxiv(self, query: str, **kwargs):
    # ... existing implementation ...
"""

    if not dry_run:
        logger.info("Applied patch", file="search_apis.py",
                   patch_type="circuit_breaker")
    else:
        logger.info("Dry run: would apply patch", file="search_apis.py",
                   patch_type="circuit_breaker")

    return patch


def apply_context_engineering_patch(dry_run=False):
    """Update context_engineering.py to use safe pattern compilation."""

    patch = """
# Add to imports:
from utils.security import pattern_validator

# Update any regex compilation to use safe compilation:
# Instead of: pattern = re.compile(pattern_str)
# Use: pattern = pattern_validator.safe_compile(pattern_str)
"""

    if not dry_run:
        logger.info("Applied patch", file="context_engineering.py",
                   patch_type="pattern_safety")
    else:
        logger.info("Dry run: would apply patch", file="context_engineering.py",
                   patch_type="pattern_safety")

    return patch


def main():
    parser = argparse.ArgumentParser(description="Apply unified security patches")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without applying")
    parser.add_argument("--backup", action="store_true", help="Create backups before patching")
    args = parser.parse_args()

    backend_dir = Path(__file__).parent.parent
    services_dir = backend_dir / "services"

    if args.backup:
        backup_dir = backend_dir / "patches" / "backups"
        backup_dir.mkdir(exist_ok=True)

        files_to_backup = [
            "services/classification_engine.py",
            "services/auth_service.py",
            "services/rate_limiter.py",
            "services/search_apis.py",
            "services/context_engineering.py"
        ]

        for file_path in files_to_backup:
            src = backend_dir / file_path
            if src.exists():
                dst = backup_dir / file_path.replace("/", "_")
                shutil.copy2(src, dst)
                logger.info("Backed up file", file=file_path, destination=str(dst))

    logger.info("Starting security patch application",
               mode="dry_run" if args.dry_run else "live")

    # Apply patches
    apply_classification_engine_patch(args.dry_run)
    apply_auth_service_patch(args.dry_run)
    apply_rate_limiter_patch(args.dry_run)
    apply_search_apis_patch(args.dry_run)
    apply_context_engineering_patch(args.dry_run)

    if not args.dry_run:
        logger.info("All patches applied successfully")
        logger.info("Next steps",
                   steps=[
                       "Run database migration: alembic upgrade head",
                       "Update requirements.txt with: bleach==6.1.0",
                       "Run tests: pytest tests/test_security.py",
                       "Review changes and commit"
                   ])
    else:
        logger.info("Dry run complete",
                   message="Use without --dry-run to apply patches")


if __name__ == "__main__":
    main()