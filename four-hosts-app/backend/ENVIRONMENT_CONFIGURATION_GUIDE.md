# Environment Configuration Guide

## Overview

This guide explains how to use environment variables to configure the Four Hosts backend application. Moving hardcoded values to environment variables improves flexibility, enables environment-specific tuning, and facilitates better operational management.

## Table of Contents

1. [Configuration Structure](#configuration-structure)
2. [Loading Environment Variables](#loading-environment-variables)
3. [Configuration Categories](#configuration-categories)
4. [Best Practices](#best-practices)
5. [Code Implementation Patterns](#code-implementation-patterns)
6. [Validation and Defaults](#validation-and-defaults)
7. [Environment-Specific Examples](#environment-specific-examples)

## Configuration Structure

Environment variables are organized into logical categories:

- **TIMEOUT_CONFIGURATIONS**: All timeout-related settings
- **RATE_LIMITING_AND_CONCURRENCY**: Rate limits and concurrency controls
- **SEARCH_CONFIGURATION**: Search behavior and provider settings
- **DEDUPLICATION_CONFIGURATION**: Result deduplication parameters
- **EARLY_FILTERING_CONFIGURATION**: Content filtering rules
- **COST_MONITORING_AND_BUDGETING**: Cost tracking and budget controls
- **AGENTIC_CONFIGURATION**: Agentic behavior settings
- **SYNTHESIS_CONFIGURATION**: Answer synthesis parameters
- **CREDIBILITY_SCORING**: Credibility assessment settings
- **QUERY_OPTIMIZATION**: Query processing and optimization
- **API_ENDPOINTS_AND_HEADERS**: External API configurations
- **CIRCUIT_BREAKER_CONFIGURATION**: Fault tolerance settings
- **CACHE_CONFIGURATION**: Caching behavior controls
- **MISCELLANEOUS**: Other configuration options

## Loading Environment Variables

The application uses `python-dotenv` to load environment variables from `.env` files:

```python
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get configuration value with fallback
timeout_sec = float(os.getenv('SEARCH_TASK_TIMEOUT_DEFAULT', '30'))
```

## Configuration Categories

### 1. Timeout Configurations

Control how long the application waits for various operations:

```python
# Example usage
task_timeout = float(os.getenv('SEARCH_TASK_TIMEOUT_DEFAULT', '30'))
provider_timeout = float(os.getenv('SEARCH_PROVIDER_TIMEOUT_DEFAULT', '25'))
```

Key variables:
- `SEARCH_TASK_TIMEOUT_DEFAULT`: Overall search timeout
- `SEARCH_PROVIDER_TIMEOUT_DEFAULT`: Individual provider timeout
- `OPENAI_CLIENT_TIMEOUT`: LLM API timeout
- `SEARCH_FETCH_TIMEOUT_DEFAULT`: URL fetch timeout

### 2. Rate Limiting and Concurrency

Manage API usage and concurrent operations:

```python
# Example usage
max_concurrent = int(os.getenv('CREDIBILITY_CHECK_CONCURRENCY', '8'))
rate_limit = int(os.getenv('SEARCH_RATE_LIMITER_DEFAULT_CPM', '60'))
```

Key variables:
- `SEARCH_RATE_LIMITER_DEFAULT_CPM`: Default rate limit (calls per minute)
- `CREDIBILITY_CHECK_CONCURRENCY`: Concurrent credibility checks
- `SEARCH_VARIANT_CONCURRENCY`: Concurrent query variants

### 3. Search Configuration

Control search behavior and provider-specific settings:

```python
# Example usage
max_results = int(os.getenv('SEARCH_MAX_RESULTS_DEFAULT', '50'))
language = os.getenv('SEARCH_LANGUAGE_DEFAULT', 'en')
```

Provider-specific settings:
- `BRAVE_MAX_RESULTS_DEFAULT`: Max results from Brave
- `GOOGLE_MAX_RESULTS_DEFAULT`: Max results from Google
- `ARXIV_MAX_RESULTS_DEFAULT`: Max results from ArXiv

### 4. Deduplication Configuration

Configure result deduplication behavior:

```python
# Example usage
similarity_threshold = float(os.getenv('DEDUP_SIMILARITY_THRESHOLD_DEFAULT', '0.8'))
title_weight = float(os.getenv('DEDUP_TITLE_WEIGHT', '0.5'))
```

### 5. Early Filtering Configuration

Control content filtering rules:

```python
# Example usage
spam_keywords = os.getenv('EARLY_FILTER_SPAM_KEYWORDS', '').split(',')
min_title_length = int(os.getenv('EARLY_FILTER_MIN_TITLE_LENGTH', '10'))
```

### 6. Cost Monitoring

Track API costs and set budgets:

```python
# Example usage
import json
api_costs = json.loads(os.getenv('SEARCH_API_COSTS', '{}'))
google_cost = api_costs.get('google', 0.005)
```

### 7. Agentic Configuration

Configure agentic search behavior:

```python
# Example usage
max_iterations = int(os.getenv('AGENTIC_MAX_ITERATIONS_DEFAULT', '2'))
coverage_threshold = float(os.getenv('AGENTIC_COVERAGE_THRESHOLD_DEFAULT', '0.75'))
```

## Best Practices

### 1. Use Descriptive Variable Names

```python
# Good
SEARCH_TASK_TIMEOUT_DEFAULT=30

# Avoid
TIMEOUT=30
```

### 2. Provide Sensible Defaults

```python
# Always provide a default in code
timeout = float(os.getenv('SEARCH_TASK_TIMEOUT_DEFAULT', '30'))

# This ensures the app works even if the env var is missing
```

### 3. Use Appropriate Types

```python
# Numbers
max_results = int(os.getenv('SEARCH_MAX_RESULTS_DEFAULT', '50'))

# Floats
threshold = float(os.getenv('DEDUP_SIMILARITY_THRESHOLD_DEFAULT', '0.8'))

# Booleans
feature_enabled = os.getenv('FEATURE_FLAG', '0').lower() in {'1', 'true', 'yes'}

# Lists
spam_keywords = [k.strip() for k in os.getenv('SPAM_KEYWORDS', '').split(',') if k.strip()]

# JSON
config = json.loads(os.getenv('COMPLEX_CONFIG', '{}'))
```

### 4. Group Related Variables

Use prefixes to group related configurations:
- `SEARCH_*` for search-related settings
- `DEDUP_*` for deduplication settings
- `CACHE_*` for cache settings

### 5. Document Units and Ranges

```python
# In documentation
SEARCH_TASK_TIMEOUT_DEFAULT=30  # seconds
CACHE_SEARCH_RESULTS_TTL=86400  # seconds (24 hours)
MAX_RETRIES=3  # number of attempts
```

## Code Implementation Patterns

### Pattern 1: Simple Configuration

```python
# In your service class
class SearchService:
    def __init__(self):
        self.timeout = float(os.getenv('SEARCH_TIMEOUT_DEFAULT', '30'))
        self.max_results = int(os.getenv('SEARCH_MAX_RESULTS_DEFAULT', '50'))
```

### Pattern 2: Configuration with Validation

```python
def get_timeout():
    """Get timeout with validation"""
    timeout = float(os.getenv('SEARCH_TIMEOUT_DEFAULT', '30'))
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    if timeout > 300:
        logger.warning("Timeout > 300s may cause performance issues")
    return timeout
```

### Pattern 3: Complex Configuration

```python
class QueryOptimizer:
    def __init__(self):
        # Load technical terms from JSON
        self.technical_terms = json.loads(os.getenv(
            'QUERY_OPTIMIZER_TECHNICAL_TERMS',
            '{"context engineering": "web applications"}'
        ))

        # Load noise terms as list
        self.noise_terms = [
            t.strip() for t in os.getenv(
                'QUERY_OPTIMIZER_NOISE_TERMS',
                'information,details,find,show,tell'
            ).split(',') if t.strip()
        ]
```

### Pattern 4: Configuration Classes

```python
@dataclass
class SearchConfig:
    max_results: int
    language: str
    timeout: float

    @classmethod
    def from_env(cls):
        return cls(
            max_results=int(os.getenv('SEARCH_MAX_RESULTS_DEFAULT', '50')),
            language=os.getenv('SEARCH_LANGUAGE_DEFAULT', 'en'),
            timeout=float(os.getenv('SEARCH_TIMEOUT_DEFAULT', '30'))
        )
```

## Validation and Defaults

### 1. Create a Configuration Module

```python
# config.py
import os
import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class Config:
    """Central configuration management"""

    @staticmethod
    def get_int(key: str, default: int, min_val: int = None, max_val: int = None) -> int:
        """Get integer configuration with validation"""
        try:
            value = int(os.getenv(key, str(default)))
            if min_val is not None and value < min_val:
                logger.warning(f"{key}={value} below minimum {min_val}, using {min_val}")
                value = min_val
            if max_val is not None and value > max_val:
                logger.warning(f"{key}={value} above maximum {max_val}, using {max_val}")
                value = max_val
            return value
        except ValueError:
            logger.error(f"Invalid {key}, using default {default}")
            return default

    @staticmethod
    def get_float(key: str, default: float, min_val: float = None, max_val: float = None) -> float:
        """Get float configuration with validation"""
        try:
            value = float(os.getenv(key, str(default)))
            if min_val is not None and value < min_val:
                value = min_val
            if max_val is not None and value > max_val:
                value = max_val
            return value
        except ValueError:
            logger.error(f"Invalid {key}, using default {default}")
            return default

    @staticmethod
    def get_bool(key: str, default: bool) -> bool:
        """Get boolean configuration"""
        return os.getenv(key, str(default)).lower() in {'1', 'true', 'yes', 'on'}

    @staticmethod
    def get_list(key: str, default: List[str]) -> List[str]:
        """Get list configuration from comma-separated string"""
        value = os.getenv(key, '')
        if not value:
            return default
        return [item.strip() for item in value.split(',') if item.strip()]

    @staticmethod
    def get_json(key: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """Get JSON configuration"""
        try:
            return json.loads(os.getenv(key, json.dumps(default)))
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {key}, using default")
            return default
```

### 2. Usage Example

```python
# In your service
from config import Config

class SearchService:
    def __init__(self):
        self.max_results = Config.get_int('SEARCH_MAX_RESULTS_DEFAULT', 50, 1, 1000)
        self.timeout = Config.get_float('SEARCH_TIMEOUT_DEFAULT', 30.0, 1.0, 300.0)
        self.enabled = Config.get_bool('SEARCH_ENABLED', True)
        self.providers = Config.get_list('SEARCH_PROVIDERS', ['google', 'brave'])
```

## Environment-Specific Examples

### Development Environment (.env.development)

```bash
# Faster timeouts for development
SEARCH_TASK_TIMEOUT_DEFAULT=10
SEARCH_PROVIDER_TIMEOUT_DEFAULT=8

# Lower rate limits for API safety
SEARCH_RATE_LIMITER_DEFAULT_CPM=10
BRAVE_RATE_LIMITER_DEFAULT_CPM=10

# Debug logging
SEARCH_LOG_BODY_MAX_DEFAULT=5000

# Allow more results for testing
SEARCH_MAX_RESULTS_DEFAULT=100
```

### Production Environment (.env.production)

```bash
# Conservative timeouts
SEARCH_TASK_TIMEOUT_DEFAULT=30
SEARCH_PROVIDER_TIMEOUT_DEFAULT=25

# Higher rate limits for performance
SEARCH_RATE_LIMITER_DEFAULT_CPM=100
BRAVE_RATE_LIMITER_DEFAULT_CPM=100

# Limited logging
SEARCH_LOG_BODY_MAX_DEFAULT=1000

# Cost controls enabled
COST_MONITORING_ENABLED=1
BUDGET_DEFAULT_COST_USD=100
```

### Testing Environment (.env.test)

```bash
# Minimal timeouts for fast tests
SEARCH_TASK_TIMEOUT_DEFAULT=5
SEARCH_PROVIDER_TIMEOUT_DEFAULT=3

# Mock external services
USE_MOCK_SERVICES=1

# Disable expensive operations
PDF_PROCESSING_ENABLED=0
DEEP_RESEARCH_ENABLED=0
```

## Migration Strategy

1. **Phase 1**: Create the `.env.considerations.example` file with all identified variables
2. **Phase 2**: Update services to use environment variables (one service at a time)
3. **Phase 3**: Add validation and configuration classes
4. **Phase 4**: Create environment-specific templates
5. **Phase 5**: Update documentation and deployment guides

## Testing Configuration Changes

1. **Unit Tests**: Test configuration loading with various env var values
2. **Integration Tests**: Verify behavior changes with different configurations
3. **Performance Tests**: Ensure new timeouts don't cause regressions
4. **Load Tests**: Verify rate limiting works under load

## Enhanced Integration Configuration

The enhanced integration system provides self-healing capabilities and ML-powered improvements:

```python
# Usage in enhanced_integration.py
self_healing_enabled = os.getenv('ENHANCED_INTEGRATION_SELF_HEALING_ENABLED', 'true').lower() in {'true', '1', 'yes'}
ml_confidence_threshold = float(os.getenv('ENHANCED_INTEGRATION_ML_CONFIDENCE_THRESHOLD', '0.8'))
```

Key configurations:
- **Self-Healing**: Enable/disable automatic paradigm switching
- **ML Enhancement**: Control ML-powered classification improvements
- **Fallback Paradigms**: Configure which paradigm to fall back to when one fails
- **Performance Thresholds**: Set thresholds for retraining and performance alerts

### Example: Paradigm Fallback Configuration

```python
# Configure fallback chains for each paradigm
fallback_map = {
    HostParadigm.DOLORES: os.getenv('ENHANCED_INTEGRATION_DOLORES_FALLBACK', 'teddy'),
    HostParadigm.TEDDY: os.getenv('ENHANCED_INTEGRATION_TEDDY_FALLBACK', 'bernard'),
    # ... other paradigms
}
```

## Answer Generation Configuration

The answer generation system uses sophisticated scoring and weighting mechanisms:

### Scoring System

```python
# Calculate result ranking score
credibility_weight = float(os.getenv('ANSWER_SCORING_CREDIBILITY_WEIGHT', '0.6'))
evidence_weight = float(os.getenv('ANSWER_SCORING_EVIDENCE_WEIGHT', '0.25'))
recency_weight = float(os.getenv('ANSWER_SCORING_RECENCY_WEIGHT', '0.15'))

score = (credibility * credibility_weight +
         evidence_density * evidence_weight +
         recency * recency_weight)
```

### Section Weight Configuration

Each paradigm uses different section weights. Configure these based on your needs:

```python
# Bernard's analytical focus
bernard_weights = {
    'executive_summary': float(os.getenv('BERNARD_SECTION_WEIGHT_SUMMARY', '0.15')),
    'quantitative_analysis': float(os.getenv('BERNARD_SECTION_WEIGHT_QUANTITATIVE', '0.25')),
    # ... other sections
}
```

### Statistical Analysis Configuration

Configure thresholds for statistical significance:

```python
# Bernard-specific statistical thresholds
p_value_threshold = float(os.getenv('ANSWER_P_VALUE_SIGNIFICANCE', '0.05'))
effect_size_threshold = float(os.getenv('ANSWER_LARGE_EFFECT_SIZE', '0.8'))
```

## Advanced Configuration Patterns

### 1. Feature Flagging with Environment Variables

```python
# Enable/disable features based on environment
ENABLE_SELF_HEALING = os.getenv('ENHANCED_INTEGRATION_SELF_HEALING_ENABLED', 'true').lower() in {'true', '1', 'yes'}
ENABLE_ML_ENHANCEMENT = os.getenv('ENHANCED_INTEGRATION_ML_ENHANCED', 'true').lower() in {'true', '1', 'yes'}

if ENABLE_SELF_HEALING:
    # Initialize self-healing system
    pass
```

### 2. Environment-Specific Thresholds

```python
# Development - more lenient
if environment == 'development':
    confidence_threshold = 0.6
    retry_attempts = 5

# Production - more strict
elif environment == 'production':
    confidence_threshold = 0.8
    retry_attempts = 3
```

### 3. Dynamic Configuration Loading

```python
def load_paradigm_config(paradigm: str) -> Dict[str, float]:
    """Load configuration for a specific paradigm"""
    prefix = f"{paradigm.upper()}_SECTION_WEIGHT_"
    return {
        'exposing': float(os.getenv(f'{prefix}EXPOSING', '0.3')),
        'voices': float(os.getenv(f'{prefix}VOICES', '0.25')),
        # ... other sections
    }
```

## Configuration Validation

Implement validation for critical configurations:

```python
def validate_config():
    """Validate environment configuration"""
    errors = []

    # Check scoring weights sum to 1.0
    weights = [
        float(os.getenv('ANSWER_SCORING_CREDIBILITY_WEIGHT', '0.6')),
        float(os.getenv('ANSWER_SCORING_EVIDENCE_WEIGHT', '0.25')),
        float(os.getenv('ANSWER_SCORING_RECENCY_WEIGHT', '0.15'))
    ]

    if abs(sum(weights) - 1.0) > 0.01:
        errors.append("Scoring weights must sum to 1.0")

    # Check thresholds are within valid ranges
    for threshold in ['ML_CONFIDENCE_THRESHOLD', 'PERFORMANCE_SCORE_THRESHOLD']:
        value = float(os.getenv(f'ENHANCED_INTEGRATION_{threshold}', '0.8'))
        if not 0 <= value <= 1:
            errors.append(f"{threshold} must be between 0 and 1")

    return errors
```

## Troubleshooting

1. **Missing Variables**: Check if all required variables are set
2. **Type Errors**: Ensure proper type conversion (int, float, bool)
3. **Validation Failures**: Check if values are within acceptable ranges
4. **Performance Issues**: Monitor timeout and rate limit changes
5. **Scoring Issues**: Verify weights sum to expected values (e.g., 1.0 for scoring weights)
6. **Section Weights**: Ensure paradigm section weights sum appropriately per paradigm

## Best Practices for Complex Configurations

1. **Group Related Variables**: Use prefixes to group related configurations
2. **Document Units**: Always specify units (seconds, tokens, percentage)
3. **Provide Defaults**: Never use environment variables without defaults
4. **Validate at Startup**: Fail fast if configuration is invalid
5. **Log Active Configuration**: Log the effective configuration at startup
6. **Use Configuration Classes**: For complex configurations, use dedicated classes

## Conclusion

Proper environment configuration management enables:
- Better operational flexibility
- Environment-specific tuning
- Safer deployments
- Easier debugging and troubleshooting
- Improved maintainability

The comprehensive configuration system supports advanced features like self-healing, ML enhancement, and sophisticated answer generation while maintaining operational flexibility through environment variables.

Follow the patterns and best practices outlined in this guide to ensure robust configuration management in your application.

Follow the patterns and best practices outlined in this guide to ensure robust configuration management in your application.