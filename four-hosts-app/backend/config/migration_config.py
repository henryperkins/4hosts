"""
Migration Configuration
Controls V1/V2 feature flags and migration settings
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MigrationConfig:
    """Configuration for V1/V2 migration"""
    
    # Feature flags
    use_v2_research_store: bool = False
    use_v2_context_engineering: bool = False
    
    # Migration settings
    enable_migration_logging: bool = True
    enable_fallback_to_v1: bool = True
    migration_test_mode: bool = True
    
    # Performance settings
    cache_enabled: bool = True
    redis_ttl_seconds: int = 86400  # 24 hours
    cleanup_interval_minutes: int = 5
    
    # Debug settings
    include_debug_info: bool = False
    log_performance_metrics: bool = True
    
    @classmethod
    def from_environment(cls) -> 'MigrationConfig':
        """Create config from environment variables"""
        return cls(
            # V2 feature flags
            use_v2_research_store=_env_bool("USE_V2_RESEARCH_STORE", False),
            use_v2_context_engineering=_env_bool("USE_V2_CONTEXT_ENGINEERING", False),
            
            # Migration settings
            enable_migration_logging=_env_bool("ENABLE_MIGRATION_LOGGING", True),
            enable_fallback_to_v1=_env_bool("ENABLE_V1_FALLBACK", True),
            migration_test_mode=_env_bool("MIGRATION_TEST_MODE", True),
            
            # Performance settings
            cache_enabled=_env_bool("ENABLE_V2_CACHE", True),
            redis_ttl_seconds=int(os.getenv("REDIS_TTL_SECONDS", "86400")),
            cleanup_interval_minutes=int(os.getenv("CLEANUP_INTERVAL_MINUTES", "5")),
            
            # Debug settings
            include_debug_info=_env_bool("INCLUDE_DEBUG_INFO", False),
            log_performance_metrics=_env_bool("LOG_PERFORMANCE_METRICS", True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "use_v2_research_store": self.use_v2_research_store,
            "use_v2_context_engineering": self.use_v2_context_engineering,
            "enable_migration_logging": self.enable_migration_logging,
            "enable_fallback_to_v1": self.enable_fallback_to_v1,
            "migration_test_mode": self.migration_test_mode,
            "cache_enabled": self.cache_enabled,
            "redis_ttl_seconds": self.redis_ttl_seconds,
            "cleanup_interval_minutes": self.cleanup_interval_minutes,
            "include_debug_info": self.include_debug_info,
            "log_performance_metrics": self.log_performance_metrics
        }
    
    def get_research_store(self):
        """Get appropriate research store instance"""
        if self.use_v2_research_store:
            from services.research_store_v2 import research_store_v2
            return research_store_v2
        else:
            from services.research_store import research_store
            return research_store
    
    def get_context_pipeline(self):
        """Get appropriate context engineering pipeline"""
        if self.use_v2_context_engineering:
            from services.context_engineering_bridge import context_engineering_bridge
            context_engineering_bridge.use_v2 = True
            return context_engineering_bridge
        else:
            from services.context_engineering import context_pipeline
            return context_pipeline


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable"""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


# Global configuration instance
migration_config = MigrationConfig.from_environment()


# Utility functions for easy access
def use_v2_research_store() -> bool:
    """Check if V2 research store should be used"""
    return migration_config.use_v2_research_store


def use_v2_context_engineering() -> bool:
    """Check if V2 context engineering should be used"""
    return migration_config.use_v2_context_engineering


def get_research_store():
    """Get the appropriate research store"""
    return migration_config.get_research_store()


def get_context_pipeline():
    """Get the appropriate context pipeline"""
    return migration_config.get_context_pipeline()


def get_migration_config() -> MigrationConfig:
    """Get the current migration configuration"""
    return migration_config


def update_migration_config(**kwargs):
    """Update migration configuration"""
    global migration_config
    
    for key, value in kwargs.items():
        if hasattr(migration_config, key):
            setattr(migration_config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")


# Environment setup examples for different stages
DEVELOPMENT_CONFIG = {
    "USE_V2_RESEARCH_STORE": "false",
    "USE_V2_CONTEXT_ENGINEERING": "false",
    "MIGRATION_TEST_MODE": "true",
    "ENABLE_V1_FALLBACK": "true",
    "INCLUDE_DEBUG_INFO": "true"
}

STAGING_CONFIG = {
    "USE_V2_RESEARCH_STORE": "true",
    "USE_V2_CONTEXT_ENGINEERING": "true", 
    "MIGRATION_TEST_MODE": "true",
    "ENABLE_V1_FALLBACK": "true",
    "INCLUDE_DEBUG_INFO": "true",
    "LOG_PERFORMANCE_METRICS": "true"
}

PRODUCTION_CONFIG = {
    "USE_V2_RESEARCH_STORE": "true",
    "USE_V2_CONTEXT_ENGINEERING": "true",
    "MIGRATION_TEST_MODE": "false",
    "ENABLE_V1_FALLBACK": "false",
    "INCLUDE_DEBUG_INFO": "false",
    "LOG_PERFORMANCE_METRICS": "true"
}


def apply_environment_config(environment: str):
    """Apply configuration for specific environment"""
    configs = {
        "development": DEVELOPMENT_CONFIG,
        "staging": STAGING_CONFIG,
        "production": PRODUCTION_CONFIG
    }
    
    if environment not in configs:
        raise ValueError(f"Unknown environment: {environment}")
    
    config = configs[environment]
    
    # Update environment variables
    for key, value in config.items():
        os.environ[key] = value
    
    # Reload configuration
    global migration_config
    migration_config = MigrationConfig.from_environment()
    
    return migration_config
