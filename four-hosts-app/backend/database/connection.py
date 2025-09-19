"""
Database Connection and Session Management
Phase 5: Production-Ready Features
"""

import os
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import structlog

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import event, text

from database.models import Base

# Configure logging (structlog is configured centrally in services.monitoring)
logger = structlog.get_logger(__name__)


# Database configuration
class DatabaseConfig:
    """Database configuration"""

    def __init__(self):
        # Build database URL from individual components
        pghost = os.getenv("PGHOST", "localhost")
        pguser = os.getenv("PGUSER", "user")
        pgpassword = os.getenv("PGPASSWORD", "password")
        pgport = os.getenv("PGPORT", "5432")
        pgdatabase = os.getenv("PGDATABASE", "fourhosts")

        self.database_url = os.getenv(
            "DATABASE_URL",
            f"postgresql+asyncpg://{pguser}:{pgpassword}@{pghost}:{pgport}/{pgdatabase}",
        )

        # Debug: Print the database URL (without password for security)
        logger.info(
            f"Database URL: postgresql+asyncpg://{pguser}:****@{pghost}:{pgport}/{pgdatabase}"
        )

        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "40"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))

        # Query settings
        self.echo_sql = os.getenv("DB_ECHO_SQL", "false").lower() == "true"
        self.slow_query_threshold = float(os.getenv("DB_SLOW_QUERY_THRESHOLD", "1.0"))

        # SSL settings
        self.ssl_mode = os.getenv("DB_SSL_MODE", "prefer")

    def get_engine_kwargs(self):
        """Get SQLAlchemy engine configuration"""
        kwargs = {
            "echo": self.echo_sql,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": True,  # Verify connections before use
            "connect_args": {
                "server_settings": {
                    "application_name": "four_hosts_research_api",
                    "jit": "off",
                },
                "command_timeout": 60,
                # Remove driver-unsupported libpq-style ssl modes for asyncpg;
                # if SSL is required, configure an ssl.SSLContext and pass here.
                # "ssl": None,
            },
        }

        # Use NullPool for serverless environments
        if os.getenv("SERVERLESS", "false").lower() == "true":
            kwargs["poolclass"] = NullPool

        return kwargs


# Global database configuration
db_config = DatabaseConfig()

# Create async engine
engine: AsyncEngine = create_async_engine(
    db_config.database_url, **db_config.get_engine_kwargs()
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# --- Database Events ---


@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set SQLite pragmas for better performance (dev only)"""
    if "sqlite" in db_config.database_url:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()


# --- Session Management ---


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# --- Database Operations ---


class DatabaseManager:
    """Database management operations"""

    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    async def create_all_tables(self):
        """Create all database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("All database tables created")

    async def drop_all_tables(self):
        """Drop all database tables (use with caution!)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("All database tables dropped")

    async def check_connection(self) -> bool:
        """Check if database is accessible"""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    async def get_table_sizes(self) -> dict:
        """Get size of all tables"""
        query = """
        SELECT
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
            pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes
        FROM pg_tables
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
        """

        async with self.engine.connect() as conn:
            result = await conn.execute(text(query))
            return {
                row.tablename: {"size": row.size, "size_bytes": row.size_bytes}
                for row in result
            }

    async def vacuum_analyze(self, table_name: Optional[str] = None):
        """Run VACUUM ANALYZE on tables"""
        async with self.engine.connect() as conn:
            if table_name:
                await conn.execute(text(f"VACUUM ANALYZE {table_name}"))
                logger.info(f"VACUUM ANALYZE completed for {table_name}")
            else:
                await conn.execute(text("VACUUM ANALYZE"))
                logger.info("VACUUM ANALYZE completed for all tables")

    async def get_slow_queries(self, min_duration_ms: int = 1000) -> list:
        """Get slow queries from pg_stat_statements"""
        query = """
        SELECT
            query,
            calls,
            total_exec_time,
            mean_exec_time,
            stddev_exec_time,
            min_exec_time,
            max_exec_time
        FROM pg_stat_statements
        WHERE mean_exec_time > :min_duration
        ORDER BY mean_exec_time DESC
        LIMIT 20;
        """

        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(
                    text(query), {"min_duration": min_duration_ms}
                )
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")
            return []


# Create global database manager
db_manager = DatabaseManager(engine)

# --- Query Builders ---


class QueryBuilder:
    """Helper class for building complex queries"""

    @staticmethod
    def build_search_query(
        search_term: str, fields: list[str], use_trigram: bool = True
    ) -> str:
        """Build full-text search query"""
        if use_trigram:
            # Use trigram similarity for fuzzy matching
            conditions = [
                f"similarity({field}, :search_term) > 0.3" for field in fields
            ]
            order_by = f"similarity({fields[0]}, :search_term) DESC"
        else:
            # Use standard ILIKE
            conditions = [f"{field} ILIKE :search_pattern" for field in fields]
            order_by = f"{fields[0]}"

        where_clause = " OR ".join(conditions)

        return f"""
        SELECT *
        FROM research_queries
        WHERE {where_clause}
        ORDER BY {order_by}
        LIMIT 100
        """

    @staticmethod
    def build_analytics_query(
        user_id: str, start_date: str, end_date: str, granularity: str = "day"
    ) -> str:
        """Build analytics aggregation query"""
        date_trunc = {
            "hour": "hour",
            "day": "day",
            "week": "week",
            "month": "month",
        }.get(granularity, "day")

        return f"""
        SELECT
            date_trunc('{date_trunc}', created_at) as period,
            COUNT(*) as total_queries,
            COUNT(DISTINCT primary_paradigm) as unique_paradigms,
            AVG(confidence_score) as avg_confidence,
            AVG(sources_analyzed) as avg_sources,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_queries,
            AVG(duration_seconds) as avg_duration
        FROM research_queries
        WHERE user_id = :user_id
            AND created_at BETWEEN :start_date AND :end_date
        GROUP BY date_trunc('{date_trunc}', created_at)
        ORDER BY period ASC
        """


# --- Connection Pooling Monitor ---


class ConnectionPoolMonitor:
    """Monitor database connection pool health"""

    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    def get_pool_status(self) -> dict:
        """Get current pool status (robust to different pool types)"""
        pool = self.engine.pool
        try:
            size = getattr(pool, "size", lambda: 0)()
            checked_in = getattr(pool, "checkedin", lambda: 0)()
            checked_out = getattr(pool, "checkedout", lambda: 0)()
            overflow = getattr(pool, "overflow", lambda: 0)()
            total = getattr(pool, "total", lambda: size)()
            status = "healthy" if (checked_in > 0) or (total and (total - checked_out) > 0) else "degraded"
        except Exception:
            size = checked_in = checked_out = overflow = total = 0
            status = "unknown"

        return {
            "pool_class": pool.__class__.__name__,
            "size": size,
            "checked_in": checked_in,
            "checked_out": checked_out,
            "overflow": overflow,
            "total": total,
            "status": status,
        }

    async def reset_pool(self):
        """Reset connection pool"""
        await self.engine.dispose()
        logger.info("Database connection pool reset")


# Create global pool monitor
pool_monitor = ConnectionPoolMonitor(engine)


# Simple wrapper for main.py
async def init_database():
    """Initialize database tables"""
    await run_migrations()


# --- Database Migrations ---


async def run_migrations():
    """Run database migrations using Alembic"""
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config(str((Path(__file__).resolve().parent.parent / "alembic.ini").resolve()))

    def _upgrade_head() -> None:
        command.upgrade(alembic_cfg, "head")

    await asyncio.to_thread(_upgrade_head)
    logger.info("Alembic migrations applied to head")


# --- Health Checks ---


async def database_health_check() -> dict:
    """Comprehensive database health check"""
    health = {"status": "unknown", "details": {}}

    try:
        # Check connection
        if await db_manager.check_connection():
            health["details"]["connection"] = "ok"
        else:
            health["status"] = "unhealthy"
            health["details"]["connection"] = "failed"
            return health

        # Check pool status
        pool_status = pool_monitor.get_pool_status()
        health["details"]["pool"] = pool_status

        if pool_status["status"] == "exhausted":
            health["status"] = "degraded"
        else:
            health["status"] = "healthy"

        # Get table sizes
        table_sizes = await db_manager.get_table_sizes()
        health["details"]["largest_tables"] = dict(list(table_sizes.items())[:5])

        return health

    except Exception as e:
        health["status"] = "unhealthy"
        health["details"]["error"] = str(e)
        return health
