"""
Database Models for Four Hosts Research API
Phase 5: Production-Ready Features
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Table,
    Index,
    UniqueConstraint,
    CheckConstraint,
    Enum,
    ARRAY,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()

# --- Enums ---


class UserRole(str, PyEnum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


class AuthProvider(str, PyEnum):
    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"
    SAML = "saml"


class ParadigmType(str, PyEnum):
    DOLORES = "dolores"
    TEDDY = "teddy"
    BERNARD = "bernard"
    MAEVE = "maeve"


class ResearchDepth(str, PyEnum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ResearchStatus(str, PyEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WebhookEvent(str, PyEnum):
    RESEARCH_STARTED = "research.started"
    RESEARCH_PROGRESS = "research.progress"
    RESEARCH_COMPLETED = "research.completed"
    RESEARCH_FAILED = "research.failed"
    RESEARCH_CANCELLED = "research.cancelled"
    CLASSIFICATION_COMPLETED = "classification.completed"
    SYNTHESIS_COMPLETED = "synthesis.completed"
    EXPORT_READY = "export.ready"


# --- Association Tables ---

user_saved_searches = Table(
    "user_saved_searches",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")),
    Column(
        "research_id",
        UUID(as_uuid=True),
        ForeignKey("research_queries.id", ondelete="CASCADE"),
    ),
    Column(
        "saved_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    ),
    Column("tags", ARRAY(String), default=[]),
    Column("notes", Text),
)

webhook_events = Table(
    "webhook_events_mapping",
    Base.metadata,
    Column(
        "webhook_id", UUID(as_uuid=True), ForeignKey("webhooks.id", ondelete="CASCADE")
    ),
    Column("event", Enum(WebhookEvent)),
)

# --- User Models ---


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255))  # Null for OAuth users
    role = Column(Enum(UserRole), default=UserRole.FREE, nullable=False)
    auth_provider = Column(
        Enum(AuthProvider), default=AuthProvider.LOCAL, nullable=False
    )

    # Profile
    full_name = Column(String(255))
    avatar_url = Column(String(500))
    bio = Column(Text)
    preferences = Column(JSONB, default={})

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_token = Column(String(255), unique=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_login = Column(DateTime(timezone=True))

    # Relationships
    api_keys = relationship(
        "APIKey", back_populates="user", cascade="all, delete-orphan"
    )
    research_queries = relationship(
        "ResearchQuery", back_populates="user", cascade="all, delete-orphan"
    )
    saved_searches = relationship(
        "ResearchQuery", secondary=user_saved_searches, back_populates="saved_by_users"
    )
    webhooks = relationship(
        "Webhook", back_populates="user", cascade="all, delete-orphan"
    )
    exports = relationship(
        "Export", back_populates="user", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_user_email_provider", "email", "auth_provider"),
        Index("idx_user_created", "created_at"),
    )


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    key_hash = Column(String(255), unique=True, nullable=False)  # Hashed API key
    name = Column(String(100), nullable=False)

    # Permissions
    role = Column(Enum(UserRole), nullable=False)
    allowed_origins = Column(ARRAY(String), default=[])
    allowed_ips = Column(ARRAY(String), default=[])
    rate_limit_tier = Column(String(50), default="standard")

    # Usage
    last_used = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    revoked_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="api_keys")

    # Indexes
    __table_args__ = (
        Index("idx_api_key_user", "user_id"),
        Index("idx_api_key_active", "is_active"),
    )


# --- Research Models ---


class ResearchQuery(Base):
    __tablename__ = "research_queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    # Query details
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), index=True)  # For deduplication
    language = Column(String(10), default="en")
    region = Column(String(10))

    # Classification
    primary_paradigm = Column(Enum(ParadigmType), nullable=False)
    secondary_paradigm = Column(Enum(ParadigmType))
    paradigm_scores = Column(JSONB, default={})
    classification_confidence = Column(Float)
    paradigm_override = Column(Enum(ParadigmType))  # User override

    # Options
    depth = Column(Enum(ResearchDepth), default=ResearchDepth.STANDARD, nullable=False)
    max_sources = Column(Integer, default=100)
    include_secondary = Column(Boolean, default=True)
    custom_prompts = Column(JSONB, default={})

    # Status
    status = Column(Enum(ResearchStatus), default=ResearchStatus.QUEUED, nullable=False)
    progress = Column(Integer, default=0)
    current_phase = Column(String(50))
    error_message = Column(Text)

    # Metrics
    sources_found = Column(Integer, default=0)
    sources_analyzed = Column(Integer, default=0)
    synthesis_score = Column(Float)
    confidence_score = Column(Float)
    duration_seconds = Column(Float)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="research_queries")
    saved_by_users = relationship(
        "User", secondary=user_saved_searches, back_populates="saved_searches"
    )
    sources = relationship(
        "ResearchSource", back_populates="research_query", cascade="all, delete-orphan"
    )
    answer = relationship(
        "ResearchAnswer",
        back_populates="research_query",
        uselist=False,
        cascade="all, delete-orphan",
    )
    exports = relationship(
        "Export", back_populates="research_query", cascade="all, delete-orphan"
    )
    webhook_deliveries = relationship(
        "WebhookDelivery", back_populates="research_query", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_research_user_created", "user_id", "created_at"),
        Index("idx_research_status", "status"),
        Index("idx_research_paradigm", "primary_paradigm"),
        CheckConstraint(
            "progress >= 0 AND progress <= 100", name="check_progress_range"
        ),
    )

    @hybrid_property
    def is_complete(self):
        return self.status == ResearchStatus.COMPLETED


class ResearchSource(Base):
    __tablename__ = "research_sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    research_id = Column(
        UUID(as_uuid=True),
        ForeignKey("research_queries.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Source info
    url = Column(Text, nullable=False)
    title = Column(Text)
    domain = Column(String(255), index=True)
    author = Column(String(255))
    published_date = Column(DateTime(timezone=True))

    # Content
    content_snippet = Column(Text)
    content_hash = Column(String(64))

    # Analysis
    relevance_score = Column(Float)
    credibility_score = Column(Float)
    bias_score = Column(Float)
    paradigm_alignment = Column(JSONB, default={})

    # Metadata
    source_type = Column(String(50))  # article, academic, social, etc.
    source_metadata = Column(JSONB, default={})

    # Status
    is_analyzed = Column(Boolean, default=False)
    analysis_error = Column(Text)

    # Timestamps
    found_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    analyzed_at = Column(DateTime(timezone=True))

    # Relationships
    research_query = relationship("ResearchQuery", back_populates="sources")
    citations = relationship(
        "Citation", back_populates="source", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_source_research", "research_id"),
        Index("idx_source_relevance", "relevance_score"),
        Index("idx_source_domain", "domain"),
    )


class ResearchAnswer(Base):
    __tablename__ = "research_answers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    research_id = Column(
        UUID(as_uuid=True),
        ForeignKey("research_queries.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Summary
    executive_summary = Column(Text)
    paradigm_summary = Column(JSONB, default={})  # Per-paradigm summaries

    # Sections
    sections = Column(JSONB, default=[])  # Array of section objects
    key_insights = Column(JSONB, default=[])
    action_items = Column(JSONB, default=[])

    # Quality metrics
    synthesis_quality_score = Column(Float)
    confidence_score = Column(Float)
    completeness_score = Column(Float)

    # Generation details
    generation_model = Column(String(100))
    generation_time_ms = Column(Integer)
    token_count = Column(Integer)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    research_query = relationship("ResearchQuery", back_populates="answer")
    citations = relationship(
        "Citation", back_populates="answer", cascade="all, delete-orphan"
    )


class Citation(Base):
    __tablename__ = "citations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    answer_id = Column(
        UUID(as_uuid=True),
        ForeignKey("research_answers.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("research_sources.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Citation details
    text = Column(Text, nullable=False)
    context = Column(Text)  # Surrounding text for context
    section_index = Column(Integer)  # Which section this citation appears in
    position = Column(Integer)  # Position within section

    # Confidence
    confidence = Column(Float)

    # Relationships
    answer = relationship("ResearchAnswer", back_populates="citations")
    source = relationship("ResearchSource", back_populates="citations")

    # Indexes
    __table_args__ = (
        Index("idx_citation_answer", "answer_id"),
        Index("idx_citation_source", "source_id"),
    )


# --- Webhook Models ---


class Webhook(Base):
    __tablename__ = "webhooks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # Configuration
    url = Column(Text, nullable=False)
    secret = Column(String(255))  # For HMAC signing
    # Note: events are stored in the webhook_events_mapping association table

    # Options
    is_active = Column(Boolean, default=True, nullable=False)
    headers = Column(JSONB, default={})
    timeout = Column(Integer, default=30)
    retry_policy = Column(
        JSONB, default={"max_attempts": 3, "initial_delay": 1, "max_delay": 60}
    )

    # Statistics
    total_deliveries = Column(Integer, default=0)
    successful_deliveries = Column(Integer, default=0)
    failed_deliveries = Column(Integer, default=0)
    last_delivery_at = Column(DateTime(timezone=True))
    last_error = Column(Text)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    user = relationship("User", back_populates="webhooks")
    deliveries = relationship(
        "WebhookDelivery", back_populates="webhook", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_webhook_user", "user_id"),
        Index("idx_webhook_active", "is_active"),
    )


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    webhook_id = Column(
        UUID(as_uuid=True),
        ForeignKey("webhooks.id", ondelete="CASCADE"),
        nullable=False,
    )
    research_id = Column(
        UUID(as_uuid=True), ForeignKey("research_queries.id", ondelete="SET NULL")
    )

    # Delivery details
    event = Column(Enum(WebhookEvent), nullable=False)
    payload = Column(JSONB, nullable=False)

    # Status
    status = Column(String(20), default="pending")  # pending, success, failed, retry
    attempts = Column(Integer, default=0)

    # Response
    response_status = Column(Integer)
    response_body = Column(Text)
    response_headers = Column(JSONB)

    # Error tracking
    error = Column(Text)
    next_retry_at = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    delivered_at = Column(DateTime(timezone=True))

    # Relationships
    webhook = relationship("Webhook", back_populates="deliveries")
    research_query = relationship("ResearchQuery", back_populates="webhook_deliveries")

    # Indexes
    __table_args__ = (
        Index("idx_delivery_webhook", "webhook_id"),
        Index("idx_delivery_status", "status"),
        Index("idx_delivery_created", "created_at"),
    )


# --- Export Models ---


class Export(Base):
    __tablename__ = "exports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    research_id = Column(
        UUID(as_uuid=True),
        ForeignKey("research_queries.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Export details
    format = Column(String(20), nullable=False)  # pdf, json, csv, excel, markdown
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer)
    file_path = Column(Text)  # S3 or local path

    # Options used
    options = Column(JSONB, default={})

    # Status
    status = Column(
        String(20), default="pending"
    )  # pending, processing, completed, failed
    error_message = Column(Text)
    download_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    expires_at = Column(DateTime(timezone=True))
    last_downloaded_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="exports")
    research_query = relationship("ResearchQuery", back_populates="exports")

    # Indexes
    __table_args__ = (
        Index("idx_export_user", "user_id"),
        Index("idx_export_research", "research_id"),
        Index("idx_export_created", "created_at"),
    )


# --- Analytics Models ---


class UsageMetrics(Base):
    __tablename__ = "usage_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))

    # Time bucket (hourly aggregation)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # Metrics
    api_requests = Column(Integer, default=0)
    research_queries = Column(Integer, default=0)
    sources_analyzed = Column(Integer, default=0)
    exports_generated = Column(Integer, default=0)
    webhooks_delivered = Column(Integer, default=0)

    # Cost metrics
    tokens_used = Column(Integer, default=0)
    compute_seconds = Column(Float, default=0)

    # Paradigm distribution
    paradigm_distribution = Column(JSONB, default={})

    # Indexes
    __table_args__ = (
        UniqueConstraint("user_id", "timestamp", name="uq_user_timestamp"),
        Index("idx_metrics_user_time", "user_id", "timestamp"),
    )


# --- Session Management ---


class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # Session details
    session_token = Column(String(255), unique=True, nullable=False)
    refresh_token = Column(String(255), unique=True)

    # Device/Client info
    ip_address = Column(String(45))  # Supports IPv6
    user_agent = Column(Text)
    device_id = Column(String(255))

    # Status
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_activity = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    expires_at = Column(DateTime(timezone=True))

    # Indexes
    __table_args__ = (
        Index("idx_session_user", "user_id"),
        Index("idx_session_token", "session_token"),
        Index("idx_session_active", "is_active"),
    )
