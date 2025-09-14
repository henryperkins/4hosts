"""Add API key index for efficient lookups

Revision ID: add_api_key_index
Revises: latest
Create Date: 2025-01-14

This migration adds a key_index column to the api_keys table for efficient
constant-time lookups, preventing the race condition vulnerability in API key validation.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text
import hashlib


# Revision identifiers
revision = 'add_api_key_index'
down_revision = None  # Update this to the latest revision in your migrations
branch_labels = None
depends_on = None


def compute_key_index(key_hash: str) -> str:
    """Compute index from first part of key hash for migration."""
    # Since we only have the hash, use part of it for indexing
    # In production, this would be computed from the actual key
    return hashlib.sha256(key_hash[:16].encode()).hexdigest()[:16]


def upgrade():
    """Add key_index column and populate it."""

    # Add the new column (nullable initially for existing data)
    op.add_column(
        'api_keys',
        sa.Column('key_index', sa.String(16), nullable=True)
    )

    # Create index for efficient lookups
    op.create_index(
        'ix_api_keys_key_index',
        'api_keys',
        ['key_index'],
        unique=False
    )

    # Populate key_index for existing records
    # Note: In production, you'd compute this from actual keys during rotation
    connection = op.get_bind()
    result = connection.execute(text("SELECT id, key_hash FROM api_keys"))

    for row in result:
        key_index = compute_key_index(row.key_hash)
        connection.execute(
            text("UPDATE api_keys SET key_index = :index WHERE id = :id"),
            {"index": key_index, "id": row.id}
        )

    # Make column non-nullable after population
    op.alter_column(
        'api_keys',
        'key_index',
        nullable=False
    )

    # Add composite index for even faster lookups
    op.create_index(
        'ix_api_keys_key_index_active',
        'api_keys',
        ['key_index', 'is_active'],
        unique=False
    )


def downgrade():
    """Remove key_index column and indexes."""

    # Drop indexes
    op.drop_index('ix_api_keys_key_index_active', table_name='api_keys')
    op.drop_index('ix_api_keys_key_index', table_name='api_keys')

    # Drop column
    op.drop_column('api_keys', 'key_index')