"""Baseline schema created from SQLAlchemy models

Revision ID: 20250827_initial_schema
Revises: 
Create Date: 2025-09-18
"""

from alembic import op

from database.models import Base

# revision identifiers, used by Alembic.
revision = "20250827_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
