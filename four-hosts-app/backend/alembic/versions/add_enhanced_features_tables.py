"""Add enhanced features tables

Revision ID: add_enhanced_features
Revises: latest
Create Date: 2025-01-30

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = 'add_enhanced_features'
down_revision = 'add_api_key_index'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = inspect(bind)

    if 'user_feedback' not in inspector.get_table_names():
        op.create_table('user_feedback',
            sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('research_id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('satisfaction_score', sa.Float(), nullable=False),
            sa.Column('paradigm_feedback', sa.String(length=50), nullable=True),
            sa.Column('comments', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(['research_id'], ['research_queries.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.CheckConstraint('satisfaction_score >= 0 AND satisfaction_score <= 1', name='check_satisfaction_range')
        )
        op.create_index('idx_feedback_research', 'user_feedback', ['research_id'], unique=False)
        op.create_index('idx_feedback_user', 'user_feedback', ['user_id'], unique=False)

    if 'paradigm_performance' not in inspector.get_table_names():
        op.create_table('paradigm_performance',
            sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('paradigm', sa.Enum('dolores', 'teddy', 'bernard', 'maeve', name='paradigmtype'), nullable=False),
            sa.Column('total_queries', sa.Integer(), nullable=True),
            sa.Column('successful_queries', sa.Integer(), nullable=True),
            sa.Column('failed_queries', sa.Integer(), nullable=True),
            sa.Column('avg_confidence_score', sa.Float(), nullable=True),
            sa.Column('avg_synthesis_quality', sa.Float(), nullable=True),
            sa.Column('avg_user_satisfaction', sa.Float(), nullable=True),
            sa.Column('avg_response_time', sa.Float(), nullable=True),
            sa.Column('window_start', sa.DateTime(timezone=True), nullable=False),
            sa.Column('window_end', sa.DateTime(timezone=True), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('paradigm', 'window_start', 'window_end', name='unique_paradigm_window')
        )
        op.create_index('idx_paradigm_performance_paradigm', 'paradigm_performance', ['paradigm'], unique=False)
        op.create_index('idx_paradigm_performance_window', 'paradigm_performance', ['window_start', 'window_end'], unique=False)

    if 'ml_training_data' not in inspector.get_table_names():
        op.create_table('ml_training_data',
            sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('query_id', sa.String(length=100), nullable=False),
            sa.Column('query_text', sa.Text(), nullable=False),
            sa.Column('query_features', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column('true_paradigm', sa.Enum('dolores', 'teddy', 'bernard', 'maeve', name='paradigmtype'), nullable=False),
            sa.Column('predicted_paradigm', sa.Enum('dolores', 'teddy', 'bernard', 'maeve', name='paradigmtype'), nullable=False),
            sa.Column('confidence_score', sa.Float(), nullable=True),
            sa.Column('user_satisfaction', sa.Float(), nullable=True),
            sa.Column('synthesis_quality', sa.Float(), nullable=True),
            sa.Column('used_for_training', sa.Boolean(), nullable=True),
            sa.Column('model_version', sa.String(length=50), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('idx_ml_training_created', 'ml_training_data', ['created_at'], unique=False)
        op.create_index('idx_ml_training_paradigm', 'ml_training_data', ['true_paradigm'], unique=False)
        op.create_index('idx_ml_training_used', 'ml_training_data', ['used_for_training'], unique=False)


def downgrade():
    op.drop_index('idx_ml_training_used', table_name='ml_training_data')
    op.drop_index('idx_ml_training_paradigm', table_name='ml_training_data')
    op.drop_index('idx_ml_training_created', table_name='ml_training_data')
    op.drop_table('ml_training_data')
    
    op.drop_index('idx_paradigm_performance_window', table_name='paradigm_performance')
    op.drop_index('idx_paradigm_performance_paradigm', table_name='paradigm_performance')
    op.drop_table('paradigm_performance')
    
    op.drop_index('idx_feedback_user', table_name='user_feedback')
    op.drop_index('idx_feedback_research', table_name='user_feedback')
    op.drop_table('user_feedback')
