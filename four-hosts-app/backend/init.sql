-- Four Hosts Research Database Schema Initialization
-- Version: 2.0
-- Last Updated: 2025-08-27

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create enum types
DO $$ BEGIN
    CREATE TYPE user_role AS ENUM ('free', 'basic', 'pro', 'enterprise', 'admin');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE auth_provider AS ENUM ('local', 'google', 'github', 'saml');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE paradigm_type AS ENUM ('dolores', 'teddy', 'bernard', 'maeve');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE research_depth AS ENUM ('quick', 'standard', 'deep');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE research_status AS ENUM ('queued', 'processing', 'in_progress', 'completed', 'failed', 'cancelled');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE webhook_event AS ENUM ('research.started', 'research.progress', 'research.completed', 'research.failed', 'research.cancelled', 'classification.completed', 'synthesis.completed', 'export.ready');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;


-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    role user_role NOT NULL DEFAULT 'free',
    auth_provider auth_provider NOT NULL DEFAULT 'local',
    full_name VARCHAR(255),
    avatar_url VARCHAR(500),
    bio TEXT,
    preferences JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT true,
    is_verified BOOLEAN NOT NULL DEFAULT false,
    verification_token VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_user_email_provider ON users(email, auth_provider);
CREATE INDEX IF NOT EXISTS idx_user_created ON users(created_at);

-- Create api_keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    role user_role NOT NULL,
    allowed_origins TEXT[] DEFAULT '{}',
    allowed_ips TEXT[] DEFAULT '{}',
    rate_limit_tier VARCHAR(50) DEFAULT 'standard',
    last_used TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_api_key_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_key_active ON api_keys(is_active);


-- Create research_queries table
CREATE TABLE IF NOT EXISTS research_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64),
    language VARCHAR(10) DEFAULT 'en',
    region VARCHAR(10),
    primary_paradigm paradigm_type NOT NULL,
    secondary_paradigm paradigm_type,
    paradigm_scores JSONB DEFAULT '{}'::jsonb,
    classification_confidence FLOAT,
    paradigm_override paradigm_type,
    depth research_depth NOT NULL DEFAULT 'standard',
    max_sources INTEGER DEFAULT 100,
    include_secondary BOOLEAN DEFAULT true,
    custom_prompts JSONB DEFAULT '{}'::jsonb,
    status research_status NOT NULL DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    current_phase VARCHAR(50),
    error_message TEXT,
    sources_found INTEGER DEFAULT 0,
    sources_analyzed INTEGER DEFAULT 0,
    synthesis_score FLOAT,
    confidence_score FLOAT,
    duration_seconds FLOAT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT check_progress_range CHECK (progress >= 0 AND progress <= 100)
);

CREATE INDEX IF NOT EXISTS idx_research_user_created ON research_queries(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_research_status ON research_queries(status);
CREATE INDEX IF NOT EXISTS idx_research_paradigm ON research_queries(primary_paradigm);
CREATE INDEX IF NOT EXISTS idx_research_query_hash ON research_queries(query_hash);

-- Create research_sources table
CREATE TABLE IF NOT EXISTS research_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    research_id UUID NOT NULL REFERENCES research_queries(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    domain VARCHAR(255),
    author VARCHAR(255),
    published_date TIMESTAMP WITH TIME ZONE,
    content_snippet TEXT,
    content_hash VARCHAR(64),
    relevance_score FLOAT,
    credibility_score FLOAT,
    bias_score FLOAT,
    paradigm_alignment JSONB DEFAULT '{}'::jsonb,
    source_type VARCHAR(50),
    source_metadata JSONB DEFAULT '{}'::jsonb,
    is_analyzed BOOLEAN DEFAULT false,
    analysis_error TEXT,
    found_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    analyzed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_source_research ON research_sources(research_id);
CREATE INDEX IF NOT EXISTS idx_source_relevance ON research_sources(relevance_score);
CREATE INDEX IF NOT EXISTS idx_source_domain ON research_sources(domain);

-- Create research_answers table
CREATE TABLE IF NOT EXISTS research_answers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    research_id UUID NOT NULL UNIQUE REFERENCES research_queries(id) ON DELETE CASCADE,
    executive_summary TEXT,
    paradigm_summary JSONB DEFAULT '{}'::jsonb,
    sections JSONB DEFAULT '[]'::jsonb,
    key_insights JSONB DEFAULT '[]'::jsonb,
    action_items JSONB DEFAULT '[]'::jsonb,
    synthesis_quality_score FLOAT,
    confidence_score FLOAT,
    completeness_score FLOAT,
    generation_model VARCHAR(100),
    generation_time_ms INTEGER,
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create citations table
CREATE TABLE IF NOT EXISTS citations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    answer_id UUID NOT NULL REFERENCES research_answers(id) ON DELETE CASCADE,
    source_id UUID NOT NULL REFERENCES research_sources(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    context TEXT,
    section_index INTEGER,
    position INTEGER,
    confidence FLOAT
);

CREATE INDEX IF NOT EXISTS idx_citation_answer ON citations(answer_id);
CREATE INDEX IF NOT EXISTS idx_citation_source ON citations(source_id);

-- Create webhooks table
CREATE TABLE IF NOT EXISTS webhooks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    secret VARCHAR(255),
    is_active BOOLEAN NOT NULL DEFAULT true,
    headers JSONB DEFAULT '{}'::jsonb,
    timeout INTEGER DEFAULT 30,
    retry_policy JSONB DEFAULT '{"max_attempts": 3, "initial_delay": 1, "max_delay": 60}'::jsonb,
    total_deliveries INTEGER DEFAULT 0,
    successful_deliveries INTEGER DEFAULT 0,
    failed_deliveries INTEGER DEFAULT 0,
    last_delivery_at TIMESTAMP WITH TIME ZONE,
    last_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_webhook_user ON webhooks(user_id);
CREATE INDEX IF NOT EXISTS idx_webhook_active ON webhooks(is_active);

-- Create webhook_events_mapping table
CREATE TABLE IF NOT EXISTS webhook_events_mapping (
    webhook_id UUID NOT NULL REFERENCES webhooks(id) ON DELETE CASCADE,
    event webhook_event NOT NULL
);

-- Create user_saved_searches table
CREATE TABLE IF NOT EXISTS user_saved_searches (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    research_id UUID NOT NULL REFERENCES research_queries(id) ON DELETE CASCADE,
    saved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tags TEXT[] DEFAULT '{}',
    notes TEXT
);

-- Create webhook_deliveries table
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    webhook_id UUID NOT NULL REFERENCES webhooks(id) ON DELETE CASCADE,
    research_id UUID REFERENCES research_queries(id) ON DELETE SET NULL,
    event webhook_event NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    response_status INTEGER,
    response_body TEXT,
    response_headers JSONB DEFAULT '{}'::jsonb,
    error TEXT,
    next_retry_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_delivery_webhook ON webhook_deliveries(webhook_id);
CREATE INDEX IF NOT EXISTS idx_delivery_status ON webhook_deliveries(status);
CREATE INDEX IF NOT EXISTS idx_delivery_created ON webhook_deliveries(created_at);

-- Create exports table
CREATE TABLE IF NOT EXISTS exports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    research_id UUID NOT NULL REFERENCES research_queries(id) ON DELETE CASCADE,
    format VARCHAR(20) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_size INTEGER,
    file_path TEXT,
    options JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    download_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_downloaded_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_export_user ON exports(user_id);
CREATE INDEX IF NOT EXISTS idx_export_research ON exports(research_id);
CREATE INDEX IF NOT EXISTS idx_export_created ON exports(created_at);

-- Create usage_metrics table
CREATE TABLE IF NOT EXISTS usage_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    api_requests INTEGER DEFAULT 0,
    research_queries INTEGER DEFAULT 0,
    sources_analyzed INTEGER DEFAULT 0,
    exports_generated INTEGER DEFAULT 0,
    webhooks_delivered INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    compute_seconds FLOAT DEFAULT 0,
    paradigm_distribution JSONB DEFAULT '{}'::jsonb,
    UNIQUE(user_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_metrics_user_time ON usage_metrics(user_id, timestamp);

-- Create user_sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address VARCHAR(45),
    user_agent TEXT,
    device_id VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_session_user ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_session_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_session_active ON user_sessions(is_active);

-- Create user_feedback table
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    research_id UUID NOT NULL REFERENCES research_queries(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    satisfaction_score FLOAT NOT NULL,
    paradigm_feedback VARCHAR(50),
    comments TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_satisfaction_range CHECK (satisfaction_score >= 0 AND satisfaction_score <= 1)
);

CREATE INDEX IF NOT EXISTS idx_feedback_research ON user_feedback(research_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback(user_id);

-- Create paradigm_performance table
CREATE TABLE IF NOT EXISTS paradigm_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paradigm paradigm_type NOT NULL,
    total_queries INTEGER DEFAULT 0,
    successful_queries INTEGER DEFAULT 0,
    failed_queries INTEGER DEFAULT 0,
    avg_confidence_score FLOAT DEFAULT 0.0,
    avg_synthesis_quality FLOAT DEFAULT 0.0,
    avg_user_satisfaction FLOAT DEFAULT 0.0,
    avg_response_time FLOAT DEFAULT 0.0,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paradigm, window_start, window_end)
);

CREATE INDEX IF NOT EXISTS idx_paradigm_performance_paradigm ON paradigm_performance(paradigm);
CREATE INDEX IF NOT EXISTS idx_paradigm_performance_window ON paradigm_performance(window_start, window_end);

-- Create ml_training_data table
CREATE TABLE IF NOT EXISTS ml_training_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id VARCHAR(100) NOT NULL,
    query_text TEXT NOT NULL,
    query_features JSONB NOT NULL,
    true_paradigm paradigm_type NOT NULL,
    predicted_paradigm paradigm_type NOT NULL,
    confidence_score FLOAT,
    user_satisfaction FLOAT,
    synthesis_quality FLOAT,
    used_for_training BOOLEAN DEFAULT false,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ml_training_paradigm ON ml_training_data(true_paradigm);
CREATE INDEX IF NOT EXISTS idx_ml_training_used ON ml_training_data(used_for_training);
CREATE INDEX IF NOT EXISTS idx_ml_training_created ON ml_training_data(created_at);


-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_research_answers_updated_at ON research_answers;
CREATE TRIGGER update_research_answers_updated_at BEFORE UPDATE ON research_answers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_webhooks_updated_at ON webhooks;
CREATE TRIGGER update_webhooks_updated_at BEFORE UPDATE ON webhooks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;

-- Insert default admin user (password: Admin123!@#)
-- Password hash generated with bcrypt
INSERT INTO users (email, username, password_hash, role, is_active, is_verified, auth_provider)
VALUES (
    'admin@fourhosts.com',
    'admin',
    '$2b$12$YourHashHere', -- This will need to be generated properly
    'admin',
    true,
    true,
    'local'
) ON CONFLICT (email) DO NOTHING;
