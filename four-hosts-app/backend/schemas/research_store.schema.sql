-- Research Store Schema (Postgres + optional pgvector)
-- Entities: source, chunk, embedding, claim, evidence_link, run, tool_call, lineage
-- Enable pgvector if available
-- CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS source (
  id UUID PRIMARY KEY,
  url TEXT NOT NULL,
  title TEXT,
  authors TEXT[],
  tld TEXT,
  published_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  parser TEXT,
  mime_type TEXT,
  authority_scores JSONB,                -- e.g., { "tld_bonus": 0.2, "backlinks": 0.5 }
  metadata JSONB                         -- arbitrary extra fields
);

CREATE INDEX IF NOT EXISTS idx_source_url ON source (url);
CREATE INDEX IF NOT EXISTS idx_source_fetched_at ON source (fetched_at);

CREATE TABLE IF NOT EXISTS chunk (
  id UUID PRIMARY KEY,
  source_id UUID REFERENCES source(id) ON DELETE CASCADE,
  section TEXT,                          -- e.g., "Abstract", "Methods", "H1: Introduction"
  position INTEGER,                      -- order within source
  text TEXT NOT NULL,
  token_count INTEGER,
  md5_hash CHAR(32) NOT NULL,
  simhash BIGINT,                        -- 64-bit simhash
  credibility_features JSONB,            -- per-chunk features if available
  metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunk (source_id);
CREATE INDEX IF NOT EXISTS idx_chunk_md5 ON chunk (md5_hash);
CREATE INDEX IF NOT EXISTS idx_chunk_simhash ON chunk (simhash);

-- Optional pgvector embeddings table (commented vector type if extension not installed)
CREATE TABLE IF NOT EXISTS embedding (
  id UUID PRIMARY KEY,
  chunk_id UUID REFERENCES chunk(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  -- embedding VECTOR(1536),              -- requires pgvector; if unavailable use BYTEA
  embedding BYTEA,                        -- fallback storage for embedding
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embedding_chunk ON embedding (chunk_id);
CREATE INDEX IF NOT EXISTS idx_embedding_model ON embedding (model);

CREATE TABLE IF NOT EXISTS claim (
  id UUID PRIMARY KEY,
  text TEXT NOT NULL,
  uncertainty REAL CHECK (uncertainty BETWEEN 0 AND 1),
  adjudication_notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Link table between claims and evidence chunks
CREATE TABLE IF NOT EXISTS evidence_link (
  id UUID PRIMARY KEY,
  claim_id UUID REFERENCES claim(id) ON DELETE CASCADE,
  source_id UUID REFERENCES source(id) ON DELETE CASCADE,
  chunk_id UUID REFERENCES chunk(id) ON DELETE CASCADE,
  span JSONB,                             -- e.g., {"start": 120, "end": 240}
  credibility_features JSONB,
  composite_score REAL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ev_claim ON evidence_link (claim_id);
CREATE INDEX IF NOT EXISTS idx_ev_source ON evidence_link (source_id);
CREATE INDEX IF NOT EXISTS idx_ev_chunk ON evidence_link (chunk_id);

CREATE TABLE IF NOT EXISTS run (
  id UUID PRIMARY KEY,
  intake_id UUID,                         -- external reference to intake object
  planner JSONB,                          -- plan, checkpoints, stopping conditions
  budgets JSONB,                          -- {max_tokens, max_cost_usd, max_wallclock_minutes}
  costs JSONB,                            -- observed costs
  status TEXT,                            -- e.g., "running", "completed", "failed"
  started_at TIMESTAMPTZ DEFAULT NOW(),
  finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_run_status ON run (status);
CREATE INDEX IF NOT EXISTS idx_run_started ON run (started_at);

CREATE TABLE IF NOT EXISTS tool_call (
  id UUID PRIMARY KEY,
  run_id UUID REFERENCES run(id) ON DELETE CASCADE,
  tool TEXT NOT NULL,
  input JSONB,
  output JSONB,
  error JSONB,
  retry_count INTEGER DEFAULT 0,
  latency_ms INTEGER,
  cost_tokens INTEGER,
  cost_usd NUMERIC(12,6),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_call_run ON tool_call (run_id);
CREATE INDEX IF NOT EXISTS idx_tool_call_tool ON tool_call (tool);

-- Lineage records how artifacts derive from others
CREATE TABLE IF NOT EXISTS lineage (
  id UUID PRIMARY KEY,
  parent_type TEXT,                       -- "source" | "chunk" | "claim" | "report" | etc.
  parent_id UUID,
  child_type TEXT,
  child_id UUID,
  relation TEXT,                          -- "derived_from", "supports", "contradicts"
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage (parent_type, parent_id);
CREATE INDEX IF NOT EXISTS idx_lineage_child ON lineage (child_type, child_id);

-- Simple view for bibliography rendering
CREATE OR REPLACE VIEW v_bibliography AS
SELECT
  s.id AS source_id,
  s.title,
  s.authors,
  s.url,
  s.published_at,
  s.updated_at,
  s.metadata
FROM source s;