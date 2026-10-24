CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS entity_embedding (
  entity_id     TEXT PRIMARY KEY,
  entity_type   TEXT NOT NULL CHECK (entity_type IN ('skill', 'job')),
  name          TEXT NOT NULL,
  embedding     vector(1024) NOT NULL,
  model_version TEXT NOT NULL,
  created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS entity_embedding_hnsw ON entity_embedding USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS embedding_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS posting_entity (
  posting_id    TEXT NOT NULL,
  month         DATE NOT NULL,
  entity_id     TEXT NOT NULL REFERENCES entity_embedding (entity_id),
  entity_type   TEXT NOT NULL,
  mention       TEXT NOT NULL,
  score         REAL NOT NULL,
  confidence    TEXT NOT NULL CHECK (confidence IN ('high', 'low')), 
  model_version TEXT NOT NULL,
  PRIMARY KEY (posting_id, entity_id, mention)
);
CREATE INDEX IF NOT EXISTS posting_entity_month ON posting_entity (month, entity_id);

CREATE TABLE IF NOT EXISTS unmapped_mention (
  mention        TEXT NOT NULL,
  entity_type    TEXT NOT NULL,
  best_entity_id TEXT,
  best_score     REAL,
  occurrences    INTEGER NOT NULL DEFAULT 1,
  first_seen     DATE,
  last_seen      DATE,
  model_version  TEXT NOT NULL,
  PRIMARY KEY (mention, entity_type, model_version)
);

-- Step 6: job<->skill graph for explanation and matching only; never feeds trend numbers.
-- cooc_* is measured evidence from postings; embed_sim is a fallback for thin jobs. Kept separate on purpose.
CREATE TABLE IF NOT EXISTS job_skill_edge (
  job_id        TEXT NOT NULL REFERENCES entity_embedding (entity_id),
  skill_id      TEXT NOT NULL REFERENCES entity_embedding (entity_id),
  cooc_postings INTEGER NOT NULL DEFAULT 0,   -- postings for this job that mention this skill
  job_postings  INTEGER NOT NULL DEFAULT 0,   -- all postings for this job
  cooc_weight   REAL,                         -- cooc_postings / job_postings; NULL when no co-occurrence
  embed_sim     REAL,                         -- cosine(job vector, skill vector); NULL outside the per-job top-k
  model_version TEXT NOT NULL,
  PRIMARY KEY (job_id, skill_id)
);

-- Written by the LightGBM forecaster, read by matching (Step 8). Embeddings code never writes here.
-- growth = forecast relative change over the horizon, e.g. 0.12 = +12% postings.
CREATE TABLE IF NOT EXISTS entity_forecast (
  entity_id      TEXT NOT NULL,
  horizon_months INTEGER NOT NULL,
  growth         REAL NOT NULL,
  model_version  TEXT NOT NULL,
  created_at     TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (entity_id, horizon_months, model_version)
);
