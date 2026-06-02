CREATE TABLE IF NOT EXISTS documents (
    document_id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    s3_bucket TEXT NOT NULL,
    s3_key TEXT NOT NULL,
    overall_status TEXT NOT NULL DEFAULT 'uploaded',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (s3_bucket, s3_key)
);

CREATE TABLE IF NOT EXISTS processing_runs (
    run_id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    strategy TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    chunk_count INTEGER,
    average_chunk_length DOUBLE PRECISION,
    processing_time_seconds DOUBLE PRECISION,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    UNIQUE (document_id, strategy)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES processing_runs(run_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS retrieval_queries (
    query_id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    result_id BIGSERIAL PRIMARY KEY,
    query_id BIGINT NOT NULL REFERENCES retrieval_queries(query_id) ON DELETE CASCADE,
    run_id BIGINT NOT NULL REFERENCES processing_runs(run_id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    chunk_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_processing_runs_document_id
    ON processing_runs (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_run_id
    ON chunks (run_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_query_id
    ON retrieval_results (query_id);
