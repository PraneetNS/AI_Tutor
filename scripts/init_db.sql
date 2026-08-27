-- =============================================================================
-- AI Tutor Production PostgreSQL Schema (Learner State + Events + Knowledge Graph)
-- =============================================================================

-- 1. Concepts & Curriculum DAG
CREATE TABLE IF NOT EXISTS concepts (
    concept_id VARCHAR(128) PRIMARY KEY,
    domain VARCHAR(64) NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS concept_prerequisites (
    concept_id VARCHAR(128) REFERENCES concepts(concept_id) ON DELETE CASCADE,
    prerequisite_id VARCHAR(128) REFERENCES concepts(concept_id) ON DELETE CASCADE,
    weight FLOAT DEFAULT 1.0,
    PRIMARY KEY (concept_id, prerequisite_id)
);

-- 2. Learner Concept Mastery (Source of truth for Bayesian Knowledge Tracing)
CREATE TABLE IF NOT EXISTS learner_concept_mastery (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    domain VARCHAR(64) NOT NULL DEFAULT 'machine_learning',
    concept_id VARCHAR(128) NOT NULL REFERENCES concepts(concept_id),
    mastery FLOAT NOT NULL DEFAULT 0.3,       -- P(L) posterior from BKT
    confidence FLOAT NOT NULL DEFAULT 0.5,    -- Observation sample confidence
    attempts INT NOT NULL DEFAULT 0,
    correct INT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, domain, concept_id)
);
CREATE INDEX IF NOT EXISTS idx_mastery_user ON learner_concept_mastery (user_id, domain);

-- 3. Learner Misconceptions Tracking
CREATE TABLE IF NOT EXISTS learner_misconceptions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    misconception_id VARCHAR(128) NOT NULL,
    description TEXT,
    confidence FLOAT NOT NULL DEFAULT 0.3,
    concepts_affected TEXT[] NOT NULL DEFAULT '{}',
    is_resolved BOOLEAN NOT NULL DEFAULT FALSE,
    first_detected TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_detected TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, misconception_id)
);
CREATE INDEX IF NOT EXISTS idx_misconception_user ON learner_misconceptions (user_id);

-- 4. Learning Sessions & Memory
CREATE TABLE IF NOT EXISTS learning_sessions (
    session_id VARCHAR(128) PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    course_id INT,
    lecture_id INT,
    pedagogy_mode VARCHAR(32) NOT NULL DEFAULT 'socratic',
    hint_level INT NOT NULL DEFAULT 0,
    is_stuck BOOLEAN NOT NULL DEFAULT FALSE,
    current_topic VARCHAR(128),
    turn_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 5. Interaction Event Store (Append-only telemetry)
CREATE TABLE IF NOT EXISTS interaction_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(128) REFERENCES learning_sessions(session_id),
    user_id VARCHAR(128) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_user ON interaction_events (user_id, created_at DESC);
