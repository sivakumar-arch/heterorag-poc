-- =============================================================================
-- HeteroRAG POC — PostgreSQL Bootstrap Schema
-- File: sql/init/00_bootstrap.sql
-- Runs via docker-entrypoint-initdb.d on first container start.
-- Data ownership rule (decision D2, D10): this store holds ATTRIBUTES only.
--   Relationships (authorship, parent/accepted answer, post tags, post links) live in
--   the graph service; text (titles, bodies, about-me, comment text) lives in the
--   document service. Identifiers are the only shared keys. Event records (votes,
--   badges, comments) keep the post and user ids they refer to.
-- There are no Flyway migrations: the old ground-truth views were superseded by the
-- XML-derived oracle (benchmark_v2/oracle.py).
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- trigram similarity for text search
CREATE EXTENSION IF NOT EXISTS btree_gin;    -- GIN index support for composite types

-- =============================================================================
-- Core tables — matching Stack Exchange XML schema exactly
-- =============================================================================

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY,
    reputation      INTEGER         NOT NULL DEFAULT 0,
    creation_date   TIMESTAMPTZ,
    display_name    TEXT,
    location        TEXT,
    up_votes        INTEGER         NOT NULL DEFAULT 0,
    down_votes      INTEGER         NOT NULL DEFAULT 0,
    views           INTEGER         NOT NULL DEFAULT 0,
    account_id      INTEGER
);

CREATE INDEX idx_users_reputation ON users (reputation DESC);
CREATE INDEX idx_users_creation_date ON users (creation_date);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS posts (
    id                  INTEGER PRIMARY KEY,
    post_type_id        SMALLINT        NOT NULL,   -- 1=Question, 2=Answer
    score               INTEGER         NOT NULL DEFAULT 0,
    view_count          INTEGER         NOT NULL DEFAULT 0,
    answer_count        INTEGER         NOT NULL DEFAULT 0,
    comment_count       INTEGER         NOT NULL DEFAULT 0,
    creation_date       TIMESTAMPTZ,
    last_edit_date      TIMESTAMPTZ,
    last_activity_date  TIMESTAMPTZ,
    closed_date         TIMESTAMPTZ
    -- Deliberately absent (owned by other services): owner, parent, accepted answer,
    -- tags (graph); title and body (document). The last editor is not loaded anywhere
    -- and is outside the virtual union (see Benchmark v2 spec, section 4).
);

CREATE INDEX idx_posts_type ON posts (post_type_id);
CREATE INDEX idx_posts_score ON posts (score DESC);
CREATE INDEX idx_posts_creation ON posts (creation_date);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS votes (
    id              INTEGER PRIMARY KEY,
    post_id         INTEGER         NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    vote_type_id    SMALLINT        NOT NULL,   -- 1=AcceptedByOriginator, 2=UpMod, 3=DownMod, etc.
    creation_date   TIMESTAMPTZ,
    user_id         INTEGER         REFERENCES users(id) ON DELETE SET NULL,
    bounty_amount   INTEGER
);

CREATE INDEX idx_votes_post ON votes (post_id);
CREATE INDEX idx_votes_type ON votes (vote_type_id);
CREATE INDEX idx_votes_user ON votes (user_id) WHERE user_id IS NOT NULL;

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS badges (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT            NOT NULL,
    class           SMALLINT        NOT NULL,   -- 1=Gold, 2=Silver, 3=Bronze
    tag_based       BOOLEAN         NOT NULL DEFAULT FALSE,
    date            TIMESTAMPTZ
);

CREATE INDEX idx_badges_user ON badges (user_id);
CREATE INDEX idx_badges_name ON badges (name);
CREATE INDEX idx_badges_class ON badges (class);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tags (
    id              INTEGER PRIMARY KEY,
    tag_name        TEXT            NOT NULL UNIQUE,
    count           INTEGER         NOT NULL DEFAULT 0
    -- Which questions carry a tag is a relationship (graph); wiki text is document.
);

CREATE INDEX idx_tags_name ON tags (tag_name);
CREATE INDEX idx_tags_count ON tags (count DESC);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS comments (
    id              INTEGER PRIMARY KEY,
    post_id         INTEGER         NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    score           INTEGER         NOT NULL DEFAULT 0,
    user_id         INTEGER         REFERENCES users(id) ON DELETE SET NULL,
    creation_date   TIMESTAMPTZ
    -- text intentionally excluded — canonical copy in Elasticsearch
);

CREATE INDEX idx_comments_post ON comments (post_id);
CREATE INDEX idx_comments_user ON comments (user_id) WHERE user_id IS NOT NULL;

-- ---------------------------------------------------------------------------

-- =============================================================================
-- Loader progress tracking — lets the loader script resume on failure
-- =============================================================================

CREATE TABLE IF NOT EXISTS _load_progress (
    file_name       TEXT PRIMARY KEY,
    rows_loaded     INTEGER         NOT NULL DEFAULT 0,
    completed_at    TIMESTAMPTZ
);

-- =============================================================================
-- Grants — future microservice users can be added here
-- =============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO heterorag;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO heterorag;
