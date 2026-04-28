-- =============================================================================
-- HeteroRAG POC — PostgreSQL Bootstrap Schema
-- File: sql/init/00_bootstrap.sql
-- Runs via docker-entrypoint-initdb.d on first container start.
-- Flyway migrations (V1__, V2__...) run afterwards to populate ground truth views.
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
    about_me        TEXT,           -- kept here for reference; canonical copy in ES
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
    accepted_answer_id  INTEGER,                    -- FK to posts.id (nullable)
    parent_id           INTEGER,                    -- non-null for answers
    score               INTEGER         NOT NULL DEFAULT 0,
    view_count          INTEGER         NOT NULL DEFAULT 0,
    answer_count        INTEGER         NOT NULL DEFAULT 0,
    comment_count       INTEGER         NOT NULL DEFAULT 0,
    owner_user_id       INTEGER         REFERENCES users(id) ON DELETE SET NULL,
    last_editor_user_id INTEGER,
    creation_date       TIMESTAMPTZ,
    last_edit_date      TIMESTAMPTZ,
    last_activity_date  TIMESTAMPTZ,
    tags                TEXT,           -- raw pipe-delimited tag string e.g. <python><pandas>
    closed_date         TIMESTAMPTZ,
    title               TEXT
    -- body intentionally excluded — canonical copy in Elasticsearch
);

CREATE INDEX idx_posts_owner ON posts (owner_user_id);
CREATE INDEX idx_posts_type ON posts (post_type_id);
CREATE INDEX idx_posts_score ON posts (score DESC);
CREATE INDEX idx_posts_creation ON posts (creation_date);
CREATE INDEX idx_posts_parent ON posts (parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX idx_posts_tags_trgm ON posts USING GIN (tags gin_trgm_ops);

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
    count           INTEGER         NOT NULL DEFAULT 0,
    excerpt_post_id INTEGER,        -- FK to posts.id (nullable)
    wiki_post_id    INTEGER         -- FK to posts.id (nullable)
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

CREATE TABLE IF NOT EXISTS post_links (
    id                  INTEGER PRIMARY KEY,
    creation_date       TIMESTAMPTZ,
    post_id             INTEGER         NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    related_post_id     INTEGER         NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    link_type_id        SMALLINT        NOT NULL    -- 1=Linked, 3=Duplicate
);

CREATE INDEX idx_post_links_post ON post_links (post_id);
CREATE INDEX idx_post_links_related ON post_links (related_post_id);
CREATE INDEX idx_post_links_type ON post_links (link_type_id);

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
