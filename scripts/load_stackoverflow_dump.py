#!/usr/bin/env python3
"""
HeteroRAG POC — Stack Overflow Data Loader
===========================================
RUNBOOK Step 2: ./scripts/load_stackoverflow_dump.sh calls this script.

Reads Stack Exchange XML dump files and loads them into all three services:
  - PostgreSQL  (Service 1: Users, Posts, Votes, Badges, Tags, Comments, PostLinks)
  - Neo4j       (Service 2: User/Question/Answer/Tag nodes + all edges)
  - Elasticsearch (Service 3: Question/Answer bodies, Comments, AboutMe, Tag wikis)

Design principles:
  - Streaming XML parse (iterparse) — never loads a full file into memory
  - Idempotent via UPSERT / index create-if-not-exists / node MERGE
  - Resumable — tracks progress in PostgreSQL _load_progress table
  - Batched writes with configurable batch size
  - Dry-run mode for validation without writing

Usage:
  python scripts/load_stackoverflow_dump.py --data-dir ./data
  python scripts/load_stackoverflow_dump.py --data-dir ./data --dry-run
  python scripts/load_stackoverflow_dump.py --data-dir ./data --only postgres
  python scripts/load_stackoverflow_dump.py --data-dir ./data --batch-size 2000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

import psycopg2
import psycopg2.extras
from elasticsearch import Elasticsearch, helpers as es_helpers
from neo4j import GraphDatabase

# =============================================================================
# Configuration
# =============================================================================

PG_DSN = os.getenv(
    "PG_DSN",
    "host=localhost port=5432 dbname=heterorag user=heterorag password=heterorag_secret",
)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "heterorag_secret")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")

DEFAULT_BATCH_SIZE = 1000
LOG_EVERY = 50_000       # log progress every N rows parsed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("heterorag.loader")

# =============================================================================
# Helpers
# =============================================================================

def parse_date(s: str | None) -> datetime | None:
    """Parse ISO-8601 date string from Stack Exchange XML."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.rstrip("Z")).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def safe_int(s: str | None, default: int = 0) -> int:
    try:
        return int(s) if s is not None else default
    except (ValueError, TypeError):
        return default


def safe_bool(s: str | None) -> bool:
    return (s or "").strip().lower() in ("1", "true", "yes")


def iterparse_rows(xml_path: Path) -> Iterator[dict[str, str]]:
    """Memory-efficient streaming parse of a Stack Exchange XML dump file.
    Yields attribute dicts for every <row> element."""
    context = ET.iterparse(str(xml_path), events=("end",))
    for event, elem in context:
        if elem.tag == "row":
            yield dict(elem.attrib)
            elem.clear()


@contextmanager
def pg_connection(dsn: str):
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def disable_fk_constraints(pg_conn):
    """
    Disable all FK constraint triggers for the duration of the bulk load.
    Re-enable with enable_fk_constraints() after loading completes.
    This is safe because the data is self-consistent within each XML dump —
    FK violations only arise from cross-table ordering issues during load.
    """
    with pg_conn.cursor() as cur:
        for table in ["votes", "badges", "comments", "post_links", "posts"]:
            cur.execute(f"ALTER TABLE {table} DISABLE TRIGGER ALL")
    pg_conn.commit()
    log.info("FK constraints disabled for bulk load")


def enable_fk_constraints(pg_conn):
    """Re-enable FK constraints after bulk load completes.
    Rolls back any aborted transaction first so the ALTER TABLE succeeds."""
    try:
        pg_conn.rollback()   # clear any aborted transaction state
    except Exception:
        pass
    with pg_conn.cursor() as cur:
        for table in ["votes", "badges", "comments", "post_links", "posts"]:
            cur.execute(f"ALTER TABLE {table} ENABLE TRIGGER ALL")
    pg_conn.commit()
    log.info("FK constraints re-enabled")


def mark_complete(pg_conn, file_name: str, rows_loaded: int, community: str = ""):
    key = f"{community}:{file_name}" if community else file_name
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO _load_progress (file_name, rows_loaded, completed_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (file_name) DO UPDATE
              SET rows_loaded = EXCLUDED.rows_loaded,
                  completed_at = EXCLUDED.completed_at
            """,
            (key, rows_loaded),
        )
    pg_conn.commit()


def is_complete(pg_conn, file_name: str, community: str = "") -> bool:
    key = f"{community}:{file_name}" if community else file_name
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT completed_at FROM _load_progress WHERE file_name = %s",
            (key,),
        )
        row = cur.fetchone()
        return row is not None and row[0] is not None


# =============================================================================
# PostgreSQL loaders
# =============================================================================

def load_users_pg(pg_conn, data_dir: Path, batch_size: int, dry_run: bool, community: str = ""):
    file_name = "Users.xml"
    if is_complete(pg_conn, file_name, community):
        log.info("PostgreSQL: %s already loaded — skipping", file_name)
        return

    xml_path = data_dir / file_name
    batch, total = [], 0

    def flush(batch):
        if dry_run or not batch:
            return
        with pg_conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO users
                  (id, reputation, creation_date, display_name, location,
                   about_me, up_votes, down_votes, views, account_id)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                  reputation    = EXCLUDED.reputation,
                  display_name  = EXCLUDED.display_name,
                  location      = EXCLUDED.location,
                  up_votes      = EXCLUDED.up_votes,
                  down_votes    = EXCLUDED.down_votes,
                  views         = EXCLUDED.views
                """,
                batch,
                template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            )
        pg_conn.commit()

    for attrs in iterparse_rows(xml_path):
        if safe_int(attrs.get("Id")) <= 0:   # skip community bot and invalid IDs
            continue
        batch.append((
            safe_int(attrs.get("Id")),
            safe_int(attrs.get("Reputation")),
            parse_date(attrs.get("CreationDate")),
            attrs.get("DisplayName", ""),
            attrs.get("Location"),
            attrs.get("AboutMe"),           # stored here but not queried via SQL
            safe_int(attrs.get("UpVotes")),
            safe_int(attrs.get("DownVotes")),
            safe_int(attrs.get("Views")),
            safe_int(attrs.get("AccountId")) or None,
        ))
        total += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
        if total % LOG_EVERY == 0:
            log.info("  Users: %d rows processed", total)

    flush(batch)
    mark_complete(pg_conn, file_name, total, community)
    log.info("PostgreSQL Users loaded: %d rows", total)


def load_posts_pg(pg_conn, data_dir: Path, batch_size: int, dry_run: bool, community: str = ""):
    file_name = "Posts.xml"
    if is_complete(pg_conn, file_name, community):
        log.info("PostgreSQL: %s already loaded — skipping", file_name)
        return

    xml_path = data_dir / file_name
    batch, total = [], 0

    def flush(batch):
        if dry_run or not batch:
            return
        with pg_conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO posts
                  (id, post_type_id, accepted_answer_id, parent_id, score,
                   view_count, answer_count, comment_count, owner_user_id,
                   last_editor_user_id, creation_date, last_edit_date,
                   last_activity_date, tags, closed_date, title)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                  score               = EXCLUDED.score,
                  view_count          = EXCLUDED.view_count,
                  answer_count        = EXCLUDED.answer_count,
                  comment_count       = EXCLUDED.comment_count,
                  tags                = EXCLUDED.tags
                """,
                batch,
                template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            )
        pg_conn.commit()

    for attrs in iterparse_rows(xml_path):
        owner_id    = safe_int(attrs.get("OwnerUserId"))   or None
        parent_id   = safe_int(attrs.get("ParentId"))       or None
        accepted_id = safe_int(attrs.get("AcceptedAnswerId")) or None
        editor_id   = safe_int(attrs.get("LastEditorUserId")) or None
        # Null out system/bot user IDs that were skipped in load_users_pg
        if owner_id is not None and owner_id <= 0:
            owner_id = None
        if editor_id is not None and editor_id <= 0:
            editor_id = None

        batch.append((
            safe_int(attrs.get("Id")),
            safe_int(attrs.get("PostTypeId")),
            accepted_id,
            parent_id,
            safe_int(attrs.get("Score")),
            safe_int(attrs.get("ViewCount")),
            safe_int(attrs.get("AnswerCount")),
            safe_int(attrs.get("CommentCount")),
            owner_id,
            editor_id,
            parse_date(attrs.get("CreationDate")),
            parse_date(attrs.get("LastEditDate")),
            parse_date(attrs.get("LastActivityDate")),
            attrs.get("Tags"),
            parse_date(attrs.get("ClosedDate")),
            attrs.get("Title"),
        ))
        total += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
        if total % LOG_EVERY == 0:
            log.info("  Posts: %d rows processed", total)

    flush(batch)
    mark_complete(pg_conn, file_name, total, community)
    log.info("PostgreSQL Posts loaded: %d rows", total)


def load_votes_pg(pg_conn, data_dir: Path, batch_size: int, dry_run: bool, community: str = ""):
    file_name = "Votes.xml"
    if is_complete(pg_conn, file_name, community):
        log.info("PostgreSQL: %s already loaded — skipping", file_name)
        return

    xml_path = data_dir / file_name
    batch, total = [], 0

    def flush(batch):
        if dry_run or not batch:
            return
        with pg_conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO votes (id, post_id, vote_type_id, creation_date, user_id, bounty_amount)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                batch,
                template="(%s,%s,%s,%s,%s,%s)",
            )
        pg_conn.commit()

    for attrs in iterparse_rows(xml_path):
        batch.append((
            safe_int(attrs.get("Id")),
            safe_int(attrs.get("PostId")),
            safe_int(attrs.get("VoteTypeId")),
            parse_date(attrs.get("CreationDate")),
            safe_int(attrs.get("UserId")) or None,
            safe_int(attrs.get("BountyAmount")) or None,
        ))
        total += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
        if total % LOG_EVERY == 0:
            log.info("  Votes: %d rows processed", total)

    flush(batch)
    mark_complete(pg_conn, file_name, total, community)
    log.info("PostgreSQL Votes loaded: %d rows", total)


def load_badges_pg(pg_conn, data_dir: Path, batch_size: int, dry_run: bool, community: str = ""):
    file_name = "Badges.xml"
    if is_complete(pg_conn, file_name, community):
        log.info("PostgreSQL: %s already loaded — skipping", file_name)
        return

    xml_path = data_dir / file_name
    batch, total = [], 0

    def flush(batch):
        if dry_run or not batch:
            return
        with pg_conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO badges (id, user_id, name, class, tag_based, date)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                batch,
                template="(%s,%s,%s,%s,%s,%s)",
            )
        pg_conn.commit()

    for attrs in iterparse_rows(xml_path):
        batch.append((
            safe_int(attrs.get("Id")),
            safe_int(attrs.get("UserId")),
            attrs.get("Name", ""),
            safe_int(attrs.get("Class"), default=3),
            safe_bool(attrs.get("TagBased")),
            parse_date(attrs.get("Date")),
        ))
        total += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
        if total % LOG_EVERY == 0:
            log.info("  Badges: %d rows processed", total)

    flush(batch)
    mark_complete(pg_conn, file_name, total, community)
    log.info("PostgreSQL Badges loaded: %d rows", total)


def load_tags_pg(pg_conn, data_dir: Path, batch_size: int, dry_run: bool, community: str = ""):
    file_name = "Tags.xml"
    if is_complete(pg_conn, file_name, community):
        log.info("PostgreSQL: %s already loaded — skipping", file_name)
        return

    xml_path = data_dir / file_name
    batch, total = [], 0

    def flush(batch):
        if dry_run or not batch:
            return
        # Process each tag row individually so one conflict doesn't abort the batch.
        # Tags are small (~1000-2000 rows per community) so row-by-row is fine.
        for row in batch:
            tag_id, tag_name, count, excerpt_post_id, wiki_post_id = row
            with pg_conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO tags (id, tag_name, count, excerpt_post_id, wiki_post_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (tag_id, tag_name, count, excerpt_post_id, wiki_post_id),
                    )
                    pg_conn.commit()
                except Exception:
                    pg_conn.rollback()
                # Now upsert by tag_name in case the name exists with a different id
                try:
                    cur.execute(
                        """
                        INSERT INTO tags (id, tag_name, count, excerpt_post_id, wiki_post_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (tag_name) DO UPDATE SET
                          count           = GREATEST(tags.count, EXCLUDED.count),
                          excerpt_post_id = COALESCE(tags.excerpt_post_id, EXCLUDED.excerpt_post_id),
                          wiki_post_id    = COALESCE(tags.wiki_post_id, EXCLUDED.wiki_post_id)
                        """,
                        (tag_id, tag_name, count, excerpt_post_id, wiki_post_id),
                    )
                    pg_conn.commit()
                except Exception:
                    pg_conn.rollback()

    for attrs in iterparse_rows(xml_path):
        batch.append((
            safe_int(attrs.get("Id")),
            attrs.get("TagName", ""),
            safe_int(attrs.get("Count")),
            safe_int(attrs.get("ExcerptPostId")) or None,
            safe_int(attrs.get("WikiPostId")) or None,
        ))
        total += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []

    flush(batch)
    mark_complete(pg_conn, file_name, total, community)
    log.info("PostgreSQL Tags loaded: %d rows", total)


def load_comments_pg(pg_conn, data_dir: Path, batch_size: int, dry_run: bool, community: str = ""):
    """Load comment metadata (no text — text goes to Elasticsearch)."""
    file_name = "Comments.xml"
    if is_complete(pg_conn, file_name, community):
        log.info("PostgreSQL: %s already loaded — skipping", file_name)
        return

    xml_path = data_dir / file_name
    batch, total = [], 0

    def flush(batch):
        if dry_run or not batch:
            return
        with pg_conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO comments (id, post_id, score, user_id, creation_date)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                batch,
                template="(%s,%s,%s,%s,%s)",
            )
        pg_conn.commit()

    for attrs in iterparse_rows(xml_path):
        batch.append((
            safe_int(attrs.get("Id")),
            safe_int(attrs.get("PostId")),
            safe_int(attrs.get("Score")),
            safe_int(attrs.get("UserId")) or None,
            parse_date(attrs.get("CreationDate")),
        ))
        total += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
        if total % LOG_EVERY == 0:
            log.info("  Comments (PG): %d rows processed", total)

    flush(batch)
    mark_complete(pg_conn, file_name, total, community)
    log.info("PostgreSQL Comments (metadata) loaded: %d rows", total)


def load_post_links_pg(pg_conn, data_dir: Path, batch_size: int, dry_run: bool, community: str = ""):
    file_name = "PostLinks.xml"
    if is_complete(pg_conn, file_name, community):
        log.info("PostgreSQL: %s already loaded — skipping", file_name)
        return

    xml_path = data_dir / file_name
    batch, total = [], 0

    def flush(batch):
        if dry_run or not batch:
            return
        with pg_conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO post_links (id, creation_date, post_id, related_post_id, link_type_id)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                batch,
                template="(%s,%s,%s,%s,%s)",
            )
        pg_conn.commit()

    for attrs in iterparse_rows(xml_path):
        batch.append((
            safe_int(attrs.get("Id")),
            parse_date(attrs.get("CreationDate")),
            safe_int(attrs.get("PostId")),
            safe_int(attrs.get("RelatedPostId")),
            safe_int(attrs.get("LinkTypeId")),
        ))
        total += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []

    flush(batch)
    mark_complete(pg_conn, file_name, total, community)
    log.info("PostgreSQL PostLinks loaded: %d rows", total)


# =============================================================================
# Neo4j loaders
# =============================================================================

NEO4J_CONSTRAINTS = [
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",
    "CREATE CONSTRAINT answer_id IF NOT EXISTS FOR (a:Answer) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
]

NEO4J_INDEXES = [
    "CREATE INDEX user_reputation IF NOT EXISTS FOR (u:User) ON (u.reputation)",
    "CREATE INDEX question_score IF NOT EXISTS FOR (q:Question) ON (q.score)",
    "CREATE INDEX answer_score IF NOT EXISTS FOR (a:Answer) ON (a.score)",
    "CREATE INDEX tag_count IF NOT EXISTS FOR (t:Tag) ON (t.count)",
]


def ensure_neo4j_schema(driver):
    with driver.session() as session:
        for stmt in NEO4J_CONSTRAINTS + NEO4J_INDEXES:
            session.run(stmt)
    log.info("Neo4j schema constraints and indexes applied")


def load_users_neo4j(driver, data_dir: Path, batch_size: int, dry_run: bool):
    xml_path = data_dir / "Users.xml"
    batch, total = [], 0

    def flush(batch):
        if dry_run or not batch:
            return
        with driver.session() as session:
            session.run(
                """
                UNWIND $rows AS r
                MERGE (u:User {id: r.id})
                SET   u.displayName = r.display_name,
                      u.reputation   = r.reputation,
                      u.location     = r.location,
                      u.creationDate = r.creation_date,
                      u.upVotes      = r.up_votes,
                      u.downVotes    = r.down_votes,
                      u.views        = r.views
                """,
                rows=batch,
            )

    for attrs in iterparse_rows(xml_path):
        if attrs.get("Id") == "-1":
            continue
        batch.append({
            "id": safe_int(attrs.get("Id")),
            "display_name": attrs.get("DisplayName", ""),
            "reputation": safe_int(attrs.get("Reputation")),
            "location": attrs.get("Location"),
            "creation_date": attrs.get("CreationDate"),
            "up_votes": safe_int(attrs.get("UpVotes")),
            "down_votes": safe_int(attrs.get("DownVotes")),
            "views": safe_int(attrs.get("Views")),
        })
        total += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
        if total % LOG_EVERY == 0:
            log.info("  Neo4j Users: %d processed", total)

    flush(batch)
    log.info("Neo4j User nodes loaded: %d", total)


def load_posts_neo4j(driver, data_dir: Path, batch_size: int, dry_run: bool):
    """Creates Question and Answer nodes plus ASKED, ANSWERED, ANSWERS edges."""
    xml_path = data_dir / "Posts.xml"

    q_batch, a_batch, total = [], [], 0
    asked_batch, answered_batch, answers_batch = [], [], []

    def flush_all():
        if dry_run:
            return
        with driver.session() as session:
            # Question nodes
            if q_batch:
                session.run(
                    """
                    UNWIND $rows AS r
                    MERGE (q:Question {id: r.id})
                    SET   q.score        = r.score,
                          q.viewCount    = r.view_count,
                          q.answerCount  = r.answer_count,
                          q.creationDate = r.creation_date,
                          q.title        = r.title
                    """,
                    rows=q_batch,
                )
            # Answer nodes
            if a_batch:
                session.run(
                    """
                    UNWIND $rows AS r
                    MERGE (a:Answer {id: r.id})
                    SET   a.score        = r.score,
                          a.creationDate = r.creation_date
                    """,
                    rows=a_batch,
                )
            # ASKED edges
            if asked_batch:
                session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (u:User {id: r.user_id})
                    MATCH (q:Question {id: r.post_id})
                    MERGE (u)-[:ASKED]->(q)
                    """,
                    rows=asked_batch,
                )
            # ANSWERED edges
            if answered_batch:
                session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (u:User {id: r.user_id})
                    MATCH (a:Answer {id: r.post_id})
                    MERGE (u)-[:ANSWERED]->(a)
                    """,
                    rows=answered_batch,
                )
            # ANSWERS edges
            if answers_batch:
                session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (a:Answer {id: r.answer_id})
                    MATCH (q:Question {id: r.question_id})
                    MERGE (a)-[:ANSWERS]->(q)
                    """,
                    rows=answers_batch,
                )

    for attrs in iterparse_rows(xml_path):
        post_type = safe_int(attrs.get("PostTypeId"))
        post_id = safe_int(attrs.get("Id"))
        owner_id = safe_int(attrs.get("OwnerUserId")) or None
        parent_id = safe_int(attrs.get("ParentId")) or None

        if post_type == 1:  # Question
            q_batch.append({
                "id": post_id,
                "score": safe_int(attrs.get("Score")),
                "view_count": safe_int(attrs.get("ViewCount")),
                "answer_count": safe_int(attrs.get("AnswerCount")),
                "creation_date": attrs.get("CreationDate"),
                "title": attrs.get("Title"),
            })
            if owner_id:
                asked_batch.append({"user_id": owner_id, "post_id": post_id})

        elif post_type == 2:  # Answer
            a_batch.append({
                "id": post_id,
                "score": safe_int(attrs.get("Score")),
                "creation_date": attrs.get("CreationDate"),
            })
            if owner_id:
                answered_batch.append({"user_id": owner_id, "post_id": post_id})
            if parent_id:
                answers_batch.append({"answer_id": post_id, "question_id": parent_id})

        total += 1
        if total % batch_size == 0:
            flush_all()
            q_batch, a_batch = [], []
            asked_batch, answered_batch, answers_batch = [], [], []
        if total % LOG_EVERY == 0:
            log.info("  Neo4j Posts: %d processed", total)

    flush_all()
    log.info("Neo4j Question/Answer nodes and ASKED/ANSWERED/ANSWERS edges loaded: %d posts", total)


def load_accepted_edges_neo4j(driver, data_dir: Path, batch_size: int, dry_run: bool):
    """Creates ACCEPTED edges: (Question)-[:ACCEPTED]->(Answer)."""
    xml_path = data_dir / "Posts.xml"
    batch, total = [], 0

    def flush(batch):
        if dry_run or not batch:
            return
        with driver.session() as session:
            session.run(
                """
                UNWIND $rows AS r
                MATCH (q:Question {id: r.question_id})
                MATCH (a:Answer   {id: r.accepted_id})
                MERGE (q)-[:ACCEPTED]->(a)
                """,
                rows=batch,
            )

    for attrs in iterparse_rows(xml_path):
        if safe_int(attrs.get("PostTypeId")) != 1:
            continue
        accepted_id = safe_int(attrs.get("AcceptedAnswerId")) or None
        if not accepted_id:
            continue
        batch.append({
            "question_id": safe_int(attrs.get("Id")),
            "accepted_id": accepted_id,
        })
        total += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []

    flush(batch)
    log.info("Neo4j ACCEPTED edges loaded: %d", total)


def load_tags_neo4j(driver, data_dir: Path, batch_size: int, dry_run: bool):
    """Creates Tag nodes and TAGGED_WITH edges. CO_OCCURS_WITH derived after load."""
    # --- Tag nodes from Tags.xml ---
    xml_path = data_dir / "Tags.xml"
    batch, total = [], 0

    def flush_tags(batch):
        if dry_run or not batch:
            return
        with driver.session() as session:
            session.run(
                """
                UNWIND $rows AS r
                MERGE (t:Tag {name: r.name})
                SET   t.count = r.count
                """,
                rows=batch,
            )

    for attrs in iterparse_rows(xml_path):
        batch.append({
            "name": attrs.get("TagName", ""),
            "count": safe_int(attrs.get("Count")),
        })
        total += 1
        if len(batch) >= batch_size:
            flush_tags(batch)
            batch = []

    flush_tags(batch)
    log.info("Neo4j Tag nodes loaded: %d", total)

    # --- TAGGED_WITH edges from Posts.xml ---
    xml_path = data_dir / "Posts.xml"
    batch, total = [], 0

    def flush_tagged(batch):
        if dry_run or not batch:
            return
        with driver.session() as session:
            session.run(
                """
                UNWIND $rows AS r
                MATCH (q:Question {id: r.question_id})
                MATCH (t:Tag {name: r.tag_name})
                MERGE (q)-[:TAGGED_WITH]->(t)
                """,
                rows=batch,
            )

    for attrs in iterparse_rows(xml_path):
        if safe_int(attrs.get("PostTypeId")) != 1:
            continue
        raw_tags = attrs.get("Tags", "")
        if not raw_tags:
            continue
        tag_names = [t.strip("<>") for t in raw_tags.split("><") if t.strip("<>")]
        question_id = safe_int(attrs.get("Id"))
        for tag_name in tag_names:
            batch.append({"question_id": question_id, "tag_name": tag_name})
            total += 1
            if len(batch) >= batch_size:
                flush_tagged(batch)
                batch = []

    flush_tagged(batch)
    log.info("Neo4j TAGGED_WITH edges loaded: %d", total)


def derive_co_occurs_with(driver, dry_run: bool):
    """Derives CO_OCCURS_WITH edges from questions that share tags.
    This is a graph analytics step — run after all TAGGED_WITH edges exist.
    Creates/updates weight property on each CO_OCCURS_WITH edge.
    """
    log.info("Neo4j: Deriving CO_OCCURS_WITH edges (this may take several minutes)...")
    if dry_run:
        log.info("  [dry-run] Skipping CO_OCCURS_WITH derivation")
        return

    with driver.session() as session:
        session.run(
            """
            MATCH (t1:Tag)<-[:TAGGED_WITH]-(q:Question)-[:TAGGED_WITH]->(t2:Tag)
            WHERE id(t1) < id(t2)
            WITH  t1, t2, COUNT(q) AS co_count
            MERGE (t1)-[r:CO_OCCURS_WITH]-(t2)
            SET   r.weight = co_count
            """,
            timeout=600,  # Allow up to 10 minutes for large datasets
        )
    log.info("Neo4j CO_OCCURS_WITH edges derived")


def load_post_links_neo4j(driver, data_dir: Path, batch_size: int, dry_run: bool):
    """Creates LINKED_TO and DUPLICATE_OF edges."""
    xml_path = data_dir / "PostLinks.xml"
    linked_batch, dup_batch, total = [], [], 0

    def flush():
        if dry_run:
            return
        with driver.session() as session:
            if linked_batch:
                session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (q1:Question {id: r.post_id})
                    MATCH (q2:Question {id: r.related_id})
                    MERGE (q1)-[:LINKED_TO]->(q2)
                    """,
                    rows=linked_batch,
                )
            if dup_batch:
                session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (q1:Question {id: r.post_id})
                    MATCH (q2:Question {id: r.related_id})
                    MERGE (q1)-[:DUPLICATE_OF]->(q2)
                    """,
                    rows=dup_batch,
                )

    for attrs in iterparse_rows(xml_path):
        link_type = safe_int(attrs.get("LinkTypeId"))
        row = {
            "post_id": safe_int(attrs.get("PostId")),
            "related_id": safe_int(attrs.get("RelatedPostId")),
        }
        if link_type == 1:
            linked_batch.append(row)
        elif link_type == 3:
            dup_batch.append(row)
        total += 1
        if total % batch_size == 0:
            flush()
            linked_batch.clear()
            dup_batch.clear()

    flush()
    log.info("Neo4j LINKED_TO / DUPLICATE_OF edges loaded: %d PostLinks", total)


# =============================================================================
# Elasticsearch loaders
# =============================================================================

ES_INDEX_MAPPINGS = {
    "heterorag_content": {
        "mappings": {
            "properties": {
                "doc_type":       {"type": "keyword"},
                "post_id":        {"type": "integer"},
                "parent_id":      {"type": "integer"},
                "user_id":        {"type": "integer"},
                "score":          {"type": "integer"},
                "creation_date":  {"type": "date"},
                "title":          {"type": "text", "analyzer": "english"},
                "body":           {"type": "text", "analyzer": "english"},
                "tags":           {"type": "keyword"},
                "tag_name":       {"type": "keyword"},
            }
        },
        "settings": {
            "number_of_shards":   1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "english": {
                        "type":      "standard",
                        "stopwords": "_english_",
                    }
                }
            }
        }
    }
}

INDEX_NAME = "heterorag_content"


def _es_index_exists(es_url: str, index_name: str) -> bool:
    """Check index existence via raw HTTP GET — avoids elasticsearch-py
    BadRequestError(400) that occurs with xpack.security disabled in ES 8.x."""
    import urllib.request, urllib.error
    url = f"{es_url.rstrip('/')}/{index_name}"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 200
    except Exception:
        return False


def ensure_es_index(es: Elasticsearch, dry_run: bool):
    if dry_run:
        log.info("[dry-run] Would create ES index %s", INDEX_NAME)
        return
    if not _es_index_exists(ES_URL, INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=ES_INDEX_MAPPINGS[INDEX_NAME])
        log.info("Elasticsearch index '%s' created", INDEX_NAME)
    else:
        log.info("Elasticsearch index '%s' already exists — skipping creation", INDEX_NAME)


def _es_bulk_flush(es: Elasticsearch, batch: list, dry_run: bool):
    if dry_run or not batch:
        return
    successes, errors = es_helpers.bulk(es, batch, raise_on_error=False)
    if errors:
        log.warning("ES bulk: %d errors (first: %s)", len(errors), errors[0])


def load_posts_es(es: Elasticsearch, data_dir: Path, batch_size: int, dry_run: bool):
    """Index question and answer bodies into Elasticsearch."""
    xml_path = data_dir / "Posts.xml"
    batch, total = [], 0

    for attrs in iterparse_rows(xml_path):
        post_type = safe_int(attrs.get("PostTypeId"))
        post_id   = safe_int(attrs.get("Id"))
        body      = attrs.get("Body", "")
        if not body:
            continue

        raw_tags  = attrs.get("Tags", "")
        tag_list  = [t.strip("<>") for t in raw_tags.split("><") if t.strip("<>")] if raw_tags else []

        doc = {
            "_index":       INDEX_NAME,
            "_id":          f"post_{post_id}",
            "doc_type":     "question" if post_type == 1 else "answer",
            "post_id":      post_id,
            "parent_id":    safe_int(attrs.get("ParentId")) or None,
            "user_id":      safe_int(attrs.get("OwnerUserId")) or None,
            "score":        safe_int(attrs.get("Score")),
            "creation_date": attrs.get("CreationDate"),
            "title":        attrs.get("Title"),
            "body":         body,
            "tags":         tag_list,
        }
        batch.append(doc)
        total += 1
        if len(batch) >= batch_size:
            _es_bulk_flush(es, batch, dry_run)
            batch = []
        if total % LOG_EVERY == 0:
            log.info("  ES Posts: %d indexed", total)

    _es_bulk_flush(es, batch, dry_run)
    log.info("Elasticsearch Posts indexed: %d", total)


def load_comments_es(es: Elasticsearch, data_dir: Path, batch_size: int, dry_run: bool):
    xml_path = data_dir / "Comments.xml"
    batch, total = [], 0

    for attrs in iterparse_rows(xml_path):
        comment_id = safe_int(attrs.get("Id"))
        text       = attrs.get("Text", "")
        if not text:
            continue

        batch.append({
            "_index":       INDEX_NAME,
            "_id":          f"comment_{comment_id}",
            "doc_type":     "comment",
            "post_id":      safe_int(attrs.get("PostId")),
            "user_id":      safe_int(attrs.get("UserId")) or None,
            "score":        safe_int(attrs.get("Score")),
            "creation_date": attrs.get("CreationDate"),
            "body":         text,
        })
        total += 1
        if len(batch) >= batch_size:
            _es_bulk_flush(es, batch, dry_run)
            batch = []
        if total % LOG_EVERY == 0:
            log.info("  ES Comments: %d indexed", total)

    _es_bulk_flush(es, batch, dry_run)
    log.info("Elasticsearch Comments indexed: %d", total)


def load_users_es(es: Elasticsearch, data_dir: Path, batch_size: int, dry_run: bool):
    """Index user AboutMe free text."""
    xml_path = data_dir / "Users.xml"
    batch, total = [], 0

    for attrs in iterparse_rows(xml_path):
        about = attrs.get("AboutMe", "")
        if not about or attrs.get("Id") == "-1":
            continue
        user_id = safe_int(attrs.get("Id"))
        batch.append({
            "_index":   INDEX_NAME,
            "_id":      f"user_about_{user_id}",
            "doc_type": "user_about",
            "user_id":  user_id,
            "body":     about,
        })
        total += 1
        if len(batch) >= batch_size:
            _es_bulk_flush(es, batch, dry_run)
            batch = []

    _es_bulk_flush(es, batch, dry_run)
    log.info("Elasticsearch User AboutMe indexed: %d", total)


def load_tag_wikis_es(es: Elasticsearch, data_dir: Path, batch_size: int, dry_run: bool):
    """Index tag wiki excerpts.
    Tags.xml contains ExcerptPostId pointing to Posts.xml bodies.
    Strategy: collect ExcerptPostId set from Tags.xml, then stream Posts.xml once
    for body retrieval.
    """
    tags_xml = data_dir / "Tags.xml"

    # Build tag_name lookup: excerpt_post_id -> tag_name
    excerpt_map: dict[int, str] = {}
    for attrs in iterparse_rows(tags_xml):
        ep_id = safe_int(attrs.get("ExcerptPostId")) or None
        if ep_id:
            excerpt_map[ep_id] = attrs.get("TagName", "")

    if not excerpt_map:
        log.info("No tag wiki excerpts found — skipping")
        return

    log.info("Tag wiki excerpts to index: %d", len(excerpt_map))

    posts_xml = data_dir / "Posts.xml"
    batch, total = [], 0

    for attrs in iterparse_rows(posts_xml):
        post_id = safe_int(attrs.get("Id"))
        if post_id not in excerpt_map:
            continue
        body = attrs.get("Body", "")
        if not body:
            continue
        batch.append({
            "_index":   INDEX_NAME,
            "_id":      f"tag_wiki_{post_id}",
            "doc_type": "tag_wiki",
            "post_id":  post_id,
            "tag_name": excerpt_map[post_id],
            "body":     body,
        })
        total += 1
        if len(batch) >= batch_size:
            _es_bulk_flush(es, batch, dry_run)
            batch = []

    _es_bulk_flush(es, batch, dry_run)
    log.info("Elasticsearch Tag wikis indexed: %d", total)


# =============================================================================
# Orchestration
# =============================================================================

def wait_for_postgres(dsn: str, retries: int = 30, delay: float = 2.0):
    for i in range(retries):
        try:
            conn = psycopg2.connect(dsn)
            conn.close()
            log.info("PostgreSQL is ready")
            return
        except psycopg2.OperationalError:
            log.info("Waiting for PostgreSQL... (%d/%d)", i + 1, retries)
            time.sleep(delay)
    raise RuntimeError("PostgreSQL did not become ready in time")


def wait_for_neo4j(uri: str, user: str, password: str, retries: int = 30, delay: float = 3.0):
    for i in range(retries):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as s:
                s.run("RETURN 1")
            driver.close()
            log.info("Neo4j is ready")
            return
        except Exception:
            log.info("Waiting for Neo4j... (%d/%d)", i + 1, retries)
            time.sleep(delay)
    raise RuntimeError("Neo4j did not become ready in time")


def wait_for_elasticsearch(url: str, retries: int = 30, delay: float = 3.0):
    import urllib.request, urllib.error
    health_url = url.rstrip("/") + "/_cluster/health"
    for i in range(retries):
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:
                if resp.status == 200:
                    log.info("Elasticsearch is ready")
                    return
        except Exception:
            pass
        log.info("Waiting for Elasticsearch... (%d/%d)", i + 1, retries)
        time.sleep(delay)
    raise RuntimeError("Elasticsearch did not become ready in time")


def main():
    parser = argparse.ArgumentParser(description="HeteroRAG Stack Overflow data loader")
    parser.add_argument("--data-dir",   required=True,  help="Directory containing Stack Exchange XML dump files")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run",    action="store_true", help="Parse and validate without writing")
    parser.add_argument("--only",       choices=["postgres", "neo4j", "elasticsearch"],
                        help="Load only the specified service")
    parser.add_argument("--community-name", default=None,
                        help="Unique name for this community (e.g. 'stats', 'dba'). "
                             "Used to namespace _load_progress entries so multiple "
                             "communities can be loaded into the same database.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    required_files = ["Users.xml", "Posts.xml", "Tags.xml"]
    optional_files = ["Votes.xml", "Badges.xml", "Comments.xml", "PostLinks.xml"]

    missing = [f for f in required_files if not (data_dir / f).exists()]
    if missing:
        log.error("Missing required XML files: %s", missing)
        log.error("Users.xml, Posts.xml, and Tags.xml are mandatory.")
        sys.exit(1)

    present_optional = [f for f in optional_files if (data_dir / f).exists()]
    absent_optional  = [f for f in optional_files if not (data_dir / f).exists()]
    if absent_optional:
        log.info("Optional files absent (will be skipped): %s", absent_optional)
    if present_optional:
        log.info("Optional files present: %s", present_optional)

    batch  = args.batch_size
    dry    = args.dry_run
    only   = args.only
    # Community prefix — namespaces _load_progress so multiple communities
    # can be loaded into the same database without skipping each other.
    community = args.community_name or data_dir.name

    log.info("=== HeteroRAG Data Loader ===")
    log.info("Data dir:      %s", data_dir)
    log.info("Community:     %s", community)
    log.info("Batch size:    %d", batch)
    log.info("Dry run:       %s", dry)
    log.info("Only:          %s", only or "all")

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    if only in (None, "postgres"):
        wait_for_postgres(PG_DSN)
        with pg_connection(PG_DSN) as pg:
            log.info("--- Loading PostgreSQL (community: %s) ---", community)
            disable_fk_constraints(pg)
            try:
                load_users_pg(pg, data_dir, batch, dry, community)
                load_tags_pg(pg, data_dir, batch, dry, community)
                load_posts_pg(pg, data_dir, batch, dry, community)
                # Optional files — skip gracefully if absent
                if (data_dir / "Votes.xml").exists():
                    load_votes_pg(pg, data_dir, batch, dry, community)
                else:
                    log.info("Votes.xml absent — skipping")
                if (data_dir / "Badges.xml").exists():
                    load_badges_pg(pg, data_dir, batch, dry, community)
                else:
                    log.info("Badges.xml absent — skipping")
                if (data_dir / "Comments.xml").exists():
                    load_comments_pg(pg, data_dir, batch, dry, community)
                else:
                    log.info("Comments.xml absent — skipping")
                if (data_dir / "PostLinks.xml").exists():
                    load_post_links_pg(pg, data_dir, batch, dry, community)
                else:
                    log.info("PostLinks.xml absent — skipping")
            finally:
                enable_fk_constraints(pg)
        log.info("PostgreSQL load complete")

    # ------------------------------------------------------------------
    # Neo4j
    # ------------------------------------------------------------------
    if only in (None, "neo4j"):
        wait_for_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            log.info("--- Loading Neo4j ---")
            ensure_neo4j_schema(driver)
            load_users_neo4j(driver, data_dir, batch, dry)
            load_tags_neo4j(driver, data_dir, batch, dry)
            load_posts_neo4j(driver, data_dir, batch, dry)
            load_accepted_edges_neo4j(driver, data_dir, batch, dry)
            if (data_dir / "PostLinks.xml").exists():
                load_post_links_neo4j(driver, data_dir, batch, dry)
            else:
                log.info("PostLinks.xml absent — skipping LINKED_TO / DUPLICATE_OF edges")
            derive_co_occurs_with(driver, dry)
        finally:
            driver.close()
        log.info("Neo4j load complete")

    # ------------------------------------------------------------------
    # Elasticsearch
    # ------------------------------------------------------------------
    if only in (None, "elasticsearch"):
        wait_for_elasticsearch(ES_URL)
        es = Elasticsearch(ES_URL)
        log.info("--- Loading Elasticsearch ---")
        ensure_es_index(es, dry)
        load_posts_es(es, data_dir, batch, dry)
        if (data_dir / "Comments.xml").exists():
            load_comments_es(es, data_dir, batch, dry)
        else:
            log.info("Comments.xml absent — skipping comment indexing")
        load_users_es(es, data_dir, batch, dry)
        load_tag_wikis_es(es, data_dir, batch, dry)
        log.info("Elasticsearch load complete")

    log.info("=== All done ===")


if __name__ == "__main__":
    main()
