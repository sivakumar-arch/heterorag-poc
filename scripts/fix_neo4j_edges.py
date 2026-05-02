"""
scripts/fix_neo4j_edges.py
===========================
One-time fix: creates TAGGED_WITH and CO_OCCURS_WITH edges in Neo4j
by reading tag data from PostgreSQL posts table.

Run when Neo4j has Question and Tag nodes but TAGGED_WITH edges are missing
(caused by Neo4j 5.x session.run() commit issue — now fixed in loader).

Usage:
    export PYTHONPATH=.
    python scripts/fix_neo4j_edges.py
"""

import os, logging
import psycopg2
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PG_DSN     = os.environ.get("PG_DSN",
    "host=localhost port=5432 dbname=heterorag user=heterorag password=heterorag_secret")
NEO4J_URI  = os.environ.get("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "heterorag_secret")

BATCH_SIZE = 500


def parse_tags(tag_str: str) -> list[str]:
    """'|python|pandas|numpy|' → ['python', 'pandas', 'numpy']"""
    if not tag_str:
        return []
    return [t.strip("<>| ") for t in tag_str.replace("><", "|").split("|") if t.strip("<>| ")]


def main():
    pg    = psycopg2.connect(PG_DSN)
    neo4j = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    # ── Step 1: verify Neo4j has Question and Tag nodes ──────────────────────
    with neo4j.session() as s:
        q_count = s.run("MATCH (q:Question) RETURN COUNT(q) AS n").single()["n"]
        t_count = s.run("MATCH (t:Tag)      RETURN COUNT(t) AS n").single()["n"]
        tw_count = s.run("MATCH ()-[:TAGGED_WITH]->() RETURN COUNT(*) AS n").single()["n"]
    log.info(f"Neo4j state: {q_count:,} Questions, {t_count:,} Tags, {tw_count:,} TAGGED_WITH edges")

    if tw_count > 0:
        log.info("TAGGED_WITH edges already exist — skipping creation, going straight to CO_OCCURS_WITH")
    else:
        # ── Step 2: fetch (post_id, tag) pairs from PostgreSQL ───────────────
        cur = pg.cursor()
        log.info("Fetching question tags from PostgreSQL...")
        cur.execute("""
            SELECT id, tags FROM posts
            WHERE post_type_id = 1
              AND tags IS NOT NULL
              AND tags != ''
        """)
        rows = cur.fetchall()
        log.info(f"Found {len(rows):,} questions with tags")

        pairs = []
        for post_id, tag_str in rows:
            for tag in parse_tags(tag_str):
                pairs.append({"question_id": post_id, "tag_name": tag})

        log.info(f"Total (question_id, tag_name) pairs: {len(pairs):,}")

        # ── Step 3: write TAGGED_WITH in batches using explicit transactions ──
        total_written = 0
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i:i + BATCH_SIZE]
            with neo4j.session() as s:
                s.execute_write(
                    lambda tx, b=batch: tx.run(
                        """
                        UNWIND $rows AS r
                        MATCH (q:Question {id: r.question_id})
                        MATCH (t:Tag      {name: r.tag_name})
                        MERGE (q)-[:TAGGED_WITH]->(t)
                        """,
                        rows=b,
                    )
                )
            total_written += len(batch)
            if total_written % 20000 == 0:
                log.info(f"  TAGGED_WITH: {total_written:,} / {len(pairs):,}")

        log.info(f"TAGGED_WITH edges written: {total_written:,}")

    # ── Step 4: derive CO_OCCURS_WITH ────────────────────────────────────────
    log.info("Deriving CO_OCCURS_WITH edges (may take 3–8 minutes)...")
    with neo4j.session() as s:
        result = s.execute_write(
            lambda tx: tx.run(
                """
                MATCH (t1:Tag)<-[:TAGGED_WITH]-(q:Question)-[:TAGGED_WITH]->(t2:Tag)
                WHERE elementId(t1) < elementId(t2)
                WITH  t1, t2, COUNT(q) AS co_count
                MERGE (t1)-[r:CO_OCCURS_WITH]-(t2)
                SET   r.weight = co_count
                RETURN COUNT(r) AS created
                """,
                timeout=600,
            )
        )
        record = result.single()
        created = record["created"] if record else "unknown"
    log.info(f"CO_OCCURS_WITH edges created: {created:,}")

    # ── Step 5: final verification ────────────────────────────────────────────
    with neo4j.session() as s:
        tw  = s.run("MATCH ()-[:TAGGED_WITH]->()  RETURN COUNT(*) AS n").single()["n"]
        cow = s.run("MATCH ()-[:CO_OCCURS_WITH]-() RETURN COUNT(*) AS n").single()["n"]
    log.info(f"Final state: TAGGED_WITH={tw:,}  CO_OCCURS_WITH={cow:,}")

    if tw > 0 and cow > 0:
        log.info("✅ All graph edges present — Neo4j is ready.")
    else:
        log.error("❌ Still missing edges — check Neo4j logs.")

    pg.close()
    neo4j.close()


if __name__ == "__main__":
    main()
