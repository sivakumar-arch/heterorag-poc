"""
benchmark_v2/fidelity.py
========================
Cross-store fidelity check: do the three live stores reproduce the oracle?

Benchmark v2 gold answers come from the oracle (``benchmark_v2/oracle.py``). That is only
meaningful if the stores contain what the oracle contains, no more and no less, and if
each fact lives in exactly one store (the data-ownership rule, decisions D2, D10, D11).
This module checks both, before any question is run:

1. Counts. Expected counts are derived from the oracle with the loader's documented
   semantics (for example, user id <= 0 is not loaded; an ASKED edge exists only when the
   author exists as a user; a graph link exists only between two questions).
2. Value totals. Sums of numeric attributes in SQL against the oracle.
3. Ownership. Exact column sets in PostgreSQL, node and relationship property keys in
   Neo4j, mapped fields in Elasticsearch. Any extra column, property or field is a failure.
4. Samples. A deterministic sample of posts compared record by record: SQL metrics, graph
   neighbourhoods (asker, tags, accepted answer, answers, links, duplicates), and
   Elasticsearch documents (exact title and body, exact field set).

Independence: this module does not import the ``heterorag`` package or the loader.
Live-store drivers are imported lazily, only for the stores actually checked.

Usage
-----
    python -m benchmark_v2.fidelity --oracle data/oracle_stats.sqlite
    python -m benchmark_v2.fidelity --oracle data/oracle_stats.sqlite --only neo4j
    python -m benchmark_v2.fidelity --oracle data/oracle_stats.sqlite --sample 500 --json out.json

Connection settings come from the same environment variables as the loader (PG_DSN,
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ES_URL) with the same local defaults. The exit
code is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable

INDEX_NAME = "heterorag_content"

# ----------------------------------------------------------------------------
# What each store is allowed to hold (the ownership rule, spelled out exactly)
# ----------------------------------------------------------------------------

PG_COLUMNS: dict[str, frozenset[str]] = {
    "users":    frozenset({"id", "reputation", "creation_date", "display_name", "location",
                           "up_votes", "down_votes", "views", "account_id"}),
    "posts":    frozenset({"id", "post_type_id", "score", "view_count", "answer_count",
                           "comment_count", "creation_date", "last_edit_date",
                           "last_activity_date", "closed_date"}),
    "votes":    frozenset({"id", "post_id", "vote_type_id", "creation_date", "user_id",
                           "bounty_amount"}),
    "badges":   frozenset({"id", "user_id", "name", "class", "tag_based", "date"}),
    "tags":     frozenset({"id", "tag_name", "count"}),
    "comments": frozenset({"id", "post_id", "score", "user_id", "creation_date"}),
}

NEO4J_NODE_PROPERTIES: dict[str, frozenset[str]] = {
    "User": frozenset({"id"}), "Question": frozenset({"id"}),
    "Answer": frozenset({"id"}), "Tag": frozenset({"name"}),
}
NEO4J_REL_PROPERTIES: dict[str, frozenset[str]] = {"CO_OCCURS_WITH": frozenset({"weight"})}

ES_FIELDS = frozenset({"doc_type", "post_id", "user_id", "tag_name", "title", "body"})
ES_DOC_FIELDS: dict[str, frozenset[str]] = {
    "question":   frozenset({"doc_type", "post_id", "title", "body"}),
    "answer":     frozenset({"doc_type", "post_id", "body"}),
    "comment":    frozenset({"doc_type", "post_id", "user_id", "body"}),
    "user_about": frozenset({"doc_type", "user_id", "body"}),
    "tag_wiki":   frozenset({"doc_type", "tag_name", "body"}),
}

REL_TYPES = ("ASKED", "ANSWERED", "ANSWERS", "ACCEPTED", "TAGGED_WITH",
             "CO_OCCURS_WITH", "LINKED_TO", "DUPLICATE_OF")


@dataclass
class Check:
    store: str
    kind: str          # count | total | ownership | sample
    name: str
    expected: Any
    observed: Any
    ok: bool
    detail: str = ""


# ----------------------------------------------------------------------------
# Expected values from the oracle
# ----------------------------------------------------------------------------

def _one(db: sqlite3.Connection, sql: str, args: tuple = ()) -> int:
    row = db.execute(sql, args).fetchone()
    return int(row[0] or 0)


def expected_counts(db: sqlite3.Connection) -> dict[str, int]:
    """Counts each store must hold, derived from the oracle with loader semantics."""
    q = {}
    # PostgreSQL ----------------------------------------------------------
    q["postgres.users"] = _one(db, "SELECT count(*) FROM users WHERE id > 0")
    q["postgres.posts"] = _one(db, "SELECT count(*) FROM posts")
    q["postgres.votes"] = _one(db, "SELECT count(*) FROM votes")
    q["postgres.badges"] = _one(db, "SELECT count(*) FROM badges")
    q["postgres.comments"] = _one(db, "SELECT count(*) FROM comments")
    q["postgres.tags"] = _one(db, "SELECT count(DISTINCT tag_name) FROM tags WHERE tag_name <> ''")
    # Neo4j nodes ---------------------------------------------------------
    q["neo4j.node.User"] = q["postgres.users"]
    q["neo4j.node.Question"] = _one(db, "SELECT count(*) FROM posts WHERE post_type_id = 1")
    q["neo4j.node.Answer"] = _one(db, "SELECT count(*) FROM posts WHERE post_type_id = 2")
    q["neo4j.node.Tag"] = q["postgres.tags"]
    # Neo4j relationships -------------------------------------------------
    q["neo4j.rel.ASKED"] = _one(db, """
        SELECT count(*) FROM posts p JOIN users u ON u.id = p.owner_user_id
        WHERE p.post_type_id = 1 AND u.id > 0""")
    q["neo4j.rel.ANSWERED"] = _one(db, """
        SELECT count(*) FROM posts p JOIN users u ON u.id = p.owner_user_id
        WHERE p.post_type_id = 2 AND u.id > 0""")
    q["neo4j.rel.ANSWERS"] = _one(db, """
        SELECT count(*) FROM posts a JOIN posts qu ON qu.id = a.parent_id AND qu.post_type_id = 1
        WHERE a.post_type_id = 2""")
    q["neo4j.rel.ACCEPTED"] = _one(db, """
        SELECT count(*) FROM posts qu JOIN posts a ON a.id = qu.accepted_answer_id AND a.post_type_id = 2
        WHERE qu.post_type_id = 1""")
    q["neo4j.rel.TAGGED_WITH"] = _one(db, """
        SELECT count(*) FROM (
          SELECT DISTINCT pt.post_id, pt.tag FROM post_tags pt
          JOIN posts p ON p.id = pt.post_id AND p.post_type_id = 1
          JOIN tags t ON t.tag_name = pt.tag)""")
    q["neo4j.rel.CO_OCCURS_WITH"] = _one(db, """
        SELECT count(*) FROM (
          SELECT DISTINCT a.tag AS t1, b.tag AS t2 FROM post_tags a
          JOIN post_tags b ON a.post_id = b.post_id AND a.tag < b.tag
          JOIN posts p ON p.id = a.post_id AND p.post_type_id = 1
          JOIN tags ta ON ta.tag_name = a.tag JOIN tags tb ON tb.tag_name = b.tag)""")
    for name, lt in (("LINKED_TO", 1), ("DUPLICATE_OF", 3)):
        q[f"neo4j.rel.{name}"] = _one(db, """
            SELECT count(*) FROM (
              SELECT DISTINCT l.post_id, l.related_post_id FROM post_links l
              JOIN posts p1 ON p1.id = l.post_id AND p1.post_type_id = 1
              JOIN posts p2 ON p2.id = l.related_post_id AND p2.post_type_id = 1
              WHERE l.link_type_id = ?)""", (lt,))
    # Elasticsearch documents ---------------------------------------------
    q["elasticsearch.doc.question"] = _one(
        db, "SELECT count(*) FROM posts WHERE post_type_id = 1 AND body <> ''")
    q["elasticsearch.doc.answer"] = _one(
        db, "SELECT count(*) FROM posts WHERE post_type_id = 2 AND body <> ''")
    q["elasticsearch.doc.comment"] = _one(db, "SELECT count(*) FROM comments WHERE text <> ''")
    q["elasticsearch.doc.user_about"] = _one(
        db, "SELECT count(*) FROM users WHERE about_me <> '' AND id <> -1")
    q["elasticsearch.doc.tag_wiki"] = _one(db, """
        SELECT count(*) FROM posts WHERE body <> '' AND id IN (
          SELECT excerpt_post_id FROM tags WHERE excerpt_post_id > 0)""")
    return q


def expected_totals(db: sqlite3.Connection) -> dict[str, int]:
    """Sums of numeric attributes that PostgreSQL must reproduce (NULL counts as 0)."""
    t = {}
    for col in ("score", "view_count", "answer_count", "comment_count"):
        t[f"postgres.posts.sum_{col}"] = _one(db, f"SELECT sum(coalesce({col}, 0)) FROM posts")
    for col in ("reputation", "up_votes", "down_votes", "views"):
        t[f"postgres.users.sum_{col}"] = _one(
            db, f"SELECT sum(coalesce({col}, 0)) FROM users WHERE id > 0")
    t["postgres.comments.sum_score"] = _one(db, "SELECT sum(coalesce(score, 0)) FROM comments")
    t["postgres.votes.sum_vote_type_id"] = _one(db, "SELECT sum(coalesce(vote_type_id, 0)) FROM votes")
    t["postgres.badges.sum_class"] = _one(db, "SELECT sum(coalesce(class, 3)) FROM badges")
    return t


def sample_ids(db: sqlite3.Connection, n: int) -> dict[str, list[int]]:
    """Deterministic sample: evenly spaced ids in id order (no randomness)."""
    def stride(ids: list[int]) -> list[int]:
        if len(ids) <= n:
            return ids
        step = len(ids) / n
        return [ids[int(i * step)] for i in range(n)]
    questions = [r[0] for r in db.execute("SELECT id FROM posts WHERE post_type_id = 1 ORDER BY id")]
    answers = [r[0] for r in db.execute("SELECT id FROM posts WHERE post_type_id = 2 ORDER BY id")]
    return {"questions": stride(questions), "answers": stride(answers)}


def expected_question_neighbourhood(db: sqlite3.Connection, qid: int) -> dict[str, Any]:
    asker = db.execute("""
        SELECT p.owner_user_id FROM posts p JOIN users u ON u.id = p.owner_user_id
        WHERE p.id = ? AND u.id > 0""", (qid,)).fetchone()
    accepted = db.execute("""
        SELECT a.id FROM posts qu JOIN posts a ON a.id = qu.accepted_answer_id AND a.post_type_id = 2
        WHERE qu.id = ?""", (qid,)).fetchone()
    return {
        "askers": [asker[0]] if asker else [],
        "tags": sorted({r[0] for r in db.execute("""
            SELECT pt.tag FROM post_tags pt JOIN tags t ON t.tag_name = pt.tag
            WHERE pt.post_id = ?""", (qid,))}),
        "accepted": [accepted[0]] if accepted else [],
        "answers": sorted(r[0] for r in db.execute(
            "SELECT id FROM posts WHERE post_type_id = 2 AND parent_id = ?", (qid,))),
        "linked": sorted({r[0] for r in db.execute("""
            SELECT l.related_post_id FROM post_links l
            JOIN posts p2 ON p2.id = l.related_post_id AND p2.post_type_id = 1
            WHERE l.post_id = ? AND l.link_type_id = 1""", (qid,))}),
        "dups": sorted({r[0] for r in db.execute("""
            SELECT l.related_post_id FROM post_links l
            JOIN posts p2 ON p2.id = l.related_post_id AND p2.post_type_id = 1
            WHERE l.post_id = ? AND l.link_type_id = 3""", (qid,))}),
    }


def expected_answer_neighbourhood(db: sqlite3.Connection, aid: int) -> dict[str, Any]:
    author = db.execute("""
        SELECT p.owner_user_id FROM posts p JOIN users u ON u.id = p.owner_user_id
        WHERE p.id = ? AND u.id > 0""", (aid,)).fetchone()
    parent = db.execute("""
        SELECT qu.id FROM posts a JOIN posts qu ON qu.id = a.parent_id AND qu.post_type_id = 1
        WHERE a.id = ?""", (aid,)).fetchone()
    return {"authors": [author[0]] if author else [], "questions": [parent[0]] if parent else []}


def expected_post_metrics(db: sqlite3.Connection, pid: int) -> tuple:
    r = db.execute("""SELECT post_type_id, coalesce(score,0), coalesce(view_count,0),
                      coalesce(answer_count,0), coalesce(comment_count,0)
                      FROM posts WHERE id = ?""", (pid,)).fetchone()
    return tuple(r) if r else ()


def expected_es_post_doc(db: sqlite3.Connection, pid: int) -> dict[str, Any] | None:
    r = db.execute("SELECT post_type_id, title, body FROM posts WHERE id = ?", (pid,)).fetchone()
    if not r or not r[2]:
        return None
    doc = {"doc_type": "question" if r[0] == 1 else "answer", "post_id": pid, "body": r[2]}
    if r[0] == 1:
        doc["title"] = r[1]
    return doc


# ----------------------------------------------------------------------------
# Pure comparison helpers
# ----------------------------------------------------------------------------

def compare_numbers(store: str, kind: str, expected: dict[str, int],
                    observed: dict[str, int]) -> list[Check]:
    out = []
    for key, exp in expected.items():
        obs = observed.get(key)
        out.append(Check(store, kind, key, exp, obs, obs == exp,
                         "" if obs == exp else ("missing" if obs is None else f"diff {obs - exp:+d}")))
    return out


def compare_exact_sets(store: str, name: str, allowed: Iterable[str],
                       observed: Iterable[str]) -> Check:
    allowed, observed = set(allowed), set(observed)
    extra, missing = sorted(observed - allowed), sorted(allowed - observed)
    ok = not extra and not missing
    detail = "" if ok else f"extra={extra} missing={missing}"
    return Check(store, "ownership", name, sorted(allowed), sorted(observed), ok, detail)


def compare_records(store: str, name: str, expected: dict[Any, Any],
                    observed: dict[Any, Any], limit: int = 5) -> Check:
    bad = [k for k in expected if observed.get(k) != expected[k]]
    ok = not bad
    detail = ""
    if bad:
        k = bad[0]
        detail = f"{len(bad)} of {len(expected)} differ; first id {k}: expected {expected[k]!r} observed {observed.get(k)!r}"
    return Check(store, "sample", name, len(expected), len(expected) - len(bad), ok, detail[:600])


# ----------------------------------------------------------------------------
# Live readers (thin wrappers; each returns plain values)
# ----------------------------------------------------------------------------

def check_postgres(db: sqlite3.Connection, dsn: str, ids: dict[str, list[int]]) -> list[Check]:
    import psycopg2  # lazy: only needed when PostgreSQL is checked
    checks: list[Check] = []
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT table_name, column_name FROM information_schema.columns "
                        "WHERE table_schema = 'public'")
            cols: dict[str, set[str]] = {}
            for t, c in cur.fetchall():
                cols.setdefault(t, set()).add(c)
            extra_tables = sorted(t for t in cols if t not in PG_COLUMNS and not t.startswith("_"))
            checks.append(Check("postgres", "ownership", "no unexpected tables", [], extra_tables,
                                not extra_tables, "" if not extra_tables else f"tables {extra_tables}"))
            for t, allowed in PG_COLUMNS.items():
                checks.append(compare_exact_sets("postgres", f"columns of {t}", allowed, cols.get(t, set())))

            obs_counts = {}
            for t in PG_COLUMNS:
                cur.execute(f"SELECT count(*) FROM {t}")
                obs_counts[f"postgres.{t}"] = cur.fetchone()[0]
            checks += compare_numbers("postgres", "count", {k: v for k, v in expected_counts(db).items()
                                      if k.startswith("postgres.")}, obs_counts)

            obs_tot = {}
            for key in expected_totals(db):
                _, table, agg = key.split(".")
                col = agg[len("sum_"):]
                cur.execute(f'SELECT coalesce(sum("{col}"), 0) FROM {table}')
                obs_tot[key] = int(cur.fetchone()[0])
            checks += compare_numbers("postgres", "total", expected_totals(db), obs_tot)

            post_ids = ids["questions"] + ids["answers"]
            cur.execute("SELECT id, post_type_id, score, view_count, answer_count, comment_count "
                        "FROM posts WHERE id = ANY(%s)", (post_ids,))
            observed = {r[0]: tuple(r[1:]) for r in cur.fetchall()}
            expected = {p: expected_post_metrics(db, p) for p in post_ids}
            checks.append(compare_records("postgres", "post metrics sample", expected, observed))
    finally:
        conn.close()
    return checks


def check_neo4j(db: sqlite3.Connection, uri: str, user: str, password: str,
                ids: dict[str, list[int]]) -> list[Check]:
    from neo4j import GraphDatabase  # lazy
    checks: list[Check] = []
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as s:
            obs = {}
            for label in NEO4J_NODE_PROPERTIES:
                obs[f"neo4j.node.{label}"] = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            for rel in REL_TYPES:
                obs[f"neo4j.rel.{rel}"] = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
            exp = {k: v for k, v in expected_counts(db).items() if k.startswith("neo4j.")}
            checks += compare_numbers("neo4j", "count", exp, obs)

            # Ownership: property keys actually present on nodes and relationships
            node_keys: dict[str, set[str]] = {}
            for r in s.run("MATCH (n) UNWIND keys(n) AS k RETURN labels(n)[0] AS l, collect(DISTINCT k) AS ks"):
                node_keys[r["l"]] = set(r["ks"])
            for label, allowed in NEO4J_NODE_PROPERTIES.items():
                present = node_keys.pop(label, set())
                extra = sorted(present - allowed)
                checks.append(Check("neo4j", "ownership", f"{label} node properties", sorted(allowed),
                                    sorted(present), not extra, "" if not extra else f"extra={extra}"))
            other = {l: sorted(k) for l, k in node_keys.items()}
            checks.append(Check("neo4j", "ownership", "no other node labels with properties", {}, other,
                                not other, "" if not other else str(other)))
            rel_keys: dict[str, set[str]] = {}
            for r in s.run("MATCH ()-[r]->() UNWIND keys(r) AS k RETURN type(r) AS t, collect(DISTINCT k) AS ks"):
                rel_keys[r["t"]] = set(r["ks"])
            for t, ks in rel_keys.items():
                extra = sorted(ks - NEO4J_REL_PROPERTIES.get(t, frozenset()))
                checks.append(Check("neo4j", "ownership", f"{t} relationship properties",
                                    sorted(NEO4J_REL_PROPERTIES.get(t, frozenset())), sorted(ks),
                                    not extra, "" if not extra else f"extra={extra}"))

            # Sample: neighbourhoods of questions and answers
            qids = ids["questions"]
            rows = s.run("""
                MATCH (q:Question) WHERE q.id IN $ids
                RETURN q.id AS id,
                  [(u:User)-[:ASKED]->(q) | u.id] AS askers,
                  [(q)-[:TAGGED_WITH]->(t:Tag) | t.name] AS tags,
                  [(q)-[:ACCEPTED]->(a:Answer) | a.id] AS accepted,
                  [(a2:Answer)-[:ANSWERS]->(q) | a2.id] AS answers,
                  [(q)-[:LINKED_TO]->(r:Question) | r.id] AS linked,
                  [(q)-[:DUPLICATE_OF]->(r2:Question) | r2.id] AS dups
            """, ids=qids).data()
            observed = {r["id"]: {"askers": sorted(r["askers"]), "tags": sorted(set(r["tags"])),
                                  "accepted": sorted(r["accepted"]), "answers": sorted(r["answers"]),
                                  "linked": sorted(set(r["linked"])), "dups": sorted(set(r["dups"]))}
                        for r in rows}
            expected = {q: expected_question_neighbourhood(db, q) for q in qids}
            checks.append(compare_records("neo4j", "question neighbourhood sample", expected, observed))

            aids = ids["answers"]
            rows = s.run("""
                MATCH (a:Answer) WHERE a.id IN $ids
                RETURN a.id AS id, [(u:User)-[:ANSWERED]->(a) | u.id] AS authors,
                       [(a)-[:ANSWERS]->(q:Question) | q.id] AS questions
            """, ids=aids).data()
            observed = {r["id"]: {"authors": sorted(r["authors"]), "questions": sorted(r["questions"])}
                        for r in rows}
            expected = {a: expected_answer_neighbourhood(db, a) for a in aids}
            checks.append(compare_records("neo4j", "answer neighbourhood sample", expected, observed))
    finally:
        driver.close()
    return checks


def _es_request(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def check_elasticsearch(db: sqlite3.Connection, es_url: str, ids: dict[str, list[int]]) -> list[Check]:
    base = es_url.rstrip("/")
    checks: list[Check] = []
    obs = {}
    for dt in ES_DOC_FIELDS:
        res = _es_request(f"{base}/{INDEX_NAME}/_count", {"query": {"term": {"doc_type": dt}}})
        obs[f"elasticsearch.doc.{dt}"] = res["count"]
    total = _es_request(f"{base}/{INDEX_NAME}/_count")["count"]
    exp = {k: v for k, v in expected_counts(db).items() if k.startswith("elasticsearch.")}
    checks += compare_numbers("elasticsearch", "count", exp, obs)
    checks.append(Check("elasticsearch", "count", "no documents of unknown type",
                        sum(exp.values()), total, total == sum(exp.values()),
                        "" if total == sum(exp.values()) else f"total {total} vs sum of types {sum(exp.values())}"))

    mapping = _es_request(f"{base}/{INDEX_NAME}/_mapping")
    props: set[str] = set()
    for body in mapping.values():
        props |= set(body.get("mappings", {}).get("properties", {}))
    checks.append(compare_exact_sets("elasticsearch", "mapped fields", ES_FIELDS, props))

    post_ids = ids["questions"] + ids["answers"]
    res = _es_request(f"{base}/{INDEX_NAME}/_mget", {"ids": [f"post_{p}" for p in post_ids]})
    observed: dict[int, Any] = {}
    for d in res["docs"]:
        pid = int(d["_id"].split("_", 1)[1])
        observed[pid] = d["_source"] if d.get("found") else None
    expected = {p: expected_es_post_doc(db, p) for p in post_ids}
    checks.append(compare_records("elasticsearch", "post document sample (exact title, body, fields)",
                                  expected, observed))
    return checks


# ----------------------------------------------------------------------------
# Orchestration and CLI
# ----------------------------------------------------------------------------

def run(oracle_path: str, only: str | None, sample: int,
        readers: dict[str, Callable[..., list[Check]]] | None = None) -> list[Check]:
    db = sqlite3.connect(f"file:{oracle_path}?mode=ro", uri=True)
    try:
        ids = sample_ids(db, sample)
        readers = readers or {
            "postgres": lambda d, i: check_postgres(d, os.getenv(
                "PG_DSN", "host=localhost port=5432 dbname=heterorag user=heterorag password=heterorag_secret"), i),
            "neo4j": lambda d, i: check_neo4j(d, os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                                              os.getenv("NEO4J_USER", "neo4j"),
                                              os.getenv("NEO4J_PASSWORD", "heterorag_secret"), i),
            "elasticsearch": lambda d, i: check_elasticsearch(d, os.getenv("ES_URL", "http://localhost:9200"), i),
        }
        out: list[Check] = []
        for name, fn in readers.items():
            if only and name != only:
                continue
            out += fn(db, ids)
        return out
    finally:
        db.close()


def render(checks: list[Check]) -> str:
    lines = []
    store = None
    for c in checks:
        if c.store != store:
            store = c.store
            lines.append(f"\n== {store.upper()}")
        mark = "OK  " if c.ok else "FAIL"
        if c.kind in ("count", "total"):
            lines.append(f"  {mark} {c.kind:9s} {c.name:34s} expected={c.expected} observed={c.observed} {c.detail}")
        else:
            lines.append(f"  {mark} {c.kind:9s} {c.name:34s} {c.detail}")
    failed = [c for c in checks if not c.ok]
    lines.append("")
    lines.append(f"{len(checks) - len(failed)} of {len(checks)} checks passed"
                 + ("" if not failed else f"; {len(failed)} FAILED"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check the live stores against the oracle")
    ap.add_argument("--oracle", required=True, help="path to the oracle SQLite file")
    ap.add_argument("--only", choices=["postgres", "neo4j", "elasticsearch"])
    ap.add_argument("--sample", type=int, default=300, help="questions and answers sampled (default 300 each)")
    ap.add_argument("--json", help="write all checks to this JSON file")
    args = ap.parse_args(argv)
    checks = run(args.oracle, args.only, args.sample)
    print(render(checks))
    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(c) for c in checks], f, indent=2, default=str)
    return 0 if checks and all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
