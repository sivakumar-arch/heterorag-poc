"""
benchmark_v2/union.py
=====================
The virtual union V as SQL views over the oracle.

V is the set of facts the three stores hold (spec section 4). The oracle holds everything
in the source files; these views cut it down to what was actually loaded, with the loaders'
documented semantics:

    users            id > 0 only (the Community user, id -1, is not loaded)
    authorship       an ASKED / ANSWERED edge exists only if the author is a loaded user
    ANSWERS          an answer's parent must be a question
    ACCEPTED         the accepted post must be an answer
    TAGGED_WITH      (question, tag) pairs where the tag exists in the tag table
    LINKED_TO,       both ends must be questions
    DUPLICATE_OF
    documents        a question or answer has a document only if its body is non-empty

Views are TEMP objects, so the oracle file itself is never modified and can be opened
read-only. Gold queries are written against these views, and ``tests`` assert that their row
counts equal the expected counts of ``benchmark_v2.fidelity``, which computes them by
separate SQL. Two independent derivations of V that agree are the check on this module.

Like the oracle, this module does not import the ``heterorag`` package.
"""

from __future__ import annotations

import re
import sqlite3

_TOKEN = re.compile(r"[a-z0-9_]+(?:[.'’][a-z0-9_]+)*")


def tokens(text: str | None) -> list[str]:
    """Lower-cased word tokens, approximating the Elasticsearch standard tokenizer
    (decision D14): letters, digits and underscore join; an apostrophe or full stop between
    word characters joins; everything else separates. Approximate by design: every
    generated instance is validated against a native Elasticsearch query."""
    if not text:
        return []
    return _TOKEN.findall(text.lower())


def has_word(text: str | None, word: str) -> int:
    return 1 if word.lower() in set(tokens(text)) else 0


VIEWS: dict[str, str] = {
    "v_users": "SELECT * FROM users WHERE id > 0",
    "v_questions": "SELECT * FROM posts WHERE post_type_id = 1",
    "v_answers": "SELECT * FROM posts WHERE post_type_id = 2",
    # --- graph relationships
    "v_asked": """SELECT p.owner_user_id AS user_id, p.id AS question_id
                  FROM posts p JOIN users u ON u.id = p.owner_user_id
                  WHERE p.post_type_id = 1 AND u.id > 0""",
    "v_answered": """SELECT p.owner_user_id AS user_id, p.id AS answer_id
                     FROM posts p JOIN users u ON u.id = p.owner_user_id
                     WHERE p.post_type_id = 2 AND u.id > 0""",
    "v_answers_of": """SELECT a.id AS answer_id, q.id AS question_id
                       FROM posts a JOIN posts q ON q.id = a.parent_id AND q.post_type_id = 1
                       WHERE a.post_type_id = 2""",
    "v_accepted": """SELECT q.id AS question_id, a.id AS answer_id
                     FROM posts q JOIN posts a ON a.id = q.accepted_answer_id AND a.post_type_id = 2
                     WHERE q.post_type_id = 1""",
    "v_tagged": """SELECT DISTINCT pt.post_id AS question_id, pt.tag AS tag
                   FROM post_tags pt
                   JOIN posts p ON p.id = pt.post_id AND p.post_type_id = 1
                   JOIN tags t ON t.tag_name = pt.tag""",
    "v_linked": """SELECT DISTINCT l.post_id AS question_id, l.related_post_id AS related_id
                   FROM post_links l
                   JOIN posts p1 ON p1.id = l.post_id AND p1.post_type_id = 1
                   JOIN posts p2 ON p2.id = l.related_post_id AND p2.post_type_id = 1
                   WHERE l.link_type_id = 1""",
    "v_duplicate_of": """SELECT DISTINCT l.post_id AS question_id, l.related_post_id AS original_id
                         FROM post_links l
                         JOIN posts p1 ON p1.id = l.post_id AND p1.post_type_id = 1
                         JOIN posts p2 ON p2.id = l.related_post_id AND p2.post_type_id = 1
                         WHERE l.link_type_id = 3""",
    # --- document store
    "v_question_docs": "SELECT id AS post_id, title FROM posts WHERE post_type_id = 1 AND body <> ''",
    "v_answer_docs": "SELECT id AS post_id FROM posts WHERE post_type_id = 2 AND body <> ''",
    "v_comment_docs": "SELECT id AS comment_id, post_id, user_id FROM comments WHERE text <> ''",
    "v_user_about_docs": "SELECT id AS user_id FROM users WHERE about_me <> '' AND id <> -1",
    "v_tag_wiki_docs": """SELECT t.tag_name AS tag_name, p.body AS body
                          FROM tags t JOIN posts p ON p.id = t.excerpt_post_id
                          WHERE t.excerpt_post_id > 0 AND p.body <> ''""",
}

# Row counts the views must reproduce (fidelity.expected_counts keys, derived separately).
VIEW_TO_EXPECTED = {
    "v_users": "postgres.users",
    "v_asked": "neo4j.rel.ASKED",
    "v_answered": "neo4j.rel.ANSWERED",
    "v_answers_of": "neo4j.rel.ANSWERS",
    "v_accepted": "neo4j.rel.ACCEPTED",
    "v_tagged": "neo4j.rel.TAGGED_WITH",
    "v_linked": "neo4j.rel.LINKED_TO",
    "v_duplicate_of": "neo4j.rel.DUPLICATE_OF",
    "v_question_docs": "elasticsearch.doc.question",
    "v_answer_docs": "elasticsearch.doc.answer",
    "v_comment_docs": "elasticsearch.doc.comment",
    "v_user_about_docs": "elasticsearch.doc.user_about",
}

_COOC_SQL = """
CREATE TEMP TABLE v_cooc AS
SELECT a.tag AS tag1, b.tag AS tag2, count(*) AS weight
FROM v_tagged a JOIN v_tagged b ON a.question_id = b.question_id AND a.tag < b.tag
GROUP BY a.tag, b.tag
"""


def open_union(oracle_path: str) -> sqlite3.Connection:
    """Open the oracle read-only and install the views and helper functions."""
    db = sqlite3.connect(f"file:{oracle_path}?mode=ro", uri=True)
    install(db)
    return db


def install(db: sqlite3.Connection) -> None:
    db.create_function("has_word", 2, has_word, deterministic=True)
    for name, sql in VIEWS.items():
        db.execute(f"DROP VIEW IF EXISTS temp.{name}")
        db.execute(f"CREATE TEMP VIEW {name} AS {sql}")


def ensure_cooccurrence(db: sqlite3.Connection) -> None:
    """Materialise tag co-occurrence weights (CO_OCCURS_WITH) once per connection."""
    exists = db.execute("SELECT 1 FROM sqlite_temp_master WHERE name = 'v_cooc'").fetchone()
    if not exists:
        db.execute(_COOC_SQL)
        db.execute("CREATE INDEX temp.idx_cooc1 ON v_cooc(tag1)")
        db.execute("CREATE INDEX temp.idx_cooc2 ON v_cooc(tag2)")


def view_counts(db: sqlite3.Connection) -> dict[str, int]:
    out = {name: db.execute(f"SELECT count(*) FROM {name}").fetchone()[0] for name in VIEW_TO_EXPECTED}
    ensure_cooccurrence(db)
    out["v_cooc"] = db.execute("SELECT count(*) FROM v_cooc").fetchone()[0]
    out["v_tag_wiki_docs"] = db.execute("SELECT count(*) FROM v_tag_wiki_docs").fetchone()[0]
    return out
