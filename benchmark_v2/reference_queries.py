"""
benchmark_v2/reference_queries.py
=================================
Hand-written native queries for the gold plans of the S, P and D templates.

They serve two purposes (Benchmark spec 9b, instance-level validation):

1. They check the stores against the oracle *per instance*: executing a gold plan with these
   queries on the live stores must reproduce the gold answer. A mismatch means the store, the
   oracle, the virtual union or the text predicate differs for that instance, and the
   instance is dropped and counted.
2. They are the upper bound for the translation stage: the plan and executor given perfect
   queries.

Parameters that come from the question (a user id, a tag, a word) are written into the query
as literals, as a translator would do. Only identifiers passed between steps use binds.

Like the rest of the benchmark generator, this module does not import the system under test.
"""

from __future__ import annotations

import json
from typing import Any, Callable


def sql_lit(v: Any) -> str:
    if isinstance(v, bool):
        raise TypeError("boolean literal not expected")
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def cy_lit(v: Any) -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return repr(v)
    return json.dumps(str(v))                      # a double-quoted Cypher string


def es_title_has_word(word: str) -> str:
    """Questions whose title contains the word (decision D14): the shared index analyzer
    tokenises the query exactly as it tokenised the titles."""
    return json.dumps({"query": {"bool": {"filter": [
        {"term": {"doc_type": "question"}},
        {"match": {"title": {"query": word, "operator": "and"}}}]}}})


def _tagged(tag: str) -> str:
    return (f"MATCH (q:Question)-[:TAGGED_WITH]->(:Tag {{name: {cy_lit(tag)}}}) "
            f"RETURN DISTINCT q.id AS id")


QUERIES: dict[str, Callable[[dict], dict[str, str]]] = {
    "S01": lambda p: {"a": f"SELECT reputation FROM users WHERE id = {sql_lit(p['user'])}"},
    "S02": lambda p: {"a": f"SELECT count(*) FROM badges WHERE user_id = {sql_lit(p['user'])}"},
    "S03": lambda p: {"a": (
        "SELECT id FROM posts WHERE post_type_id = 1 "
        f"AND EXTRACT(YEAR FROM creation_date) = {sql_lit(int(p['year']))} "
        f"ORDER BY coalesce(view_count, 0) DESC, id ASC LIMIT {sql_lit(p['k'])}")},
    "S05": lambda p: {"a": (f"MATCH (a:Answer)-[:ANSWERS]->(q:Question {{id: {cy_lit(p['question'])}}}) "
                            "RETURN DISTINCT a.id AS id")},
    "S06": lambda p: {"a": (f"MATCH (d:Question)-[:DUPLICATE_OF]->(q:Question {{id: {cy_lit(p['question'])}}}) "
                            "RETURN DISTINCT d.id AS id")},
    "S07": lambda p: {"a": (f"MATCH (t:Tag {{name: {cy_lit(p['tag'])}}})-[r:CO_OCCURS_WITH]-(o:Tag) "
                            f"RETURN o.name AS name ORDER BY r.weight DESC, o.name ASC LIMIT {int(p['k'])}")},
    "S08": lambda p: {"a": (f"MATCH (q:Question {{id: {cy_lit(p['question'])}}})-[:TAGGED_WITH]->(t:Tag) "
                            "RETURN DISTINCT t.name AS name")},
    "S09": lambda p: {"a": es_title_has_word(p["word"])},
    "P01": lambda p: {"a": _tagged(p["tag"]), "b": es_title_has_word(p["word"])},
    "P02": lambda p: {"a": f"SELECT id FROM users WHERE reputation >= {sql_lit(p['min_reputation'])}",
                      "b": ("MATCH (u:User)-[:ANSWERED]->(:Answer)-[:ANSWERS]->"
                            f"(:Question {{id: {cy_lit(p['question'])}}}) RETURN DISTINCT u.id AS id")},
    "P03": lambda p: {"a": ("SELECT id FROM posts WHERE post_type_id = 1 "
                            f"AND coalesce(score, 0) >= {sql_lit(p['min_score'])}"),
                      "b": es_title_has_word(p["word"])},
    "P04": lambda p: {"a": es_title_has_word(p["word"]), "b": _tagged(p["tag"])},
    # ---- class D. D01 and D05 are written as independent queries (the plan has no bind); the
    # executor may rewrite them as dependent queries. D04 needs its bind: the graph has no badges.
    "D01": lambda p: {"a": ("MATCH (u:User)-[:ASKED]->(:Question)-[:TAGGED_WITH]->"
                            f"(:Tag {{name: {cy_lit(p['tag'])}}}) RETURN DISTINCT u.id AS id"),
                      "b": f"SELECT id FROM users WHERE reputation >= {sql_lit(p['min_reputation'])}"},
    "D04": lambda p: {"a": f"SELECT DISTINCT user_id AS id FROM badges WHERE name = {sql_lit(p['badge'])}",
                      "b": ("MATCH (u:User)-[:ASKED]->(:Question)-[:TAGGED_WITH]->(t:Tag) "
                            "WHERE u.id IN $a RETURN DISTINCT t.name AS name")},
    "D05": lambda p: {"a": _tagged(p["tag"]), "b": es_title_has_word(p["word"])},
}


def reference_queries(template_id: str, params: dict) -> dict[str, str]:
    try:
        return QUERIES[template_id](params)
    except KeyError:
        raise KeyError(f"no reference queries for template {template_id!r}") from None
