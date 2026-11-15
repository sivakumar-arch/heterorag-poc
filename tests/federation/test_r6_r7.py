"""R6 (literal coverage, plan level) and R7 (additive bind, step level), and the dev-pass detectors."""
from types import SimpleNamespace as NS

from benchmark_v2.dev_pass import fired
from heterorag.federation.semantic_checks import (CHECKS, Signatures, additive_bind_errors,
                                                  literal_coverage_errors)


def step(store, kind="set", bind=True):
    return NS(store=store, kind=kind, bind=NS(name="s1", entity="user") if bind else None)


def test_r7_in_checks():
    assert "R7" in CHECKS


def test_r7_sql_fires_on_aggregates_and_limits_only_with_a_bind():
    ok = "SELECT id FROM users WHERE reputation >= 5 AND id = ANY(%(s1)s)"
    assert additive_bind_errors(step("sql"), ok) == []
    for bad in (ok + " LIMIT 10", "SELECT user_id FROM badges WHERE user_id = ANY(%(s1)s) GROUP BY user_id HAVING count(*) > 1",
                "SELECT id FROM users WHERE id = ANY(%(s1)s) OFFSET 5"):
        assert additive_bind_errors(step("sql"), bad), bad
    assert additive_bind_errors(step("sql", bind=False), ok + " LIMIT 10") == []
    assert additive_bind_errors(step("sql", kind="scalar"), "SELECT count(*) FROM users WHERE id = ANY(%(s1)s)") == []


def test_r7_ignores_words_inside_string_literals():
    q = "SELECT id FROM users WHERE display_name = 'limit' AND id = ANY(%(s1)s)"
    assert additive_bind_errors(step("sql"), q) == []


def test_r7_cypher_and_doc():
    assert additive_bind_errors(step("graph"), "MATCH (u:User)-[:ASKED]->(q:Question) WHERE u.id IN $s1 RETURN DISTINCT q.id") == []
    assert additive_bind_errors(step("graph"), "MATCH (u:User) WHERE u.id IN $s1 RETURN u.id LIMIT 5")
    assert additive_bind_errors(step("doc"), '{"terms": {"id": "$bind:s1"}}') == []
    assert additive_bind_errors(step("doc"), '{"size": 5, "query": {"terms": {"id": "$bind:s1"}}}')


def test_r6_numbers_and_quotes():
    q = 'Which users who asked a question tagged "repeated-measures" have a reputation of at least 123,354?'
    good = ["MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE toLower(t.name) = toLower('repeated-measures') RETURN q.id",
            "SELECT id FROM users WHERE reputation >= 123354"]
    assert literal_coverage_errors(q, good) == []
    assert literal_coverage_errors(q, good[:1])                       # the number is missing
    assert literal_coverage_errors(q, {"s1": good[1]})                # the tag is missing
    assert literal_coverage_errors("Which tags?", good) == []         # no literal, nothing to cover


def test_dev_pass_detectors_on_a_minimal_row():
    plan = {"steps": [
        {"id": "s1", "op": "retrieve", "output": "user", "kind": "set", "store": "graph", "question": "x"},
        {"id": "s2", "op": "retrieve", "output": "user", "kind": "set", "store": "sql", "question": "y",
         "inputs": ["s1"], "bind": {"input": "s1", "entity": "user", "param": ""}}]}
    row = {"question_id": "Q", "text": "users with reputation at least 77", "executed_plan": plan,
           "queries": {"s1": "MATCH (u:User) RETURN u.id", "s2": "SELECT id FROM users WHERE id = ANY(%(s1)s)"},
           "guarantee": "exact", "answer_digest": {"type": "ids", "size": 5},
           "step_records": [{"step": "s1", "status": "complete", "value_size": 3, "outside_bound": None},
                            {"step": "s2", "status": "complete", "value_size": 9, "outside_bound": 6}]}
    res, why = fired(row, Signatures.poc())
    assert res["R6"] is True and res["T2"] is True and res["C5"] is True
    assert res["T1"] is False and res["T3"] is False and res["R7"] is False
    assert res["ALL"] is True
