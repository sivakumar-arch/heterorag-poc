"""R4 (the bind is never negated) and C5 (a same-entity bound step returns only ids of its bound set)."""
import json

import pytest

from heterorag.federation import Bind, Executor, Guarantee, Op, Plan, Step, Store
from heterorag.federation.guarantee import Kind
from heterorag.federation.semantic_checks import negated_bind_errors
from heterorag.federation.step_validator import StepValidator
from tests.federation.fake_descriptors import catalogue
from tests.federation.test_executor import Fake, bound, retrieve, table

ON = StepValidator(catalogue(), semantic_checks=True)
OFF = StepValidator(catalogue())


def b(store, output="user", entity="question", mode=None):
    return Step("s2", Op.RETRIEVE, store, mode, "q", inputs=("s1",), bind=Bind("s1", entity), output=output,
                kind=Kind.SET)


# ---------------------------------------------------------------- R4
@pytest.mark.parametrize("q", [
    "SELECT id FROM users WHERE id NOT IN (%(s1)s)",
    "SELECT id FROM users WHERE id NOT IN %(s1)s",
    "SELECT id FROM users WHERE id <> ALL(%(s1)s)",
    "SELECT id FROM users WHERE id != ALL (%(s1)s)",
    "SELECT id FROM users WHERE NOT (id = ANY(%(s1)s))",
    "SELECT id FROM users u WHERE NOT EXISTS (SELECT 1 FROM posts p WHERE p.owner_user_id = u.id AND p.id = ANY(%(s1)s))",
])
def test_negated_sql_bind_is_flagged(q):
    assert negated_bind_errors(b(Store.SQL), q)
    assert any("negated position" in e for e in ON.check(b(Store.SQL), q))
    assert not any("negated position" in e for e in OFF.check(b(Store.SQL), q))


@pytest.mark.parametrize("q", [
    "SELECT id FROM users WHERE id = ANY(%(s1)s) AND reputation > 5",
    "SELECT DISTINCT owner_user_id AS id FROM posts WHERE id = ANY(%(s1)s)",
    "SELECT id FROM users WHERE id IN (%(s1)s) AND display_name NOT LIKE 'x%%'",
    "SELECT id FROM users WHERE NOT (reputation < 5) AND id = ANY(%(s1)s)",
])
def test_positive_sql_bind_passes(q):
    assert negated_bind_errors(b(Store.SQL), q) == []


def test_cypher_negation():
    step = b(Store.GRAPH, "user", "question")
    assert negated_bind_errors(step, "MATCH (q:Question) WHERE NOT q.id IN $s1 RETURN q.id")
    assert negated_bind_errors(step, "MATCH (q:Question) WHERE NOT (q.id IN $s1) RETURN q.id")
    assert negated_bind_errors(step, "MATCH (q:Question)<-[:ASKED]-(u:User) WHERE q.id IN $s1 RETURN DISTINCT u.id") == []


def test_elasticsearch_must_not():
    step = b(Store.DOC, "question", "user", mode=None)
    neg = json.dumps({"query": {"bool": {"must": [{"term": {"doc_type": "question"}}],
                                         "must_not": [{"terms": {"owner_user_id": "$bind:s1"}}]}}})
    pos = json.dumps({"query": {"bool": {"must": [{"term": {"doc_type": "question"}},
                                                   {"terms": {"owner_user_id": "$bind:s1"}}]}}})
    assert negated_bind_errors(step, neg) and negated_bind_errors(step, pos) == []
    assert negated_bind_errors(step, "not json") == []


def test_no_bind_no_error():
    assert negated_bind_errors(Step("s1", Op.RETRIEVE, Store.SQL, None, "q", output="user"),
                               "SELECT id FROM users WHERE id NOT IN (1, 2)") == []


# ---------------------------------------------------------------- C5
def two_step(clients_rows, contain, bind_entity="user", out="user"):
    plan = Plan((retrieve("s1", Store.GRAPH, bind_entity),
                 bound("s2", Store.SQL, "s1", bind_entity, entity=out)))
    clients = {Store.GRAPH: Fake(table({"s1": clients_rows[0]})), Store.SQL: Fake(table({"s2": clients_rows[1]}))}
    return Executor(clients, containment=contain).run(plan, {"s1": "N1", "s2": "N2"})


def test_result_inside_the_bound_set_is_untouched():
    r = two_step(([1, 2, 3], [2, 3]), contain=True)
    assert r.answer.exact and r.answer.value == [2, 3] and r.records["s2"].outside_bound == 0


def test_result_outside_the_bound_set_fails_the_step_and_abstains_when_on():
    r = two_step(([1, 2, 3], [2, 3, 99, 100]), contain=True)
    s2 = r.records["s2"]
    assert s2.guarantee is Guarantee.NONE and "containment violated: 2 of 4" in s2.error and s2.outside_bound == 2
    assert r.answer.type == "abstain"


def test_flag_off_only_counts_and_the_answer_is_unchanged():
    r = two_step(([1, 2, 3], [2, 3, 99, 100]), contain=False)
    assert r.records["s2"].outside_bound == 2 and r.answer.value == [2, 3, 99, 100]
    assert r.records["s2"].guarantee is Guarantee.EXACT


def test_identifiers_compare_as_strings():
    r = two_step((["1", "2"], [1, 2]), contain=True)
    assert r.answer.exact and r.records["s2"].outside_bound == 0


def test_not_applicable_when_the_entities_differ_or_no_bind():
    r = two_step(([1, 2], [7, 8]), contain=True, bind_entity="question", out="user")
    assert r.answer.value == [7, 8] and r.records["s2"].outside_bound is None
    r2 = Executor({Store.GRAPH: Fake(table({"s1": [5]}))}).run(Plan((retrieve("s1", Store.GRAPH),)), {"s1": "N"})
    assert r2.records["s1"].outside_bound is None


def test_environment_flag_is_the_default(monkeypatch):
    monkeypatch.setenv("HETERORAG_SEMANTIC_CHECKS", "1")
    assert Executor({})._contain is True
    monkeypatch.delenv("HETERORAG_SEMANTIC_CHECKS")
    assert Executor({})._contain is False
