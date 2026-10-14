"""Static checks on translated queries. The hand-written reference queries must pass them."""
import json

import pytest

from benchmark_v2 import questions
from benchmark_v2.reference_queries import reference_queries
from heterorag.federation import Bind, Mode, Op, Plan, Step, Store
from heterorag.federation.guarantee import Kind
from heterorag.federation.step_validator import StepValidator
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)
from tests.federation.fake_descriptors import catalogue
from tests.federation.test_reference_and_validation import records_for

V = StepValidator(catalogue())


def sql(output="question", kind=Kind.SET, k=None, bind=None, sid="s1"):
    return Step(sid, Op.RETRIEVE, Store.SQL, None, "q", inputs=(bind.input,) if bind else (), bind=bind,
                output=output, kind=kind, k=k)


def gr(output="question", kind=Kind.SET, k=None, bind=None):
    return Step("s1", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=(bind.input,) if bind else (), bind=bind,
                output=output, kind=kind, k=k)


def doc(output="question", mode=Mode.FILTER, k=None, bind=None):
    return Step("s1", Op.RETRIEVE, Store.DOC, mode, "q", inputs=(bind.input,) if bind else (), bind=bind,
                output=output, k=k)


def test_every_reference_query_passes_the_validator(world):
    recs = records_for(world, [t.id for t in questions.all_templates()], per=3)
    seen = set()
    for r in recs:
        plan = Plan.from_dict({"steps": r["plan"]})
        for sid, text in reference_queries(r["template"], r["params"]).items():
            step = plan.by_id()[sid]
            assert V.check(step, text) == [], (r["template"], sid, text)
            seen.add(r["template"])
    assert seen == {t.id for t in questions.all_templates()}


# ---------------------------------------------------------------- SQL

@pytest.mark.parametrize("q,frag", [
    ("DELETE FROM users", "SELECT"),
    ("SELECT id FROM users; DROP TABLE users", "one SQL statement"),
    ("SELECT id INTO t FROM users", "read-only"),
    ("SELECT id FROM users LIMIT 5", "remove LIMIT"),
    ("SELECT id FROM people", "unknown table"),
    ("SELECT u.nope FROM users u", "no column"),
    ("SELECT id FROM posts WHERE title LIKE 'a%'", "percent"),
    ("SELECT reputation FROM users", "identifier"),
    ("SELECT id FROM users WHERE id = ANY(%(s9)s)", "placeholders"),
])
def test_sql_rejections(q, frag):
    errs = V.check(sql("user"), q)
    assert any(frag in e for e in errs), errs


def test_sql_accepts_common_shapes():
    assert V.check(sql("user"), "SELECT id FROM users WHERE reputation >= 500") == []
    assert V.check(sql("question"), "SELECT p.id FROM posts p WHERE p.post_type_id = 1 AND EXTRACT(YEAR FROM p.creation_date) = 2019") == []
    assert V.check(sql("tag"), "SELECT tag_name FROM tags WHERE \"count\" > 100") == []
    assert V.check(sql("user"), "SELECT DISTINCT b.user_id AS id FROM badges b WHERE b.name = 'Teacher'") == []
    assert V.check(sql("question", kind=Kind.SET, bind=Bind("s0", "question")),
                   "SELECT id FROM posts WHERE id = ANY(%(s0)s) AND score > 3") == []
    assert V.check(Step("s1", Op.RETRIEVE, Store.SQL, None, "q", output="number", kind=Kind.SCALAR),
                   "SELECT reputation FROM users WHERE id = 5") == []


def test_sql_comment_and_literal_do_not_hide_or_trigger_checks():
    assert V.check(sql("user"), "SELECT id FROM users WHERE display_name = 'drop; delete'") == []
    assert any("read-only" in e for e in V.check(sql("user"), "SELECT id FROM users /* ok */ ; DELETE FROM users")
               ) or any("one SQL" in e for e in V.check(sql("user"), "SELECT id FROM users; DELETE FROM users"))


def test_sql_ranked_needs_order_and_exact_limit():
    s = sql("question", Kind.RANKED, k=5)
    assert V.check(s, "SELECT id, score FROM posts WHERE post_type_id = 1 ORDER BY score DESC, id ASC LIMIT 5") == []
    assert any("LIMIT 5" in e for e in V.check(s, "SELECT id, score FROM posts ORDER BY score DESC LIMIT 10"))
    assert any("ORDER BY" in e for e in V.check(s, "SELECT id, score FROM posts LIMIT 5"))


# ---------------------------------------------------------------- Cypher

@pytest.mark.parametrize("q,frag", [
    ("MATCH (q:Question) DELETE q", "read-only"),
    ("MATCH (q:Question) SET q.x = 1 RETURN q.id", "read-only"),
    ("CALL db.labels()", "read-only"),
    ("MATCH (q:Post) RETURN q.id", "unknown node label"),
    ("MATCH (q:Question)-[:LIKES]->(t:Tag) RETURN q.id", "unknown relationship"),
    ("MATCH (q:Question) RETURN q.score", "unknown property"),
    ("MATCH (q:Question) RETURN q.id LIMIT 5", "remove LIMIT"),
    ("MATCH (q:Question) WHERE q.id IN $s9 RETURN q.id", "parameters"),
    ("MATCH (q:Question) RETURN count(q)", "identifier"),
    ("MATCH (q:Question)", "RETURN"),
    ("MATCH (:Question {id: 5})<-[:ANSWERS]-(:Answer)\nRETURN DISTINCT Answer.id", "not defined"),
    ("MATCH (q:Question) RETURN DISTINCT x.id", "not defined"),
    ("MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) RETURN DISTINCT q.id, z.name", "not defined"),
])
def test_cypher_rejections(q, frag):
    errs = V.check(gr(), q)
    assert any(frag in e for e in errs), errs


def test_cypher_accepts_common_shapes():
    assert V.check(gr("question"), "MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag {name: \"python\"}) RETURN DISTINCT q.id") == []
    assert V.check(gr("tag", Kind.RANKED, k=3),
                   "MATCH (a:Tag {name: 'x'})-[r:CO_OCCURS_WITH]-(o:Tag) RETURN o.name, r.weight ORDER BY r.weight DESC, o.name ASC LIMIT 3") == []
    assert V.check(gr("question", bind=Bind("s0", "user")),
                   "MATCH (u:User)-[:ASKED]->(q:Question) WHERE u.id IN $s0 RETURN DISTINCT q.id") == []
    assert V.check(gr("question"), "MATCH (q:Question) WHERE q.id = 7 // set aside\nRETURN q.id") == []
    # variables from relationships, aliases, paths and WITH
    assert V.check(gr("tag", Kind.RANKED, k=2),
                   "MATCH (a:Tag {name: 'x'})-[r:CO_OCCURS_WITH]-(o:Tag) WITH o, r.weight AS w RETURN o.name, w ORDER BY w DESC, o.name ASC LIMIT 2") == []
    assert V.check(gr("answer"), "MATCH p = (a:Answer)-[:ANSWERS]->(q:Question {id: 5}) RETURN DISTINCT a.id") == []
    assert V.check(gr("answer"), "MATCH (a:Answer)-[:ANSWERS]->(:Question {id: 5}) RETURN DISTINCT a.id") == []


S05_001_QUERY = ("MATCH (:Answer)-[:ANSWERS]->(q:Question {id: 151334})\nRETURN DISTINCT a.id\n"
                 "MATCH (a:Answer)-[:ANSWERS]->(q:Question {id: 151334})\nRETURN DISTINCT a.id")
S05_005_QUERY = "MATCH (:Answer)-[:ANSWERS]->(q:Question {id: 459166})\nRETURN DISTINCT q.id"


def test_cypher_rejects_a_repeated_statement_s05_001():
    errs = V.check(gr("answer"), S05_001_QUERY)
    assert any("exactly one RETURN" in e for e in errs), errs


def test_cypher_rejects_union_and_two_returns():
    q = "MATCH (a:Answer) RETURN a.id UNION MATCH (b:Answer) RETURN b.id"
    assert any("exactly one RETURN" in e for e in V.check(gr("answer"), q))


def test_cypher_rejects_the_wrong_entity_s05_005():
    """The query is valid and runs, and it returns the question's own id: a silent wrong answer."""
    errs = V.check(gr("answer"), S05_005_QUERY)
    assert any("Question" in e and "answer" in e for e in errs), errs


def test_cypher_entity_check_accepts_correct_and_unlabelled_returns():
    assert V.check(gr("answer"), "MATCH (a:Answer)-[:ANSWERS]->(q:Question {id: 5}) RETURN DISTINCT a.id") == []
    assert V.check(gr("answer"), "MATCH (a)-[:ANSWERS]->(q:Question {id: 5}) RETURN DISTINCT a.id") == []
    assert V.check(gr("question"), "MATCH (:Answer)-[:ANSWERS]->(q:Question {id: 5}) RETURN DISTINCT q.id") == []
    assert V.check(gr("tag"), "MATCH (:Question {id: 5})-[:TAGGED_WITH]->(t:Tag) RETURN DISTINCT t.name") == []


def test_sql_rejects_an_id_from_the_wrong_table():
    errs = V.check(sql("user"), "SELECT p.id FROM posts p WHERE p.score > 3")
    assert any("'posts'" in e and "'users'" in e for e in errs), errs
    assert V.check(sql("user"), "SELECT u.id FROM users u WHERE u.reputation > 3") == []
    assert V.check(sql("answer"), "SELECT p.id FROM posts p WHERE p.post_type_id = 2") == []


# ---------------------------------------------------------------- Elasticsearch

def es(*filters, **extra):
    q = {"bool": {"filter": list(filters)}}
    return json.dumps({"query": q, **extra})


def test_doc_accepts_filter_and_search_and_bind():
    f = [{"term": {"doc_type": "question"}}, {"match": {"title": {"query": "regex", "operator": "and"}}}]
    assert V.check(doc(), es(*f)) == []
    assert V.check(doc(mode=Mode.SEARCH, k=5),
                   es({"term": {"doc_type": "question"}}, {"multi_match": {"query": "regex", "fields": ["title^2", "body"]}})) == []
    assert V.check(doc(bind=Bind("s0", "question")),
                   es({"term": {"doc_type": "question"}}, {"terms": {"post_id": "$bind:s0"}})) == []
    assert V.check(doc("user"), es({"term": {"doc_type": "user_about"}}, {"match": {"body": "teacher"}})) == []


@pytest.mark.parametrize("q,frag", [
    ("not json", "valid JSON"),
    (json.dumps({"query": {"match_all": {}}, "size": 5}), "exactly one key"),
    (es({"match": {"title": "x"}}), "doc_type"),
    (es({"term": {"doc_type": "answer"}}, {"match": {"body": "x"}}), "doc_type"),
    (es({"term": {"doc_type": "question"}}, {"match": {"nonsense": "x"}}), "unknown fields"),
    (es({"term": {"doc_type": "question"}}, {"script": {"script": "1"}}), "not allowed"),
    (es({"term": {"doc_type": "question"}}, {"query_string": {"query": "a AND b", "fields": ["title"]}}), "not allowed"),
    (es({"term": {"doc_type": "question"}}, {"terms": {"post_id": "$bind:s9"}}), "sentinels"),
])
def test_doc_rejections(q, frag):
    errs = V.check(doc(), q)
    assert any(frag in e for e in errs), errs


def test_doc_unknown_output_entity():
    assert any("no identifier" in e for e in V.check(doc("comment"), es({"term": {"doc_type": "comment"}})))
