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
REF = StepValidator(catalogue(), case_insensitive_text=False)      # reference queries are exact-match ground truth


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
            assert REF.check(step, text) == [], (r["template"], sid, text)
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
    assert V.check(sql("user"), "SELECT DISTINCT b.user_id AS id FROM badges b WHERE lower(b.name) = lower('Teacher')") == []
    assert V.check(sql("question", kind=Kind.SET, bind=Bind("s0", "question")),
                   "SELECT id FROM posts WHERE id = ANY(%(s0)s) AND score > 3") == []
    assert V.check(Step("s1", Op.RETRIEVE, Store.SQL, None, "q", output="number", kind=Kind.SCALAR),
                   "SELECT reputation FROM users WHERE id = 5") == []


def test_sql_comment_and_literal_do_not_hide_or_trigger_checks():
    assert V.check(sql("user"), "SELECT id FROM users WHERE lower(display_name) = lower('drop; delete')") == []
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
    assert V.check(gr("question"), "MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE toLower(t.name) = toLower('python') RETURN DISTINCT q.id") == []
    assert V.check(gr("tag", Kind.RANKED, k=3),
                   "MATCH (a:Tag)-[r:CO_OCCURS_WITH]-(o:Tag) WHERE toLower(a.name) = toLower('x') RETURN o.name, r.weight ORDER BY r.weight DESC, o.name ASC LIMIT 3") == []
    assert V.check(gr("question", bind=Bind("s0", "user")),
                   "MATCH (u:User)-[:ASKED]->(q:Question) WHERE u.id IN $s0 RETURN DISTINCT q.id") == []
    assert V.check(gr("question"), "MATCH (q:Question) WHERE q.id = 7 // set aside\nRETURN q.id") == []
    # variables from relationships, aliases, paths and WITH
    assert V.check(gr("tag", Kind.RANKED, k=2),
                   "MATCH (a:Tag)-[r:CO_OCCURS_WITH]-(o:Tag) WHERE toLower(a.name) = toLower('x') WITH o, r.weight AS w RETURN o.name, w ORDER BY w DESC, o.name ASC LIMIT 2") == []
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


# ---------------------------------------------------------------- text is compared without regard to case

@pytest.mark.parametrize("q", [
    "SELECT DISTINCT user_id FROM badges WHERE name = 'reliability'",
    "SELECT DISTINCT b.user_id FROM badges b WHERE b.name = 'Reliability'",
    "SELECT DISTINCT user_id FROM badges WHERE 'reliability' = name",
    "SELECT DISTINCT user_id FROM badges WHERE name IN ('caret', 'survival')",
    "SELECT DISTINCT user_id FROM badges WHERE name = lower('Caret')",         # the column side is not wrapped
    "SELECT DISTINCT user_id FROM badges WHERE name <> 'caret'",
])
def test_sql_rejects_a_text_column_compared_directly_with_a_literal(q):
    errs = V.check(sql("user"), q)
    assert any("without regard to case" in e and "lower(" in e for e in errs), errs


@pytest.mark.parametrize("q", [
    "SELECT DISTINCT user_id FROM badges WHERE lower(name) = lower('Reliability')",
    "SELECT DISTINCT b.user_id FROM badges b WHERE lower(b.name) = lower('caret')",
    "SELECT DISTINCT user_id FROM badges WHERE name ILIKE 'caret'",
    "SELECT id FROM users WHERE reputation = 5 AND strpos(lower(display_name), lower('Bob')) > 0",
    "SELECT id FROM users WHERE display_name = '2020-01-01'",                 # a date or number, not a name
    "SELECT id FROM posts WHERE post_type_id = 1 AND creation_date >= '2019-01-01'",
    "SELECT id FROM users WHERE id = 5",
])
def test_sql_accepts_case_insensitive_and_non_text_comparisons(q):
    assert not [e for e in V.check(sql("user"), q) if "without regard to case" in e]


def test_sql_literal_inside_a_comment_or_another_string_is_not_a_comparison():
    q = "SELECT id FROM users WHERE lower(display_name) = lower('a') -- name = 'x'"
    assert not [e for e in V.check(sql("user"), q) if "without regard to case" in e]
    q = "SELECT id FROM users WHERE lower(display_name) = lower('name = ''x''')"
    assert not [e for e in V.check(sql("user"), q) if "without regard to case" in e]


@pytest.mark.parametrize("q,frag", [
    ('MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag {name: "python"}) RETURN DISTINCT q.id', "property map"),
    ("MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE t.name = 'python' RETURN DISTINCT q.id", "toLower(t.name)"),
    ('MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE t.name <> "python" RETURN DISTINCT q.id', "toLower(t.name)"),
    ("MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE t.name CONTAINS 'py' RETURN DISTINCT q.id", "toLower(t.name)"),
    ("MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE t.name STARTS WITH 'py' RETURN DISTINCT q.id", "toLower(t.name)"),
    ("MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE t.name IN ['a', 'b'] RETURN DISTINCT q.id", "IN"),
])
def test_cypher_rejects_a_string_compared_directly_with_a_property(q, frag):
    errs = V.check(gr("question"), q)
    assert any("case" in e and frag in e for e in errs), errs


@pytest.mark.parametrize("q", [
    "MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE toLower(t.name) = toLower('Python') RETURN DISTINCT q.id",
    "MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE toLower(t.name) CONTAINS toLower('py') RETURN DISTINCT q.id",
    "MATCH (q:Question {id: 5})-[:TAGGED_WITH]->(t:Tag) RETURN DISTINCT q.id",          # an integer in a map
    "MATCH (q:Question) WHERE q.id IN [1, 2] RETURN DISTINCT q.id",
    "MATCH (q:Question) WHERE q.creation_date = '2020-01-01' RETURN DISTINCT q.id",     # a date, not a name
])
def test_cypher_accepts_case_insensitive_and_non_text_conditions(q):
    assert not [e for e in V.check(gr("question"), q) if "case" in e]


def _doc_query(*clauses):
    return json.dumps({"query": {"bool": {"filter": [{"term": {"doc_type": "tag_wiki"}}, *clauses]}}})


def test_doc_rejects_a_clause_on_a_keyword_field_that_is_case_sensitive():
    for clause in ({"term": {"tag_name": "python"}}, {"term": {"tag_name": {"value": "python"}}},
                   {"terms": {"tag_name": ["python", "r"]}}, {"match": {"tag_name": "python"}},
                   {"term": {"tag_name": {"value": "python", "case_insensitive": False}}}):
        errs = V.check(doc("tag"), _doc_query(clause))
        assert any("case-sensitive" in e and "tag_name" in e for e in errs), (clause, errs)


def test_doc_accepts_a_case_insensitive_keyword_clause_text_matches_doc_type_and_bind_sentinels():
    ok = {"term": {"tag_name": {"value": "Python", "case_insensitive": True}}}
    assert not [e for e in V.check(doc("tag"), _doc_query(ok)) if "case" in e]
    assert not [e for e in V.check(doc("tag"), _doc_query({"match": {"body": {"query": "Bayes", "operator": "and"}}}))
                if "case" in e]
    bound = _doc_query({"terms": {"tag_name": "$bind:s0"}})
    assert not [e for e in V.check(doc("tag", bind=Bind("s0", "tag")), bound) if "case" in e]


def test_the_case_rule_can_be_switched_off_for_exact_match_ground_truth():
    q = "SELECT DISTINCT user_id FROM badges WHERE name = 'b8'"
    assert V.check(sql("user"), q) and REF.check(sql("user"), q) == []
    g = 'MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag {name: "python"}) RETURN DISTINCT q.id'
    assert V.check(gr("question"), g) and REF.check(gr("question"), g) == []


def test_the_catalogue_knows_which_columns_and_fields_are_text():
    cat = catalogue()
    assert {"name", "tag_name", "display_name", "location"} <= cat.sql.text_columns
    assert not ({"id", "reputation", "creation_date", "class", "tag_based"} & cat.sql.text_columns)
    assert "tag_name" in cat.doc.keyword_fields and "body" not in cat.doc.keyword_fields
