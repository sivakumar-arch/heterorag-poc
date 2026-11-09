"""The semantic checks (C0, R1 to R3) are off by default and change nothing then; with the flag they reject the
query shapes that weak local models wrote (CP1b floor runs) and still accept the reference queries."""
import pytest

from benchmark_v2 import questions
from benchmark_v2.reference_queries import reference_queries
from heterorag.federation import Bind, Op, Plan, Step, Store
from heterorag.federation.guarantee import Kind
from heterorag.federation.llm_config import semantic_checks_enabled
from heterorag.federation.step_validator import StepValidator
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)
from tests.federation.fake_descriptors import catalogue
from tests.federation.test_reference_and_validation import records_for

OFF = StepValidator(catalogue())
ON = StepValidator(catalogue(), semantic_checks=True)
REF_ON = StepValidator(catalogue(), case_insensitive_text=False, semantic_checks=True)


def bound(store, output="user", entity="question"):
    return Step("s2", Op.RETRIEVE, store, None, "q", inputs=("s1",), bind=Bind("s1", entity), output=output,
                kind=Kind.SET)


def plain(store, output="user"):
    return Step("s1", Op.RETRIEVE, store, None, "q", output=output, kind=Kind.SET)


BIND_WRONG_ENTITY = ("SELECT users.id FROM users JOIN posts ON users.id = ANY(%(s1)s) AND posts.post_type_id = 1 "
                     "AND users.reputation >= 123354")
NEUTRALISED = "SELECT id FROM users WHERE reputation >= 3648 AND (id = ANY(%(s1)s) OR TRUE)"
REVERSED = ("MATCH (t:Tag)-[:TAGGED_WITH]->(q:Question) WHERE toLower(t.name) = toLower('social-network') "
            "RETURN DISTINCT q.id")
RENAMED = "SELECT DISTINCT name AS id FROM badges WHERE name = 'Teacher'"


def msgs(v, step, q):
    return " | ".join(v.check(step, q))


def test_default_is_off_and_the_environment_switch_works():
    assert semantic_checks_enabled({}) is False
    assert semantic_checks_enabled({"HETERORAG_SEMANTIC_CHECKS": "0"}) is False
    for v in ("1", "true", "ON", "Yes"):
        assert semantic_checks_enabled({"HETERORAG_SEMANTIC_CHECKS": v}) is True
    assert StepValidator(catalogue()).semantic is False


@pytest.mark.parametrize("step,q,frag", [
    (bound(Store.SQL), BIND_WRONG_ENTITY, "are question identifiers, but they are compared"),
    (bound(Store.SQL), NEUTRALISED, "vacuous"),
    (plain(Store.GRAPH, "question"), REVERSED, "relationship direction"),
    (plain(Store.SQL), RENAMED, "not an identifier column"),
])
def test_the_flag_adds_exactly_these_rejections(step, q, frag):
    assert frag not in msgs(OFF, step, q)
    assert frag in msgs(ON, step, q)


def test_flag_off_returns_the_same_errors_as_before_for_every_probe_query():
    for step, q in [(bound(Store.SQL), BIND_WRONG_ENTITY), (bound(Store.SQL), NEUTRALISED),
                    (plain(Store.GRAPH, "question"), REVERSED), (plain(Store.SQL), RENAMED)]:
        assert OFF.check(step, q) == StepValidator(catalogue(), semantic_checks=False).check(step, q)


@pytest.mark.parametrize("output,q", [
    ("user", "SELECT DISTINCT user_id AS id FROM badges WHERE name = 'Teacher'"),
    ("user", "SELECT DISTINCT owner_user_id AS id FROM posts WHERE score > 3"),
    ("question", "SELECT post_id AS id FROM comments WHERE score > 3"),
    ("tag", "SELECT tag AS tag_name FROM post_tags"),
    ("user", "SELECT count(*) AS id FROM badges"),
])
def test_identifier_columns_and_expressions_still_pass_with_the_flag(output, q):
    assert [e for e in ON.check(plain(Store.SQL, output), q) if "identifier" in e] == []


def test_messages_are_not_duplicated():
    errs = ON.check(bound(Store.SQL), NEUTRALISED)
    assert len(errs) == len(set(errs))


def test_every_reference_query_passes_with_the_flag_on(world):
    recs = records_for(world, [t.id for t in questions.all_templates()], per=3)
    for r in recs:
        plan = Plan.from_dict({"steps": r["plan"]})
        for sid, text in reference_queries(r["template"], r["params"]).items():
            assert REF_ON.check(plan.by_id()[sid], text) == [], (r["template"], sid, text)
