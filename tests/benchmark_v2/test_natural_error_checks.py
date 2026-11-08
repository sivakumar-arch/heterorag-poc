"""The candidate checks against queries that weak local models actually wrote (CP1b floor runs, 2026-11-07)
and against the shapes the reference queries use."""
import pytest

from benchmark_v2 import questions
from benchmark_v2.natural_error_checks import (Signatures, candidate_errors, cypher_bind_entity_errors,
                                               neutralised_condition_errors, relationship_signature_errors,
                                               sql_bind_entity_errors)
from benchmark_v2.reference_queries import reference_queries
from heterorag.federation import Bind, Op, Plan, Step, Store
from heterorag.federation.guarantee import Kind
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)
from tests.federation.test_reference_and_validation import records_for

SIGS = Signatures.poc()

# --- queries copied from the recorded rows (silent errors) -------------------------------------------------
QWEN_D01_021 = ("MATCH (t:Tag)-[:TAGGED_WITH]->(:Question)<-[:ASKED]-(u:User) WHERE toLower(t.name) = "
                "toLower('repeated-measures') AND u.id IN $s1 RETURN DISTINCT u.id")
QWEN_D04_022 = ("MATCH (u:User) WHERE u.id IN $s1 MATCH (q:Question)-[:ASKED]->(u) MATCH (q)-[:TAGGED_WITH]->(t:Tag) "
                "RETURN DISTINCT t.name")
GPTOSS_D05_021 = ("MATCH (t:Tag)-[:TAGGED_WITH]->(q:Question) WHERE toLower(t.name) = toLower('social-network') "
                  "RETURN DISTINCT q.id")
MISTRAL_D01_001 = ("SELECT users.id FROM users JOIN posts ON users.id = ANY(%(s1)s) AND posts.post_type_id = 1 "
                   "AND users.reputation >= 123354")
MISTRAL_D01_032 = ("SELECT DISTINCT users.id FROM users JOIN posts ON users.id = ANY(%(s1)s) AND posts.post_type_id = 1 "
                   "AND posts.id = ANY(%(s1)s) WHERE users.reputation >= 61")
GPTOSS_D01_003 = "SELECT users.id FROM users WHERE reputation >= 3648 AND (users.id = ANY(%(s1)s) OR TRUE);"
GPTOSS_D01_015 = "SELECT id FROM users WHERE reputation >= 867 AND (TRUE OR id = ANY(%(s1)s));"

# --- queries that must pass -------------------------------------------------------------------------------
GOOD_CYPHER = [
    "MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE toLower(t.name) = toLower('dlm') RETURN DISTINCT q.id",
    "MATCH (q:Question)<-[:ASKED]-(u:User) WHERE q.id IN $s1 RETURN DISTINCT u.id",
    "MATCH (u:User)-[:ASKED]->(q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE u.id IN $s1 RETURN DISTINCT t.name",
    "MATCH (a:Tag)-[:CO_OCCURS_WITH]-(b:Tag) WHERE a.name = 'x' RETURN DISTINCT b.name",
]


def step(store, entity=None, name="s1"):
    return Step("s2", Op.RETRIEVE, store, None, "q", inputs=(name,) if entity else (),
                bind=Bind(name, entity) if entity else None, output="user", kind=Kind.SET)


@pytest.mark.parametrize("q", [QWEN_D01_021, QWEN_D04_022, GPTOSS_D05_021])
def test_reversed_relationships_are_flagged(q):
    assert relationship_signature_errors(q, SIGS), q


@pytest.mark.parametrize("q", GOOD_CYPHER)
def test_correct_relationship_patterns_pass(q):
    assert relationship_signature_errors(q, SIGS) == []


def test_the_message_names_the_store_triple_and_what_was_read():
    msg = relationship_signature_errors(GPTOSS_D05_021, SIGS)[0]
    assert "Question-[:TAGGED_WITH]->Tag" in msg and "Tag-[:TAGGED_WITH]->Question" in msg


@pytest.mark.parametrize("q", [MISTRAL_D01_001, MISTRAL_D01_032])
def test_question_identifiers_compared_with_users_id_are_flagged(q):
    errs = sql_bind_entity_errors(q, "s1", "question")
    assert errs and "users.id" in errs[0], errs


def test_a_user_bind_against_users_id_and_a_question_bind_against_posts_id_pass():
    assert sql_bind_entity_errors("SELECT id FROM users WHERE id = ANY(%(s1)s)", "s1", "user") == []
    assert sql_bind_entity_errors("SELECT p.id FROM posts p WHERE p.id = ANY(%(s0)s) AND p.score > 3", "s0", "question") == []
    assert sql_bind_entity_errors("SELECT user_id FROM badges WHERE user_id = ANY(%(s1)s)", "s1", "user") == []


def test_a_cypher_bind_on_the_wrong_node_kind_is_flagged_and_the_right_one_passes():
    assert cypher_bind_entity_errors("MATCH (u:User) WHERE u.id IN $s1 RETURN DISTINCT u.id", "s1", "question")
    assert cypher_bind_entity_errors("MATCH (q:Question) WHERE q.id IN $s1 RETURN DISTINCT q.id", "s1", "question") == []


@pytest.mark.parametrize("q,comment", [(GPTOSS_D01_003, "--"), (GPTOSS_D01_015, "--"),
                                       ("MATCH (u:User) WHERE u.reputation > 5 OR true RETURN u.id", "//")])
def test_a_tautology_in_a_condition_is_flagged(q, comment):
    assert neutralised_condition_errors(q, comment)


def test_text_that_only_contains_the_word_true_in_a_literal_passes():
    assert neutralised_condition_errors("SELECT id FROM users WHERE display_name = 'or true'") == []
    assert neutralised_condition_errors("SELECT id FROM users WHERE reputation > 5 AND tag_based = true") == []


def test_candidate_errors_routes_by_store():
    g = candidate_errors(step(Store.GRAPH, "user"), QWEN_D01_021, SIGS)
    assert g["R1"] and not g["R2"] and not g["R3"]
    s = candidate_errors(step(Store.SQL, "question"), MISTRAL_D01_001, SIGS)
    assert s["R2"] and not s["R1"]
    s = candidate_errors(step(Store.SQL, "user"), GPTOSS_D01_003, SIGS)
    assert s["R3"] and not s["R2"]


def test_no_reference_query_is_flagged(world):
    for rec in records_for(world, [t.id for t in questions.all_templates()], per=3):
        plan = Plan.from_dict({"steps": rec["plan"]})
        for sid, text in reference_queries(rec["template"], rec["params"]).items():
            st = plan.by_id()[sid]
            got = candidate_errors(st, text, SIGS)
            assert not any(got.values()), (rec["template"], sid, text, got)
