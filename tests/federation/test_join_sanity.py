"""R5: tables must be joined by column conditions; identifier columns of different entities cannot be equal."""
import pytest

from heterorag.federation import Bind, Op, Step, Store
from heterorag.federation.guarantee import Kind
from heterorag.federation.semantic_checks import join_sanity_errors
from heterorag.federation.step_validator import StepValidator
from tests.federation.fake_descriptors import catalogue

# recorded from the live sanity run (Mistral, 2026-11-09): no condition links users to posts
CARTESIAN_1 = ("SELECT users.id FROM users JOIN posts ON posts.id = ANY(%(s1)s) AND posts.post_type_id = 1 "
               "AND users.reputation >= 123354")
CARTESIAN_2 = ("SELECT DISTINCT users.id FROM users JOIN posts ON posts.id = ANY(%(s1)s) WHERE posts.post_type_id = 1 "
               "AND users.reputation >= 61")


@pytest.mark.parametrize("q", [CARTESIAN_1, CARTESIAN_2,
                               "SELECT users.id FROM users, badges WHERE badges.name = 'Teacher'"])
def test_a_cartesian_product_is_flagged(q):
    errs = join_sanity_errors(q)
    assert errs and "not all connected" in errs[0]


@pytest.mark.parametrize("q", [
    "SELECT DISTINCT users.id FROM users JOIN posts ON posts.owner_user_id = users.id WHERE posts.id = ANY(%(s1)s)",
    "SELECT DISTINCT u.id FROM users u JOIN posts p ON p.owner_user_id = u.id JOIN comments c ON c.post_id = p.id "
    "WHERE c.score > 3",
    "SELECT id FROM users WHERE reputation > 5",
    "SELECT id FROM posts WHERE post_type_id = 1 AND EXTRACT(YEAR FROM creation_date) = 2017",
    "SELECT id FROM users WHERE id IN (SELECT user_id FROM badges WHERE name = 'x')",     # sub-select: not judged
    "SELECT users.id FROM users JOIN badges USING (id) WHERE badges.name = 'x'",           # USING: not judged
    "SELECT DISTINCT user_id AS id FROM badges WHERE name = 'x'",
])
def test_connected_or_not_judged_queries_pass(q):
    assert join_sanity_errors(q) == []


def test_equality_between_identifiers_of_different_entities_is_a_type_error():
    errs = join_sanity_errors("SELECT users.id FROM users JOIN posts ON posts.id = users.id WHERE posts.score > 1")
    assert any("post identifier" in e and "user identifier" in e for e in errs)
    assert join_sanity_errors("SELECT users.id FROM users JOIN badges ON badges.user_id = users.id") == []


def test_the_validator_applies_it_only_with_the_flag():
    step = Step("s2", Op.RETRIEVE, Store.SQL, None, "q", inputs=("s1",), bind=Bind("s1", "question"),
                output="user", kind=Kind.SET)
    on, off = StepValidator(catalogue(), semantic_checks=True), StepValidator(catalogue())
    assert any("not all connected" in e for e in on.check(step, CARTESIAN_1))
    assert not any("not all connected" in e for e in off.check(step, CARTESIAN_1))


def test_the_message_names_no_column_of_any_schema():
    # a hint that names a column the service does not expose sends the translator to a column the schema check then rejects
    errs = join_sanity_errors("SELECT users.id FROM users JOIN posts ON posts.id = ANY(%(s1)s)")
    assert errs and "owner_user_id" not in " ".join(errs) and "CANNOT_ANSWER" in " ".join(errs)
