"""Tests for the cross-store fidelity check (benchmark_v2/fidelity.py).

The live-store readers need running services and are exercised by the user's first real
run. These tests cover everything that does not: the expected values derived from the
oracle (checked by hand against the synthetic dump), the comparison logic, the
ownership rule definitions, and the CLI exit code.
"""
import sqlite3
from pathlib import Path

from benchmark_v2 import fidelity, oracle
from tests.benchmark_v2.test_oracle import make_dump


def _db(tmp_path: Path) -> sqlite3.Connection:
    dump = make_dump(tmp_path / "dump")
    path = tmp_path / "oracle.sqlite"
    oracle.build(dump, path, log=lambda s: None)
    return sqlite3.connect(path)


def test_expected_counts_follow_loader_semantics(tmp_path):
    e = fidelity.expected_counts(_db(tmp_path))
    # users: the Community user (id -1) is not loaded
    assert e["postgres.users"] == 2 and e["neo4j.node.User"] == 2
    assert e["postgres.posts"] == 5 and e["postgres.votes"] == 2
    assert e["postgres.badges"] == 1 and e["postgres.comments"] == 1 and e["postgres.tags"] == 2
    assert (e["neo4j.node.Question"], e["neo4j.node.Answer"], e["neo4j.node.Tag"]) == (3, 1, 2)
    assert e["neo4j.rel.ASKED"] == 3 and e["neo4j.rel.ANSWERED"] == 1
    assert e["neo4j.rel.ANSWERS"] == 1 and e["neo4j.rel.ACCEPTED"] == 1
    # post 3 is tagged r and regression, which are not in Tags.xml, so no edge is created
    assert e["neo4j.rel.TAGGED_WITH"] == 2
    assert e["neo4j.rel.CO_OCCURS_WITH"] == 1
    assert e["neo4j.rel.LINKED_TO"] == 1 and e["neo4j.rel.DUPLICATE_OF"] == 1
    assert e["elasticsearch.doc.question"] == 3 and e["elasticsearch.doc.answer"] == 1
    assert e["elasticsearch.doc.comment"] == 1 and e["elasticsearch.doc.user_about"] == 1
    assert e["elasticsearch.doc.tag_wiki"] == 1       # excerpt post 4 has a body


def test_expected_totals(tmp_path):
    t = fidelity.expected_totals(_db(tmp_path))
    assert t["postgres.posts.sum_score"] == 13
    assert t["postgres.posts.sum_view_count"] == 106
    assert t["postgres.users.sum_reputation"] == 1520        # id -1 excluded
    assert t["postgres.comments.sum_score"] == 2
    assert t["postgres.badges.sum_class"] == 3


def test_expected_neighbourhoods(tmp_path):
    db = _db(tmp_path)
    q1 = fidelity.expected_question_neighbourhood(db, 1)
    assert q1 == {"askers": [2], "tags": ["pca", "python"], "accepted": [2], "answers": [2],
                  "linked": [], "dups": []}
    q3 = fidelity.expected_question_neighbourhood(db, 3)
    assert q3["tags"] == [] and q3["linked"] == [1] and q3["askers"] == [3]
    assert fidelity.expected_question_neighbourhood(db, 5)["dups"] == [1]
    assert fidelity.expected_answer_neighbourhood(db, 2) == {"authors": [3], "questions": [1]}


def test_expected_post_records(tmp_path):
    db = _db(tmp_path)
    assert fidelity.expected_post_metrics(db, 1) == (1, 7, 100, 1, 1)
    assert fidelity.expected_post_metrics(db, 999) == ()
    doc = fidelity.expected_es_post_doc(db, 1)
    assert doc == {"doc_type": "question", "post_id": 1, "body": "<p>What is PCA?</p>",
                   "title": "What is PCA?"}
    assert fidelity.expected_es_post_doc(db, 2) == {"doc_type": "answer", "post_id": 2,
                                                    "body": "Rotation of axes."}
    assert fidelity.expected_es_post_doc(db, 999) is None


def test_sample_is_deterministic_and_bounded(tmp_path):
    db = _db(tmp_path)
    a, b = fidelity.sample_ids(db, 2), fidelity.sample_ids(db, 2)
    assert a == b and len(a["questions"]) == 2 and a["answers"] == [2]
    assert fidelity.sample_ids(db, 100)["questions"] == [1, 3, 5]


def test_compare_numbers_reports_difference_and_missing():
    out = fidelity.compare_numbers("x", "count", {"a": 5, "b": 3, "c": 1}, {"a": 5, "b": 2})
    assert [c.ok for c in out] == [True, False, False]
    assert out[1].detail == "diff -1" and out[2].detail == "missing"


def test_ownership_comparison_flags_extra_and_missing_fields():
    ok = fidelity.compare_exact_sets("postgres", "t", {"id", "score"}, {"id", "score"})
    assert ok.ok
    bad = fidelity.compare_exact_sets("postgres", "t", {"id", "score"}, {"id", "score", "owner_user_id"})
    assert not bad.ok and "owner_user_id" in bad.detail
    missing = fidelity.compare_exact_sets("postgres", "t", {"id", "score"}, {"id"})
    assert not missing.ok and "score" in missing.detail


def test_compare_records_names_the_first_difference():
    exp = {1: {"tags": ["a"]}, 2: {"tags": ["b"]}}
    assert fidelity.compare_records("neo4j", "s", exp, dict(exp)).ok
    res = fidelity.compare_records("neo4j", "s", exp, {1: {"tags": ["a"]}, 2: {"tags": ["z"]}})
    assert not res.ok and res.observed == 1 and "first id 2" in res.detail


def test_ownership_definitions_encode_the_rule():
    assert "owner_user_id" not in fidelity.PG_COLUMNS["posts"]
    assert not {"parent_id", "accepted_answer_id", "tags", "title"} & fidelity.PG_COLUMNS["posts"]
    assert "about_me" not in fidelity.PG_COLUMNS["users"]
    assert "post_links" not in fidelity.PG_COLUMNS
    assert fidelity.NEO4J_NODE_PROPERTIES["Question"] == {"id"}
    assert fidelity.NEO4J_NODE_PROPERTIES["Tag"] == {"name"}
    assert not {"score", "tags", "parent_id", "creation_date"} & fidelity.ES_FIELDS
    # event records keep their identifiers (decision D10)
    assert {"post_id", "user_id"} <= fidelity.PG_COLUMNS["votes"] | fidelity.PG_COLUMNS["comments"]
    assert {"post_id", "user_id"} <= fidelity.ES_DOC_FIELDS["comment"]


def test_run_and_exit_code_with_injected_readers(tmp_path, capsys):
    dump = make_dump(tmp_path / "dump")
    path = tmp_path / "oracle.sqlite"
    oracle.build(dump, path, log=lambda s: None)

    def good(db, ids):
        return [fidelity.Check("postgres", "count", "postgres.users", 2, 2, True)]

    def bad(db, ids):
        return [fidelity.Check("neo4j", "count", "neo4j.node.User", 2, 3, False, "diff +1")]

    assert fidelity.run(str(path), None, 10, readers={"postgres": good}) [0].ok
    only = fidelity.run(str(path), "neo4j", 10, readers={"postgres": good, "neo4j": bad})
    assert len(only) == 1 and only[0].store == "neo4j"
    text = fidelity.render(only)
    assert "FAIL" in text and "1 FAILED" in text


def test_fidelity_does_not_import_the_system_under_test():
    import ast
    tree = ast.parse(Path(fidelity.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"heterorag", "elasticsearch", "scripts"})
