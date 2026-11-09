"""The replay counts firings of the candidate checks by recorded outcome, with no model call."""
import json

from benchmark_v2.natural_error_checks import Signatures
from benchmark_v2.replay_checks import main, outcome_class, render, replay_rows

SIGS = Signatures.poc()
PLAN = {"steps": [
    {"id": "s1", "op": "retrieve", "store": "graph", "question": "q", "output": "question", "kind": "set"},
    {"id": "s2", "op": "retrieve", "store": "sql", "question": "q", "inputs": ["s1"],
     "bind": {"input": "s1", "entity": "question"}, "output": "user", "kind": "set"}]}
GOOD_S1 = "MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE toLower(t.name) = toLower('dlm') RETURN DISTINCT q.id"
BAD_S2 = "SELECT users.id FROM users WHERE users.id = ANY(%(s1)s) AND users.reputation >= 61"
GOOD_S2 = "SELECT DISTINCT posts.id FROM posts WHERE posts.id = ANY(%(s1)s)"


def row(qid, outcome, s1, s2, plan=PLAN):
    return {"question_id": qid, "outcome": outcome, "executed_plan": plan, "queries": {"s1": s1, "s2": s2}}


ROWS = [row("A", "correct", GOOD_S1, GOOD_S2),
        row("B", "wrong", GOOD_S1, BAD_S2),
        row("C", "abstain_unexpected", GOOD_S1, BAD_S2),
        {"question_id": "D", "outcome": "abstain_unexpected", "plan": None, "queries": {}}]


def test_outcome_classes():
    assert [outcome_class(r) for r in ROWS] == ["correct", "silent", "other", "other"]
    assert outcome_class({"outcome": "partial_violated"}) == "silent"          # a false SUBSET label is a silent error
    assert outcome_class({"outcome": "partial_valid"}) == "other"


def test_counts_are_split_by_outcome_and_unreadable_rows_are_skipped():
    counts, firings, skipped = replay_rows(ROWS, SIGS)
    assert counts["R2"] == {"silent": 1, "other": 1} and skipped == 1
    assert counts["R1"] == {} and counts["R3"] == {}
    assert {f["question_id"] for f in firings} == {"B", "C"} and all(f["check"] == "R2" for f in firings)
    text = render("x.json", ROWS, counts, skipped)
    assert "R2" in text and "1 skipped" in text


def test_command_line_prints_a_table_and_the_requested_firings(tmp_path, capsys):
    f = tmp_path / "rows.json"
    f.write_text(json.dumps({"rows": ROWS}))
    assert main([str(f), "--show", "silent"]) == 0
    out = capsys.readouterr().out
    assert "rows.json: 4 rows (1 correct, 1 silent, 2 other)" in out and "[rows.json] B step s2 R2 (wrong)" in out
