"""Blind case sheets: what they show, what they must not show, and that checks-on files are refused."""
import json

import pytest

from benchmark_v2 import case_sheets as cs

PLAN = [{"id": "a", "op": "retrieve", "store": "graph", "question": "users who asked a question with the tag", "output": "user"},
        {"id": "b", "op": "retrieve", "store": "sql", "question": "users with reputation >= R", "output": "user",
         "bind": {"input": "a", "name": "s1", "entity": "user"}},
        {"id": "c", "op": "intersect", "inputs": ["a", "b"], "output": "user"}]
SECRET = "SECRET_CHECK_TEXT"


def q(i, gold=(1, 2, 3)):
    return {"question_id": f"Q{i}", "text": f"Which users asked about topic {i}?", "answer_type": "ids",
            "gold": list(gold), "plan": PLAN, "template": "D01"}


def row(i, outcome="wrong", checks=False, got=(1, 2, 9)):
    return {"question_id": f"Q{i}", "template": "D01", "text": f"Which users asked about topic {i}?", "outcome": outcome,
            "guarantee": "exact", "answer_type": "ids", "plan": {"steps": PLAN}, "gold_plan": PLAN,
            "queries": {"a": "MATCH (u:User) RETURN u.id", "b": "SELECT id FROM users WHERE reputation >= 5"},
            "got": list(got), "answer_digest": {"type": "ids", "size": len(got), "head": list(got)},
            "semantic_checks": checks, "translation_attempt_errors": {"b": [SECRET]}, "step_errors": [SECRET],
            "translation_errors": {}, "model": "mistral-secret"}


def write(tmp_path, name, rows, meta=None):
    p = tmp_path / name
    p.write_text(json.dumps({"meta": meta or {}, "rows": rows}))
    return str(p)


def questions(n):
    return {f"Q{i}": q(i) for i in range(n)}


def test_a_classify_sheet_shows_the_plans_queries_and_the_answer_against_the_key(tmp_path):
    f = write(tmp_path, "run_mistral.json", [row(0), row(1, "correct", got=(1, 2, 3))])
    sheets, key = cs.build([f], questions(2), "classify", 0, 7)
    text = "\n".join(sheets)
    assert len(sheets) == 1 and key["S001"]["question_id"] == "Q0"
    assert "Which users asked about topic 0?" in text and "connections cabinet (graph)" in text
    assert "must look only at the ids handed over from step a" in text and "keep the items found by both steps" in text
    assert "SELECT id FROM users WHERE reputation >= 5" in text
    assert "In the system answer but not in the correct answer: 9 (1 items)" in text
    assert "In the correct answer but not in the system answer: 3 (1 items)" in text


def test_the_sheet_leaks_nothing_about_the_model_the_file_rejected_drafts_or_the_checks(tmp_path):
    f = write(tmp_path, "run_mistral.json", [row(0)])
    sheets, _ = cs.build([f], questions(1), "classify", 0, 7)
    text = "\n".join(sheets)
    for hidden in (SECRET, "mistral", "run_", "translation_attempt", "step_errors", "semantic"):
        assert hidden not in text


def test_a_result_file_made_with_the_checks_on_is_refused(tmp_path):
    f = write(tmp_path, "on.json", [row(0, checks=True)])
    with pytest.raises(ValueError, match="semantic checks on"):
        cs.build([f], questions(1), "classify", 0, 7)
    g = write(tmp_path, "on_meta.json", [row(0)], meta={"semantic_checks": True})
    with pytest.raises(ValueError):
        cs.build([g], questions(1), "classify", 0, 7)


def test_order_is_shuffled_by_the_seed_and_the_same_seed_gives_the_same_order(tmp_path):
    f = write(tmp_path, "r.json", [row(i) for i in range(8)])
    a = cs.build([f], questions(8), "classify", 0, 1)[1]
    b = cs.build([f], questions(8), "classify", 0, 1)[1]
    c = cs.build([f], questions(8), "classify", 0, 2)[1]
    assert a == b and [v["question_id"] for v in a.values()] != [v["question_id"] for v in c.values()]
    assert sorted(a) == [f"S{i:03d}" for i in range(1, 9)]


def test_a_wording_sheet_has_one_item_per_question_mixes_correct_ones_and_shows_no_system_output(tmp_path):
    f1 = write(tmp_path, "a.json", [row(0), row(1, "correct", got=(1, 2, 3)), row(2, "correct", got=(1, 2, 3))])
    f2 = write(tmp_path, "b.json", [row(0), row(1, "correct", got=(1, 2, 3)), row(2, "correct", got=(1, 2, 3))])
    sheets, key = cs.build([f1, f2], questions(3), "wording", 2, 5)
    text = "\n".join(sheets)
    assert len(sheets) == 3 and sorted(v["question_id"] for v in key.values()) == ["Q0", "Q1", "Q2"]
    assert "SELECT" not in text and "System" not in text and "9" not in text.replace("topic", "")
    assert "**Correct answer shown:** 1, 2, 3 (3 items)" in text


def test_long_answers_are_cut_and_a_truncated_record_is_said_to_be(tmp_path):
    big = list(range(100, 140))
    r = row(0, got=tuple(range(20)))
    r["answer_digest"] = {"type": "ids", "size": 40, "head": list(range(20))}
    f = write(tmp_path, "r.json", [r])
    ques = {"Q0": q(0, gold=big)}
    text = "\n".join(cs.build([f], ques, "classify", 0, 7)[0])
    assert "(40 items in all)" in text and "only the first 20 of the system's 40 items are recorded" in text


def test_main_writes_the_sheets_the_form_and_the_key(tmp_path, capsys):
    f = write(tmp_path, "r.json", [row(0), row(1)])
    qf = tmp_path / "q.jsonl"
    qf.write_text("\n".join(json.dumps(v) for v in questions(2).values()))
    out = tmp_path / "sheets"
    assert cs.main([f, "--questions", str(qf), "--kind", "classify", "--out-prefix", str(out)]) == 0
    assert (tmp_path / "sheets.md").read_text().count("## Case") == 2
    assert (tmp_path / "sheets_form.csv").read_text().splitlines()[0].startswith("case_id,class")
    assert set(json.loads((tmp_path / "sheets.key.json").read_text())) == {"S001", "S002"}
    assert cs.main([f, "--questions", str(tmp_path / "nope.jsonl"), "--out-prefix", str(out)]) == 2
