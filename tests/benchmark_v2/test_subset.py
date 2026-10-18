"""Deriving a run's question file: exclusion of used questions, id prefix, per-template limit."""
import json

from benchmark_v2 import subset as ss


def rec(t, i, text=None):
    return {"template": t, "question_id": f"{t}-{i:03d}", "text": text or f"{t} question {i}", "gold": [i]}


def test_excluded_questions_are_dropped_by_template_and_text_not_by_id():
    recs = [rec("D01", 1), rec("D01", 2), rec("D04", 1)]
    used = {("D01", "D01 question 2"), ("D04", "other text")}
    kept, dropped, dup = ss.subset(recs, used)
    assert [r["question_id"] for r in kept] == ["D01-001", "D04-001"] and dict(dropped) == {"D01": 1} and not dup


def test_whitespace_differences_do_not_hide_an_overlap():
    a, b = rec("D05", 1, "Which  questions\ntagged x"), rec("D05", 9, "Which questions tagged x")
    assert ss._key(a) == ss._key(b)


def test_prefix_and_limit_keep_questions_unchanged_otherwise():
    recs = [rec("D01", i) for i in range(1, 6)] + [rec("D07", i) for i in range(1, 4)]
    kept, _, _ = ss.subset(recs, set(), prefix="CP1-", limit=2)
    assert [r["question_id"] for r in kept] == ["CP1-D01-001", "CP1-D01-002", "CP1-D07-001", "CP1-D07-002"]
    assert kept[0]["gold"] == [1] and kept[0]["text"] == recs[0]["text"]


def test_cli_writes_the_file_and_reports_counts(tmp_path, capsys):
    q, used, out = tmp_path / "q.jsonl", tmp_path / "used.jsonl", tmp_path / "out.jsonl"
    q.write_text("".join(json.dumps(r) + "\n" for r in [rec("D01", 1), rec("D01", 2)]))
    used.write_text(json.dumps(rec("D01", 1)) + "\n")
    assert ss.main(["--questions", str(q), "--exclude", str(used), "--id-prefix", "X-", "--out", str(out)]) == 0
    written = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["question_id"] for r in written] == ["X-D01-002"]
    assert "1 dropped as already used" in capsys.readouterr().out


def test_duplicate_ids_after_prefixing_are_refused(tmp_path):
    q, out = tmp_path / "q.jsonl", tmp_path / "out.jsonl"
    q.write_text("".join(json.dumps(r) + "\n" for r in [rec("D01", 1), rec("D01", 1, "different")]))
    assert ss.main(["--questions", str(q), "--out", str(out)]) == 2 and not out.exists()


def test_the_same_instance_in_another_paraphrase_is_dropped_by_its_parameters():
    used = [dict(rec("D01", 1, "List the askers of x questions"), params={"tag": "x", "min_reputation": 5})]
    again = dict(rec("D01", 7, "Which users asked a question tagged x"), params={"min_reputation": 5, "tag": "x"})
    other = dict(rec("D01", 8, "Which users asked a question tagged y"), params={"tag": "y", "min_reputation": 5})
    excluded = {ss._key(r) for r in used} | {ss._pkey(r) for r in used}
    kept, dropped, _ = ss.subset([again, other], excluded)
    assert [r["question_id"] for r in kept] == ["D01-008"] and dict(dropped) == {"D01": 1}


def test_records_without_parameters_are_still_compared_by_text(tmp_path):
    kept, dropped, _ = ss.subset([rec("D01", 1)], {ss._key(rec("D01", 1))})
    assert not kept and dict(dropped) == {"D01": 1}
