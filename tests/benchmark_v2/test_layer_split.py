import json

from benchmark_v2 import layer_split as ls


def row(o, eq, plan=None, recs=None):
    return {"question_id": "q", "outcome": o, "plan_equal": eq, "plan": plan, "step_records": recs}


PLAN = {"steps": [{"id": "a", "op": "retrieve"}, {"id": "b", "op": "retrieve", "bind": {"input": "a", "entity": "user"}}]}


def test_layer_counts():
    rows = [row("wrong", True), row("wrong", False), row("partial_violated", False), row("correct", False),
            row("correct", True), row("abstain_unexpected", False), {"question_id": "z", "outcome": "wrong"}]
    c = ls.layer_counts(rows)
    assert c["silent"] == 3 and c["silent_plan_equal"] == 1 and c["silent_plan_differs"] == 2
    assert c["correct"] == 2 and c["correct_plan_differs"] == 1 and c["no_plan_flag"] == 1


def test_bound_inputs_counts_inexact_sources_and_skips_rows_without_records():
    exact = [{"step": "a", "guarantee": "exact"}, {"step": "b", "guarantee": "exact"}]
    capped = [{"step": "a", "guarantee": "subset"}, {"step": "b", "guarantee": "subset"}]
    rows = [row("correct", True, PLAN, exact), row("wrong", True, PLAN, capped), row("wrong", True, None, None)]
    c = ls.bound_inputs(rows)
    assert c == {"rows_with_records": 2, "rows_without": 1, "bound_steps": 2, "inexact_input": 1, "silent_with_inexact": 1}


def test_main_prints_each_file_and_a_total(tmp_path, capsys):
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"rows": [row("wrong", True), row("correct", False)]}))
    g = tmp_path / "b.json"
    g.write_text(json.dumps({"rows": [row("wrong", False)]}))
    assert ls.main([str(f), str(g)]) == 0
    out = capsys.readouterr().out
    assert "a.json" in out and "structure equal" in out and "all files" in out and "no step records" in out
    assert ls.main([str(tmp_path / "missing.json")]) == 2
