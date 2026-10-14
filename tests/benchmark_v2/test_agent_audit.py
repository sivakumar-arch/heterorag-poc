"""The audit of agent runs separates harness conditions from method failures."""
import json

from benchmark_v2 import agent_audit as aa


def row(qid="D05-0001", outcome="wrong", stop="answered", transcript=None, **kw):
    base = {"question_id": qid, "template": qid.split("-")[0], "outcome": outcome, "stop": stop, "turns": 3,
            "tool_calls": 2, "tool_errors": 0, "f1": None, "precision": None, "recall": None,
            "answer_kind": "set", "text": "q?", "reason": "", "transcript": transcript or []}
    base.update(kw)
    return base


def sql_call(**kw): return {"tool": "sql", "input": "q", "result": "r", "error": False, "rows": 10, "exhausted": True, **kw}


def test_flags_report_each_harness_condition():
    assert aa.flags(row(stop="max_turns")) == ["budget"]
    assert aa.flags(row(stop="no_tool")) == ["no_tool"]
    assert aa.flags(row(transcript=[{"tool": "_turn", "stop_reason": "max_tokens"}])) == ["max_tokens"]
    assert aa.flags(row(transcript=[sql_call(exhausted=False)])) == ["cap_hit"]
    assert aa.flags(row(transcript=[sql_call(rows=500)])) == ["display_clipped"]
    assert aa.flags(row(transcript=[sql_call(cut=True)])) == ["output_cut"]
    assert aa.flags(row(transcript=[sql_call()])) == []


def test_tool_errors_and_their_causes_are_flagged():
    e = sql_call(error=True, result="RequestError: Result window is too large, from + size must be <= 10000")
    t = sql_call(tool="python", error=True, result="_Timeout: the code ran longer than 30 s")
    assert set(aa.flags(row(transcript=[e]))) == {"tool_errors=1", "es_window"}
    assert set(aa.flags(row(transcript=[t]))) == {"tool_errors=1", "python_timeout"}


def test_diagnosis_reads_precision_and_recall():
    d = lambda **kw: aa.diagnosis(row(**kw))
    assert d(outcome="correct") == ""
    assert d(outcome="no_answer") == "no answer given"
    assert "abstained" in d(outcome="abstain_unexpected")
    assert d(precision=1.0, recall=0.5, f1=0.67).startswith("incomplete")
    assert d(precision=0.5, recall=1.0, f1=0.67).startswith("over-complete")
    assert d(precision=0.0, recall=0.0, f1=0.0).startswith("disjoint")
    assert d(precision=0.5, recall=0.5, f1=0.5).startswith("partly")
    assert d(answer_kind="number") == "wrong number"


def test_render_lists_failures_and_counts_flags(tmp_path):
    rows = [row("D01-0001", "correct"), row("D05-0001", "no_answer", "max_turns", [sql_call()]),
            row("D07-0001", "wrong", transcript=[sql_call()], answer_kind="number")]
    text = aa.render({"arm": "plain", "n": 3}, rows)
    assert "D01-0001" not in text and "D05-0001" in text and "budget" in text
    assert "(none: finished, wrong, no harness flag)" in text


def test_cli_prints_a_summary_and_a_single_transcript(tmp_path, capsys):
    p = tmp_path / "pilot.json"
    r = row("D05-0007", "wrong", transcript=[sql_call(chars=900)], got=[1, 2])
    p.write_text(json.dumps({"meta": {"arm": "python"}, "rows": [r]}))
    assert aa.main([str(p)]) == 0
    assert "D05-0007" in capsys.readouterr().out
    assert aa.main([str(p), "--show", "D05-0007"]) == 0
    out = capsys.readouterr().out
    assert "1. sql" in out and "in : q" in out
    assert aa.main([str(p), "--show", "nope"]) == 0
    assert "not in this file" in capsys.readouterr().out
