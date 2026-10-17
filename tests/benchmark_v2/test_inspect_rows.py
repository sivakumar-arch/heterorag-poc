"""Reading selected rows of a run_slice output."""
import json

from benchmark_v2 import inspect_rows as ir


PLAN = {"steps": [{"id": "a", "op": "retrieve", "store": "graph"},
                  {"id": "b", "op": "retrieve", "store": "sql", "inputs": ["a"], "bind": {"input": "a", "entity": "user"}},
                  {"id": "c", "op": "intersect", "inputs": ["a", "b"]}]}


def row(qid="D01-002", outcome="partial_valid", **kw):
    return dict({"question_id": qid, "outcome": outcome, "guarantee": "subset", "bind_policy": "cost", "text": "q?",
                 "plan": PLAN, "executed_plan": PLAN, "estimates": {"a": {"n": 3, "basis": "count"}},
                 "bind_decisions": [{"action": "bind", "level": "feasibility", "step": "b", "source": "a",
                                     "reason": "independent execution would exceed the cap"}],
                 "queries": {"a": "MATCH (u)\n RETURN u.id"}, "step_errors": [], "reason": ""}, **kw)


def test_steps_line_shows_stores_inputs_and_binds():
    assert ir.steps_line(PLAN) == "a:retrieve/graph  b:retrieve/sql(a)+bind  c:intersect(a,b)"
    assert ir.steps_line(None) == "none"


def test_describe_lists_decisions_estimates_and_queries():
    text = ir.describe(row())
    assert "decision : bind        feasibility   b <- a" in text and "estimates: a=3 (count)" in text
    assert "query a  : MATCH (u) RETURN u.id" in text and "executed" not in text


def test_a_rejected_plan_shows_the_planner_replies():
    r = row("D07-001", "abstain_unexpected", plan=None, executed_plan=None, plan_abstain="invalid_plan",
            plan_errors=["plan depth 4 exceeds 3"], plan_repairs=1, plan_raw=['{"steps": [1]}', '{"steps": [2]}'],
            estimates={}, bind_decisions=[])
    text = ir.describe(r)
    assert "invalid_plan" in text and "plan depth 4 exceeds 3" in text and '{"steps": [1]}' in text and '{"steps": [2]}' in text


def test_cli_selects_by_id_or_by_outcome(tmp_path, capsys):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"rows": [row("D01-001", "correct"), row("D01-002")]}))
    assert ir.main([str(p), "--ids", "D01-001"]) == 0 and "D01-001  correct" in capsys.readouterr().out
    assert ir.main([str(p), "--not-correct"]) == 0
    out = capsys.readouterr().out
    assert "D01-002" in out and "D01-001" not in out
    assert ir.main([str(p), "--ids", "nope"]) == 2


def test_a_failed_restricted_translation_is_shown():
    text = ir.describe(row(bind_fallbacks=["cannot restrict b: Elasticsearch rejected the query: boom"]))
    assert "fallback : cannot restrict b: Elasticsearch rejected the query: boom" in text
