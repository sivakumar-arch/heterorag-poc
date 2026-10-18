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


def test_describe_shows_rejected_translations_and_the_replies():
    text = ir.describe(row(translation_errors={"s1": ["the first returned item must be the answer identifier (.id), found 'a'"]},
                           translation_raw={"s1": ["MATCH (q)<-[:ANSWERS]-(a)\n RETURN DISTINCT a", "MATCH (a)\nRETURN a"]}))
    assert "translation s1 rejected: the first returned item must be the answer identifier" in text
    assert "reply 1: MATCH (q)<-[:ANSWERS]-(a) RETURN DISTINCT a" in text and "reply 2: MATCH (a) RETURN a" in text


def test_gold_is_shown_next_to_what_was_returned_for_dev_questions_only():
    ids = row(answer_type="ids", got=[1, 2, 9, 10])
    rec = {"question_id": "D01-002", "split": "dev", "answer_type": "ids", "gold": [1, 2, 3, 4, 5]}
    text = ir.describe(ids, rec=rec)
    assert "gold     : 5 ids, first 8 [1, 2, 3, 4, 5]" in text
    assert "returned : first 4 ids [1, 2, 9, 10]; 2 of them are in the gold set, 2 are not" in text
    scalar = ir.describe(row(got=41), rec={"split": "dev", "answer_type": "scalar", "gold": 37})
    assert "gold     : 37   returned: 41" in scalar
    assert "gold" not in ir.describe(ids, rec=dict(rec, split="test"))          # the test split's gold is never printed
    assert "gold" not in ir.describe(ids)


def test_cli_prints_gold_with_a_question_file(tmp_path, capsys):
    run = tmp_path / "r.json"
    run.write_text(json.dumps({"meta": {}, "rows": [row(qid="Q1", answer_type="scalar", got=5)]}))
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"question_id": "Q1", "split": "dev", "answer_type": "scalar", "gold": 7}))
    assert ir.main([str(run), "--ids", "Q1", "--questions", str(q)]) == 0
    assert "gold     : 7   returned: 5" in capsys.readouterr().out
