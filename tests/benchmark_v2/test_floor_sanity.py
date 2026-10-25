"""The sanity check passes a healthy run and names each plumbing fault."""
import json

from benchmark_v2 import floor_sanity as fs

M = "qwen3:30b"


def pipe_row(qid="Q1", **kw):
    r = {"question_id": qid, "plan_valid": True, "plan_abstain": None, "native_calls": [1], "plan_raw": ["{}"],
         "translation_raw": {"s1": ["select 1"]}, "total_s": 100.0,
         "usage": {"planner": {"calls": 2, "reply_tokens": 50}, "translator": {"calls": 1, "reply_tokens": 20}}}
    r.update(kw)
    return r


def agent_row(qid="Q1", **kw):
    r = {"question_id": qid, "turns": 3, "tool_calls": 2, "answer_source": "text", "total_s": 40.0,
         "adapter": {"thinking_turns": 3}, "transcript": [{"tool": "sql", "error": False}]}
    r.update(kw)
    return r


def pipeline(rows=None, **meta):
    m = {"complete": True, "planner_model": f"ollama:{M}", "translator_model": f"ollama:{M}", "models_returned": [M]}
    m.update(meta)
    return {"meta": m, "rows": rows if rows is not None else [pipe_row()]}


def agent(rows=None, **meta):
    m = {"complete": True, "model": f"ollama:{M}", "models_returned": [M]}
    m.update(meta)
    return {"meta": m, "rows": rows if rows is not None else [agent_row()]}


def run(cost=None, python=None, plain=None):
    return fs.check(M, cost or pipeline(), python or agent(), plain or agent())


def status(res, name_part):
    return [s for s, n, _ in res if name_part in n]


def test_a_healthy_run_has_no_fail():
    res = run()
    assert not [x for x in res if x[0] == "FAIL"], fs.render(res)
    assert any("full run of 102" in d for _, _, d in res)


def test_an_infrastructure_row_or_incomplete_file_fails():
    assert "FAIL" in status(run(python=agent([agent_row(infra=True)])), "python: finished cleanly")
    assert "FAIL" in status(run(plain=agent(complete=False)), "plain: finished cleanly")
    assert "FAIL" in status(run(cost=pipeline([pipe_row(reason="crash: KeyError: 'x'")])), "pipeline: finished cleanly")


def test_a_different_model_answering_fails():
    assert "FAIL" in status(run(cost=pipeline(models_returned=["llama3.1:8b"])), "pipeline: model identity")
    assert "FAIL" in status(run(python=agent(model="claude-haiku-4-5")), "python: model identity")


def test_a_pipeline_without_a_planner_reply_or_with_a_think_tag_fails():
    res = run(cost=pipeline([pipe_row(usage={"planner": {"calls": 0, "reply_tokens": 0}})]))
    assert "FAIL" in status(res, "pipeline: planner replied")
    res = run(cost=pipeline([pipe_row(plan_raw=["<think>hmm</think>{}"])]))
    assert "FAIL" in status(res, "no reasoning markup")


def test_an_agent_that_never_calls_a_tool_fails_and_unreadable_arguments_warn():
    res = run(python=agent([agent_row(tool_calls=0)]))
    assert "FAIL" in status(res, "python: tool calls were read")
    res = run(plain=agent([agent_row(adapter={"unreadable_arguments": 2})]))
    assert "WARN" in status(res, "plain: arguments readable")


def test_many_unanswered_or_retried_questions_warn():
    rows = [agent_row(f"Q{i}", answer_source=None, adapter={"retries": 1}) for i in range(4)]
    res = run(python=agent(rows))
    assert "WARN" in status(res, "python: how answers arrived") and "WARN" in status(res, "python: HTTP retries")


def test_main_reports_and_sets_the_exit_status(tmp_path, capsys):
    paths = {}
    for name, d in (("cost", pipeline()), ("python", agent()), ("plain", agent())):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(d))
    args = ["--model", M] + sum((["--" + k, str(v)] for k, v in paths.items()), [])
    assert fs.main(args) == 0 and "0 FAIL" in capsys.readouterr().out
    paths["python"].write_text(json.dumps(agent([agent_row(tool_calls=0)])))
    assert fs.main(args) == 1


def test_empty_model_replies_fail_the_pipeline_check():
    rows = [pipe_row(f"Q{i}", plan_raw=["", ""]) for i in range(4)]
    res = run(cost=pipeline(rows))
    assert "FAIL" in status(res, "pipeline: no empty replies")
    rows = [pipe_row("Q1", plan_raw=["", ""])] + [pipe_row(f"Q{i}") for i in range(2, 9)]
    assert "WARN" in status(run(cost=pipeline(rows)), "pipeline: no empty replies")
    rows = [pipe_row("Q1", translation_raw={"s2": ["", ""]})] + [pipe_row(f"Q{i}") for i in range(2, 9)]
    assert "WARN" in status(run(cost=pipeline(rows)), "pipeline: no empty replies")
    assert status(run(), "pipeline: no empty replies") == ["PASS"]


def test_turns_cut_at_the_token_limit_warn_and_the_setting_is_shown():
    cut = agent_row(stop="no_tool", transcript=[{"tool": "_turn", "stop_reason": "max_tokens"}])
    res = run(python=agent([cut], ollama_think="false"))
    assert "WARN" in status(res, "python: replies cut at the token limit")
    assert any("python': 'false'" in d for s_, n, d in res if n == "settings")
