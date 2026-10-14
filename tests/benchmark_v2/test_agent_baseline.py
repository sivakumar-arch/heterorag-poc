"""The free-form tool-use agent: loop, tools, the python sandbox, scoring and the CLI, with a
scripted model and in-memory stores (no API calls, no live databases)."""
import json
import time
from types import SimpleNamespace as NS

import pytest

from benchmark_v2 import agent_baseline as ab
from benchmark_v2.agent_baseline import AgentResult, Call, RawResult, Turn
from tests.federation.fake_descriptors import DESCRIPTORS


class Stores:
    """In-memory stores: each tool returns a fixed or computed RawResult."""

    def __init__(self, sql=None, cypher=None, es=None):
        self._f = {"sql": sql, "cypher": cypher, "es": es}
        self.log = []

    def _run(self, name, arg, limit):
        self.log.append((name, arg, limit))
        f = self._f[name]
        if f is None:
            raise RuntimeError(f"no {name} store")
        out = f(arg) if callable(f) else f
        if isinstance(out, Exception):
            raise out
        return out

    def sql(self, q, limit): return self._run("sql", q, limit)
    def cypher(self, q, limit): return self._run("cypher", q, limit)
    def es(self, b, limit): return self._run("es", b, limit)


def ids(*vals, col="id", exhausted=True):
    return RawResult([col], [[v] for v in vals], exhausted)


class Scripted:
    """A model that plays back prepared turns and records what it was sent."""
    model = "scripted-model"

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def turn(self, system, messages, tools, max_tokens):
        self.seen.append((system, [dict(m) for m in messages], tools))
        return self.turns.pop(0)


def T(*calls, text="", pt=100, rt=10):
    content = ([{"type": "text", "text": text}] if text else []) + [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
    return Turn(content, list(calls), text, pt, rt, 0.01)


def C(name, id="t1", **inp): return Call(id, name, inp)


def final_set(*vals, id="f"): return C("final_answer", id, kind="set", values=list(vals))


SYSTEM, TOOLS = "system", ab.tool_definitions("plain", 10_000)


def run(llm, stores, arm="plain", **kw):
    return ab.run_agent("question?", llm, stores, SYSTEM, ab.tool_definitions(arm, 10_000), arm=arm, **kw)


# ------------------------------------------------------------------ the loop

def test_agent_queries_then_answers_and_usage_is_summed():
    llm = Scripted([T(C("sql", query="SELECT id FROM users")), T(final_set(1, 2, 3))])
    res = run(llm, Stores(sql=ids(1, 2, 3)))
    assert res.answered and res.kind == "set" and res.values == [1, 2, 3] and res.stop == "answered"
    assert (res.turns, res.tool_calls, res.tool_errors, res.rows_read) == (2, 1, 0, 3)
    assert (res.prompt_tokens, res.reply_tokens) == (200, 20)
    # the tool result reached the model as a tool_result block
    second_messages = llm.seen[1][1]
    block = second_messages[-1]["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "t1" and not block["is_error"]
    assert json.loads(block["content"])["rows_returned"] == 3


def test_a_store_error_is_returned_to_the_model_and_counted():
    llm = Scripted([T(C("sql", query="SELEC")), T(C("sql", "t2", query="SELECT 1")), T(final_set(7))])
    stores = Stores(sql=lambda q: ValueError("syntax error at or near SELEC") if q == "SELEC" else ids(7))
    res = run(llm, stores)
    assert res.answered and res.tool_errors == 1 and res.tool_calls == 2
    err = llm.seen[1][1][-1]["content"][0]
    assert err["is_error"] and "syntax error" in err["content"]


def test_the_turn_budget_ends_the_run_without_an_answer():
    llm = Scripted([T(C("sql", f"t{i}", query="SELECT 1")) for i in range(3)])
    res = run(llm, Stores(sql=ids(1)), max_turns=3)
    assert res.stop == "max_turns" and not res.answered and res.turns == 3


def test_a_reply_without_tools_is_nudged_once():
    llm = Scripted([T(text="I think it is 5"), T(final_set(5))])
    assert run(llm, Stores()).answered
    llm = Scripted([T(text="a"), T(text="b")])
    res = run(llm, Stores())
    assert res.stop == "no_tool" and not res.answered


def test_a_failing_model_call_is_recorded_not_raised():
    class Boom:
        def turn(self, *a): raise ConnectionError("api down")
    res = run(Boom(), Stores())
    assert res.stop == "error" and "api down" in res.reason


def test_final_answer_ends_the_run_even_with_other_calls_in_the_turn():
    llm = Scripted([T(final_set(1), C("sql", "t9", query="SELECT 1"))])
    stores = Stores(sql=ids(1))
    res = run(llm, stores)
    assert res.answered and stores.log == []


def test_results_shown_to_the_model_are_truncated_but_the_count_is_exact():
    big = ids(*range(500), exhausted=False)
    d = json.loads(ab.render_rows(big, show=5, max_chars=10_000))
    assert len(d["rows"]) == 5 and d["rows_returned"] == 500 and d["all_rows_returned"] is False and "note" in d
    assert ab.render_rows(big, show=500, max_chars=100).endswith("(truncated)")


def test_unknown_tool_and_python_in_the_plain_arm_are_errors():
    llm = Scripted([T(C("python", code="print(1)")), T(final_set(1))])
    res = run(llm, Stores(), arm="plain")
    assert res.tool_errors == 1
    assert "unknown tool" in llm.seen[1][1][-1]["content"][0]["content"]


# ------------------------------------------------------------------ the store limit

def test_every_call_is_made_with_the_cap():
    stores = Stores(sql=ids(1), cypher=ids(1), es=RawResult(["post_id"], [[1]]))
    llm = Scripted([T(C("sql", "a", query="q"), C("cypher", "b", query="q"), C("es", "c", body={"query": {}})),
                    T(final_set(1))])
    run(llm, stores, cap=123)
    assert {x[2] for x in stores.log} == {123}


# ------------------------------------------------------------------ the python tool

def py(code, stores=None, **kw):
    return ab.run_python(code, stores or Stores(sql=ids(1, 2, 3, 4), cypher=ids(3, 4, 5)), 100, **kw)


def test_python_combines_results_from_two_stores_in_code():
    out, err = py("a = set(sql('q')); b = set(cypher('q')); print(sorted(a & b))")
    assert (out, err) == ("[3, 4]", False)


def test_python_helpers_report_truncation_and_columns():
    s = Stores(sql=RawResult(["id"], [[1], [2]], exhausted=False))
    out, err = py("r = sql('q'); print(r.truncated, r.columns, len(r))", s)
    assert out == "True ['id'] 2" and not err


def test_python_multi_column_rows_are_tuples():
    s = Stores(sql=RawResult(["id", "n"], [[1, "a"], [2, "b"]]))
    out, _ = py("print(sql('q'))", s)
    assert out == "[(1, 'a'), (2, 'b')]"


def test_python_errors_are_returned_not_raised():
    out, err = py("print('before'); 1/0")
    assert err and "before" in out and "ZeroDivisionError" in out


def test_python_cannot_import_system_modules_or_open_files():
    for code in ("import os", "import subprocess", "open('/etc/passwd')", "__import__('os')", "import socket"):
        out, err = py(code)
        assert err, code
    out, err = py("import collections, itertools, math, json, re; print(math.sqrt(16))")
    assert (out, err) == ("4.0", False)


def test_python_timeout_interrupts_a_runaway_loop():
    t0 = time.perf_counter()
    out, err = py("while True: pass", seconds=1)
    assert err and "longer than" in out and time.perf_counter() - t0 < 5


def test_python_output_is_truncated_and_empty_output_is_named():
    out, _ = py("print('x' * 20000)")
    assert len(out) < ab.SHOW_CHARS + 100 and "truncated" in out
    assert py("x = 1")[0] == "(no output)"


def test_python_rows_are_counted_through_the_loop():
    llm = Scripted([T(C("python", code="print(len(sql('q')))")), T(final_set(1))])
    res = run(llm, Stores(sql=ids(1, 2, 3, 4)), arm="python")
    assert res.rows_read == 4 and res.tool_errors == 0


# ------------------------------------------------------------------ scoring

def rec(at, gold, **kw): return {"answer_type": at, "gold": gold, **kw}


def ans(kind, values=(), number=None, stop="answered"):
    return AgentResult(kind=kind, values=list(values), number=number, stop=stop)


def test_scoring_sets_compares_as_text_and_reports_f1():
    s = ab.score(rec("ids", [1, 2, 3, 4]), ans("set", ["1", 2, 3, 4]))
    assert s["outcome"] == "correct" and s["f1"] == 1.0
    s = ab.score(rec("ids", [1, 2, 3, 4]), ans("set", [1, 2, 9]))
    assert s["outcome"] == "wrong" and s["precision"] == pytest.approx(2 / 3, abs=1e-3) and s["recall"] == 0.5
    assert ab.score(rec("ids", [1, 2]), ans("set", [1, 1, 2]))["outcome"] == "wrong"       # duplicates
    assert ab.score(rec("ids", ["a", "b"]), ans("set", ["b", "a"]))["outcome"] == "correct"
    assert ab.score(rec("ids", [1]), ans("set", []))["f1"] == 0.0


def test_scoring_ranked_needs_the_order():
    assert ab.score(rec("ranked", [3, 1, 2]), ans("ranked", [3, 1, 2]))["outcome"] == "correct"
    assert ab.score(rec("ranked", [3, 1, 2]), ans("ranked", [1, 3, 2]))["outcome"] == "wrong"
    assert ab.score(rec("ranked", [3, 1, 2]), ans("set", [3, 1, 2]))["outcome"] == "correct"   # kind label is not scored


def test_scoring_scalars_abstentions_and_missing_answers():
    assert ab.score(rec("scalar", 1438), ans("number", number=1438.0))["outcome"] == "correct"
    assert ab.score(rec("scalar", 1438), ans("number", number=1437))["outcome"] == "wrong"
    assert ab.score(rec("scalar", 5), ans("set", [5]))["outcome"] == "wrong"                    # wrong answer kind
    assert ab.score(rec("scalar", 5), ans("number"))["outcome"] == "wrong"
    assert ab.score(rec("ids", [1]), ans("abstain"))["outcome"] == "abstain_unexpected"
    assert ab.score(rec("ids", [1]), AgentResult(stop="max_turns"))["outcome"] == "no_answer"


# ------------------------------------------------------------------ prompt and tools

def test_system_prompt_carries_the_same_facts_as_the_federation_prompts():
    p = ab.build_system_prompt(DESCRIPTORS, "plain", 10_000, 10)
    from heterorag.federation.contract import ID_CONTRACT
    from heterorag.federation.plan import Store
    assert "posts.id" in p and "Question.id" in p and "post_id" in p               # identifier contract, all stores
    assert "TABLE users" in p and "RELATIONSHIPS" in p and "doc_type" in p          # schemas
    assert "10000 rows" in p and "at most 10 turns" in p and "final_answer" in p
    assert "python tool" not in p
    q = ab.build_system_prompt(DESCRIPTORS, "python", 10_000, 10)
    assert "python tool" in q


def test_tool_sets_per_arm():
    assert [t["name"] for t in ab.tool_definitions("plain", 100)] == ["sql", "cypher", "es", "final_answer"]
    assert [t["name"] for t in ab.tool_definitions("python", 100)] == ["sql", "cypher", "es", "python", "final_answer"]
    for t in ab.tool_definitions("python", 100):
        assert t["input_schema"]["type"] == "object" and t["input_schema"]["required"]


# ------------------------------------------------------------------ the Anthropic adapter

class FakeMessages:
    def __init__(self, resp): self.resp, self.kwargs = resp, None
    def create(self, **kw):
        self.kwargs = kw
        return self.resp


def test_anthropic_adapter_converts_blocks_and_reads_usage():
    resp = NS(content=[NS(type="text", text="thinking"), NS(type="tool_use", id="tu1", name="sql", input={"query": "q"})],
              usage=NS(input_tokens=321, output_tokens=45), stop_reason="tool_use", model="claude-haiku-4-5-20251001")
    client = NS(messages=FakeMessages(resp))
    agent = ab.AnthropicAgent(model="claude-haiku-4-5-20251001", client=client)
    t = agent.turn("sys", [{"role": "user", "content": "q"}], TOOLS, 500)
    assert t.calls == [Call("tu1", "sql", {"query": "q"})] and t.text == "thinking"
    assert (t.prompt_tokens, t.reply_tokens, t.stop_reason) == (321, 45, "tool_use")
    assert t.content[1] == {"type": "tool_use", "id": "tu1", "name": "sql", "input": {"query": "q"}}
    assert client.messages.kwargs["system"] == "sys" and "temperature" not in client.messages.kwargs
    assert agent.models_seen == {"claude-haiku-4-5-20251001"}


# ------------------------------------------------------------------ one question and the CLI

def records():
    return [
        {"question_id": "D05-001", "template": "D05", "class": "D", "split": "dev", "stratum": "x",
         "text": "which?", "answer_type": "ids", "gold": [1, 2]},
        {"question_id": "S03-001", "template": "S03", "class": "S", "split": "test", "stratum": "y",
         "text": "top?", "answer_type": "ids", "gold": [9]},
    ]


def test_cli_runs_and_hides_transcripts_and_gold_for_the_test_split(tmp_path, capsys):
    q = tmp_path / "q.jsonl"
    q.write_text("\n".join(json.dumps(r) for r in records()))
    llm = Scripted([T(C("sql", query="q")), T(final_set(1, 2, 3)),            # dev question: wrong (extra 3)
                    T(final_set(9))])                                         # test question: correct
    out = tmp_path / "o.json"
    code = ab.main(["--questions", str(q), "--out", str(out), "--arm", "plain"], llm=llm,
                   stores=Stores(sql=ids(1, 2, 3)), desc=DESCRIPTORS)
    assert code == 0
    data = json.loads(out.read_text())
    dev, test = data["rows"]
    assert dev["outcome"] == "wrong" and dev["got"] == [1, 2, 3] and dev["transcript"]
    assert test["outcome"] == "correct" and "transcript" not in test and "got" not in test
    assert data["meta"]["arm"] == "plain" and data["meta"]["cap"] == 10_000
    printed = capsys.readouterr().out
    assert "template" in printed and "mean_f1" in printed and "all" in printed


def test_cli_filters_and_a_crash_is_recorded(tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text("\n".join(json.dumps(r) for r in records()))

    class Crash:
        model = "m"
        def turn(self, *a): raise MemoryError("boom")

    out = tmp_path / "o.json"
    ab.main(["--questions", str(q), "--templates", "D05", "--out", str(out)], llm=Crash(), stores=Stores(),
            desc=DESCRIPTORS)
    row = json.loads(out.read_text())["rows"][0]
    assert row["outcome"] == "no_answer" and row["template"] == "D05"
    assert ab.main(["--questions", str(q), "--templates", "NOPE", "--out", str(out)], llm=Crash(),
                   stores=Stores(), desc=DESCRIPTORS) == 2


def test_the_agent_module_makes_no_live_connection_on_import():
    import importlib
    importlib.reload(ab)                       # importing must not need drivers or credentials
