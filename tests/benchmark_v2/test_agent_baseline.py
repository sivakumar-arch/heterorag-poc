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


def test_transcript_records_what_the_store_returned_against_what_the_model_saw():
    llm = Scripted([T(C("sql", query="q")), T(C("sql", id="t2", query="q2")), T(final_set(1))])
    res = run(llm, Stores(sql=lambda q: ids(*range(100), exhausted=(q == "q"))))
    first, second = res.transcript[0], res.transcript[1]
    assert (first["rows"], first["shown"], first["exhausted"]) == (100, ab.SHOW_ROWS, True)
    assert second["exhausted"] is False
    capped = run(Scripted([T(C("sql", query="x")), T(final_set(1))]), Stores(sql=ids(*range(5), exhausted=False)))
    assert capped.transcript[0]["exhausted"] is False and capped.transcript[0]["cut"] is False


def test_transcript_notes_a_reply_cut_off_at_the_token_limit():
    cut = T(C("sql", query="q"))
    cut.stop_reason = "max_tokens"
    res = run(Scripted([cut, T(final_set(1))]), Stores(sql=ids(1)))
    assert res.transcript[0]["tool"] == "_turn" and res.transcript[0]["stop_reason"] == "max_tokens"


def test_python_combines_results_from_two_stores_in_code():
    out, err = py("a = {r[0] for r in sql('q')}; b = {r[0] for r in cypher('q')}; print(sorted(a & b))")
    assert (out, err) == ("[3, 4]", False)


def test_python_rows_are_always_tuples_even_with_one_column():
    # regression: single-column rows used to be unwrapped to bare values, so the usual row[0] returned
    # the first character of a tag name (live pilot, D04-002 and D07-001) and consumed the turn budget
    s = Stores(sql=RawResult(["tag"], [["alpha"], ["beta"]]))
    out, err = py("r = sql('q'); print(r, r[0][0], [x[0] for x in r])", s)
    assert (out, err) == ("[('alpha',), ('beta',)] alpha ['alpha', 'beta']", False)


def test_python_variables_persist_between_calls_of_one_question_only():
    llm = Scripted([T(C("python", code="ids = [r[0] for r in sql('q')]")),
                    T(C("python", id="t2", code="print(len(ids))")), T(final_set(1))])
    res = run(llm, Stores(sql=ids(1, 2, 3, 4)), arm="python")
    assert res.transcript[1]["result"] == "4" and res.tool_errors == 0 and res.rows_read == 4
    again = run(Scripted([T(C("python", code="print(len(ids))")), T(final_set(1))]), Stores(sql=ids(1)), arm="python")
    assert again.tool_errors == 1 and "NameError" in again.transcript[0]["result"]


def test_python_has_the_builtins_models_reach_for_without_filesystem_or_process_access():
    out, err = py("print(type(sql('q')).__name__, hasattr(sql('q'), 'columns'), isinstance(1, int), chr(65))")
    assert (out, err) == ("Rows True True A", False)
    for name in ("open", "exec", "eval", "compile", "input", "globals", "vars"):
        assert py(name)[1] is True, name          # a NameError: the name is not available


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


def test_system_prompt_gives_the_agent_the_same_case_note_as_the_translator():
    # the pipeline and the agent get the same environment knowledge: text is compared without regard to case
    p = ab.build_system_prompt(DESCRIPTORS, "plain", 10_000, 10)
    assert "without regard to case" in p and "lower() in SQL" in p and "toLower() in Cypher" in p
    assert '"case_insensitive": true' in p


def test_show_rows_and_show_chars_change_what_the_model_is_shown_and_the_prompt_says_so():
    # sensitivity runs: the agent sees 100 rows instead of 50; the prompt, the transcript and the result file agree
    p = ab.build_system_prompt(DESCRIPTORS, "plain", 10_000, 10, show_rows=100)
    assert "first 100 rows" in p and "first 50 rows" not in p
    assert "first 50 rows" in ab.build_system_prompt(DESCRIPTORS, "plain", 10_000, 10)        # the default is unchanged
    llm = Scripted([T(C("sql", query="q")), T(final_set(1))])
    res = ab.run_agent("q", llm, Stores(sql=ids(*range(80))), "sys", ab.tool_definitions("plain", 10_000),
                       arm="plain", cap=10_000, max_turns=5, show_rows=100, show_chars=50_000)
    shown = json.loads(llm.seen[1][1][-1]["content"][0]["content"])
    assert len(shown["rows"]) == 80 and "note" not in shown
    assert res.transcript[0]["shown"] == 80
    default = Scripted([T(C("sql", query="q")), T(final_set(1))])
    ab.run_agent("q", default, Stores(sql=ids(*range(80))), "sys", ab.tool_definitions("plain", 10_000),
                 arm="plain", cap=10_000, max_turns=5)
    assert len(json.loads(default.seen[1][1][-1]["content"][0]["content"])["rows"]) == ab.SHOW_ROWS


def test_show_chars_limits_the_python_tool_output_too():
    out, err = ab.run_python("print('x' * 300)", Stores(), 100, max_chars=100)
    assert not err and out.endswith("(output truncated at 100 characters)")
    out, _ = ab.run_python("print('x' * 300)", Stores(), 100, max_chars=1000)
    assert "truncated" not in out


def test_cli_records_the_display_limits_in_the_result_file(tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text("\n".join(json.dumps(r) for r in records()))
    out = tmp_path / "o.json"
    ab.main(["--questions", str(q), "--out", str(out), "--show-rows", "100", "--show-chars", "20000"],
            llm=Scripted([T(final_set(1)), T(final_set(9))]), stores=Stores(), desc=DESCRIPTORS)
    meta = json.loads(out.read_text())["meta"]
    assert meta["show_rows"] == 100 and meta["show_chars"] == 20000


def test_summary_shows_n_a_for_a_template_whose_answers_are_numbers():
    row = {"template": "D07", "outcome": "correct", "f1": None, "turns": 3, "tool_calls": 2, "tool_errors": 0,
           "usage": {"prompt_tokens": 10, "reply_tokens": 5}, "total_s": 1.0}
    setrow = {**row, "template": "D01", "f1": 1.0}
    text = ab.summarise([row, setrow])
    d07 = [l for l in text.splitlines() if l.startswith("D07")][0]
    d01 = [l for l in text.splitlines() if l.startswith("D01")][0]
    assert "n/a" in d07 and "0.000" not in d07 and "1.000" in d01


# ------------------------------------------------------------------ an answer written as text (Batch 56)

TEXT_ANSWER = '{"kind":"set","number":0,"reason":"r","values":[1352]}'


def test_a_text_answer_is_not_an_answer_by_default():
    llm = Scripted([T(text=TEXT_ANSWER), T(text=TEXT_ANSWER)])
    res = run(llm, Stores())
    assert res.stop == "no_tool" and not res.answered
    assert [x for x in res.transcript if x["tool"] == "_text"][0]["text"] == TEXT_ANSWER


def test_accept_text_final_reads_a_reply_that_is_only_a_valid_answer():
    res = run(Scripted([T(text=TEXT_ANSWER)]), Stores(), accept_text_final=True)
    assert res.answered and res.source == "text" and res.kind == "set" and res.values == [1352] and res.turns == 1
    fenced = "```json\n" + TEXT_ANSWER + "\n```"
    assert run(Scripted([T(text=fenced)]), Stores(), accept_text_final=True).source == "text"
    num = run(Scripted([T(text='{"kind":"number","number":4.5}')]), Stores(), accept_text_final=True)
    assert num.answered and num.number == 4.5


@pytest.mark.parametrize("text", [
    "The answer is " + TEXT_ANSWER,                     # prose around it
    TEXT_ANSWER + TEXT_ANSWER,                          # two objects
    '{"kind":"set","values":"1352"}',                   # values is not a list
    '{"kind":"set","values":[1],"extra":1}',            # unknown key
    '{"kind":"number","number":"4"}',                   # number as text
    '{"kind":"number","number":true}',                  # a boolean is not a number
    '{"kind":"guess","values":[1]}',                    # kind not in the tool's enum
    '[1352]', '', 'not json {',
])
def test_accept_text_final_is_narrow(text):
    assert ab.parse_text_answer(text) is None
    res = run(Scripted([T(text=text), T(text=text)]), Stores(), accept_text_final=True)
    assert res.stop == "no_tool" and not res.answered


def test_a_tool_call_is_still_what_counts_when_both_exist():
    res = run(Scripted([T(final_set(7), text=TEXT_ANSWER)]), Stores(), accept_text_final=True)
    assert res.source == "tool" and res.values == [7]


def test_the_row_records_where_the_answer_came_from_and_the_summary_says_so():
    rec = {"question_id": "Q", "template": "D01", "split": "dev", "text": "q", "answer_type": "set", "gold": [1352]}
    row = ab.run_one(rec, Scripted([T(text=TEXT_ANSWER)]), Stores(), SYSTEM, TOOLS, arm="plain", cap=10_000,
                     max_turns=5, accept_text_final=True)
    assert row["outcome"] == "correct" and row["answer_source"] == "text"
    assert "answers read from reply text" in ab.summarise([row])
    row2 = ab.run_one(rec, Scripted([T(final_set(1352))]), Stores(), SYSTEM, TOOLS, arm="plain", cap=10_000, max_turns=5)
    assert row2["answer_source"] == "tool" and "answers read from reply text" not in ab.summarise([row2])
    row3 = ab.run_one(rec, Scripted([T(text=TEXT_ANSWER), T(text=TEXT_ANSWER)]), Stores(), SYSTEM, TOOLS, arm="plain",
                      cap=10_000, max_turns=5)
    assert row3["outcome"] == "no_answer" and row3["answer_source"] is None


def test_adapter_counters_are_recorded_per_question():
    class Counting(Scripted):
        stats = {"retries": 0}

        def turn(self, *a):
            self.stats["retries"] += 1
            return super().turn(*a)
    rec = {"question_id": "Q", "template": "D01", "split": "dev", "text": "q", "answer_type": "set", "gold": [1]}
    llm = Counting([T(final_set(1))])
    llm.stats["retries"] = 5                                   # counted by an earlier question or an earlier process
    row = ab.run_one(rec, llm, Stores(), SYSTEM, TOOLS, arm="plain", cap=10_000, max_turns=5)
    assert row["adapter"] == {"retries": 1}


def test_audit_flags_a_run_that_ended_with_an_answer_written_as_text():
    from benchmark_v2 import agent_audit
    row = {"stop": "no_tool", "transcript": [{"tool": "_text", "text": TEXT_ANSWER}, {"tool": "_text", "text": TEXT_ANSWER}]}
    assert "answer_as_text" in agent_audit.flags(row)
    row["transcript"] = [{"tool": "_text", "text": "I do not know"}]
    assert "answer_as_text" not in agent_audit.flags(row)


def test_resume_refuses_a_file_written_under_the_other_text_answer_protocol(tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text("\n".join(json.dumps(r) for r in records()))
    out = tmp_path / "o.json"
    out.write_text(json.dumps({"meta": {"accept_text_final": False}, "rows": [], "complete": False}))
    code = ab.main(["--questions", str(q), "--out", str(out), "--arm", "plain", "--resume", "--accept-text-final"],
                   llm=Scripted([]), stores=Stores(), desc=DESCRIPTORS)
    assert code == 78 and json.loads(out.read_text())["rows"] == []
