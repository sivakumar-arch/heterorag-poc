"""Local models through Ollama: routing, provider options, and the agent adapter (no server, no model)."""
import json
import sys
from types import SimpleNamespace as NS

import pytest

from benchmark_v2 import agent_baseline as ab
from benchmark_v2.agent_baseline import OllamaAgent, make_agent
from heterorag.federation import llm_config as lc
from heterorag.llm_provider import OllamaProvider
from tests.benchmark_v2.test_agent_baseline import Stores, ids
from tests.federation.fake_descriptors import DESCRIPTORS


class FakeClient:
    """Plays back prepared chat responses and records the calls."""
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def chat(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def msg(content="", calls=(), **extra):
    return {"message": {"role": "assistant", "content": content,
                        "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]},
            "prompt_eval_count": 120, "eval_count": 15, "model": "llama3.1:8b", **extra}


# ------------------------------------------------------------------ routing and settings

def test_model_names_route_to_the_right_backend():
    assert lc.split_model("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")
    assert lc.split_model("claude-haiku-4-5-20251001") == ("anthropic", "claude-haiku-4-5-20251001")
    assert lc.split_model("ollama:qwen3:8b")[1] == "qwen3:8b"


def test_ollama_settings_defaults_and_overrides():
    d = lc.ollama_settings({})
    assert d == {"host": "http://localhost:11434", "num_ctx": 32768, "timeout": 600.0}
    o = lc.ollama_settings({"HETERORAG_OLLAMA_HOST": "http://h:1", "HETERORAG_OLLAMA_NUM_CTX": "65536", "HETERORAG_OLLAMA_TIMEOUT": "30"})
    assert o == {"host": "http://h:1", "num_ctx": 65536, "timeout": 30.0}


def test_make_provider_builds_a_metered_local_provider(monkeypatch):
    made = {}

    class Client:
        def __init__(self, **kw): made.update(kw)
        def chat(self, **kw):
            made["chat"] = kw
            return msg("<think>hmm</think> {\"ok\": true}")
    monkeypatch.setitem(sys.modules, "ollama", NS(Client=Client))
    p = lc.make_provider("ollama:llama3.1:8b", "planner")
    resp = p.complete("hello", max_tokens=50, temperature=0.0)
    assert resp.text == '{"ok": true}' and resp.model == "llama3.1:8b" and p.models_seen == ["llama3.1:8b"]
    assert made["host"] == "http://localhost:11434" and made["timeout"] == 600.0
    opts = made["chat"]["options"]
    assert opts == {"num_predict": 50, "num_ctx": 32768}                     # no temperature: the model default
    assert p.total.calls == 1 and p.total.prompt_tokens == 120


# ------------------------------------------------------------------ the provider

def test_the_legacy_provider_still_sends_the_temperature_and_no_context_size():
    c = FakeClient(msg("x"))
    OllamaProvider("llama3", client=c).complete("p", max_tokens=9, temperature=0.3)
    assert c.calls[0]["options"] == {"num_predict": 9, "temperature": 0.3}


def test_a_prompt_that_would_fill_the_context_window_is_refused_not_truncated():
    p = OllamaProvider("m", num_ctx=1000, client=FakeClient(msg("x")))
    with pytest.raises(RuntimeError, match="context window"):
        p.complete("a" * 3000)
    assert p._client.calls == []                                           # nothing was sent
    assert p.complete("a" * 100).text == "x"


def test_think_blocks_are_removed_only_when_asked():
    raw = "<think>\nlong\nreasoning</think>\nanswer"
    assert OllamaProvider("m", strip_think=True, client=FakeClient(msg(raw))).complete("p").text == "answer"
    assert "<think>" in OllamaProvider("m", client=FakeClient(msg(raw))).complete("p").text


# ------------------------------------------------------------------ the agent adapter

TOOLS = [{"name": "sql", "description": "d", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}]


def agent(*responses, **kw):
    c = FakeClient(*responses)
    return OllamaAgent("ollama:llama3.1:8b", client=c, num_ctx=kw.pop("num_ctx", 32768), **kw), c


def test_messages_and_tools_are_converted_to_the_ollama_format():
    a, c = agent(msg("done"))
    history = [{"role": "user", "content": "question"},
               {"role": "assistant", "content": [{"type": "text", "text": "let me look"},
                                                 {"type": "tool_use", "id": "t9", "name": "sql", "input": {"query": "select 1"}}]},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t9", "content": "{\"rows\": []}", "is_error": False}]}]
    a.turn("SYSTEM", history, TOOLS, 777)
    sent = c.calls[0]
    assert sent["model"] == "llama3.1:8b" and sent["options"] == {"num_predict": 777, "num_ctx": 32768}
    assert sent["messages"] == [
        {"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "question"},
        {"role": "assistant", "content": "let me look", "tool_calls": [{"function": {"name": "sql", "arguments": {"query": "select 1"}}}]},
        {"role": "tool", "tool_name": "sql", "content": "{\"rows\": []}"}]
    assert sent["tools"] == [{"type": "function", "function": {"name": "sql", "description": "d", "parameters": TOOLS[0]["input_schema"]}}]


def test_a_reply_with_tool_calls_becomes_blocks_with_stable_unique_ids():
    a, _ = agent(msg("thinking", [("sql", {"query": "q1"}), ("cypher", {"query": "q2"})]))
    t = a.turn("s", [{"role": "user", "content": "q"}], TOOLS, 100)
    assert [c.name for c in t.calls] == ["sql", "cypher"] and len({c.id for c in t.calls}) == 2
    assert t.content[0] == {"type": "text", "text": "thinking"} and t.content[1]["id"] == t.calls[0].id
    assert (t.stop_reason, t.prompt_tokens, t.reply_tokens, t.model) == ("tool_use", 120, 15, "llama3.1:8b")
    assert a.models_seen == {"llama3.1:8b"}


def test_attribute_style_responses_and_stop_reasons():
    obj = NS(message=NS(content="no tool here", tool_calls=None), prompt_eval_count=None, eval_count=None, done_reason="length", model="m1")
    a, _ = agent(obj)
    t = a.turn("s", [{"role": "user", "content": "q"}], TOOLS, 10)
    assert t.calls == [] and t.stop_reason == "max_tokens" and t.prompt_tokens == 0 and a.models_seen == {"m1"}
    a2, _ = agent(msg("fine"))
    assert a2.turn("s", [{"role": "user", "content": "q"}], TOOLS, 10).stop_reason == "end_turn"


def test_arguments_given_as_text_are_read_and_counted():
    a, _ = agent(msg("", [("final_answer", json.dumps({"kind": "set", "values": "[1, 2, 3]"})), ("sql", "not json")]))
    t = a.turn("s", [{"role": "user", "content": "q"}], TOOLS, 10)
    assert t.calls[0].input == {"kind": "set", "values": [1, 2, 3]} and t.calls[1].input == {}
    assert a.stats == {"arguments_as_text": 2, "values_as_text": 1, "unreadable_arguments": 1}


def test_a_conversation_that_would_overflow_the_window_is_refused():
    a, c = agent(msg("x"), num_ctx=500)
    with pytest.raises(RuntimeError, match="context window"):
        a.turn("s" * 2000, [{"role": "user", "content": "q"}], TOOLS, 10)
    assert c.calls == []


def test_transient_errors_are_retried_and_other_errors_raised(monkeypatch):
    monkeypatch.setattr(ab.time, "sleep", lambda s: None)
    class Overloaded(Exception):
        status_code = 503
    a, c = agent(Overloaded("busy"), msg("ok"), retries=2)
    assert a.turn("s", [{"role": "user", "content": "q"}], TOOLS, 10).text == "ok" and len(c.calls) == 2
    a, _ = agent(ValueError("model does not support tools"), msg("never"))
    with pytest.raises(ValueError):
        a.turn("s", [{"role": "user", "content": "q"}], TOOLS, 10)


def test_the_loop_runs_end_to_end_through_the_adapter():
    a, c = agent(msg("", [("sql", {"query": "select id from t"})]),
                 msg("", [("final_answer", {"kind": "set", "values": [1, 2]})]))
    stores = Stores(sql=ids(1, 2))
    system = ab.build_system_prompt(DESCRIPTORS, "plain", 10000, 10)
    res = ab.run_agent("which?", a, stores, system, ab.tool_definitions("plain", 10000), arm="plain", cap=10000, max_turns=5)
    assert res.stop == "answered" and res.values == [1, 2] and res.turns == 2 and res.tool_calls == 1
    second = c.calls[1]["messages"]
    assert second[-1]["role"] == "tool" and second[-1]["tool_name"] == "sql" and "rows" in second[-1]["content"]
    assert second[0]["content"] == system                                   # the same prompt as for every other model


def test_make_agent_picks_the_backend_from_the_model_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "ollama", NS(Client=lambda **kw: FakeClient()))
    assert type(make_agent("ollama:qwen3:8b")).__name__ == "OllamaAgent" and make_agent("ollama:qwen3:8b").model == "qwen3:8b"
    monkeypatch.setenv("HETERORAG_AGENT_MODEL", "ollama:llama3.1:8b")
    assert make_agent().model == "llama3.1:8b"
    monkeypatch.setitem(sys.modules, "anthropic", NS(Anthropic=lambda **kw: object()))
    monkeypatch.setenv("HETERORAG_AGENT_MODEL", "claude-haiku-4-5-20251001")
    assert isinstance(make_agent(), ab.AnthropicAgent) and make_agent().model == "claude-haiku-4-5-20251001"
