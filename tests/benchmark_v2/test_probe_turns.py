"""The turn probe prints what the model returned and passes the reply on unchanged."""
from benchmark_v2 import probe_turns as pt
from benchmark_v2.agent_baseline import Call, Turn


class Fake:
    model, models_seen = "m", set()

    def __init__(self, *turns):
        self.turns = list(turns)

    def turn(self, system, messages, tools, max_tokens):
        t = self.turns.pop(0)
        if isinstance(t, Exception):
            raise t
        return t


def test_each_reply_is_printed_and_returned_unchanged():
    t1 = Turn([{"type": "thinking", "thinking": "plan the query"}, {"type": "tool_use", "id": "c1", "name": "sql", "input": {"query": "select 1"}}],
              [Call("c1", "sql", {"query": "select 1"})], "", 10, 5, 0.1, "m", "tool_use")
    t2 = Turn([{"type": "text", "text": "the answer is 4"}], [], "the answer is 4", 12, 6, 0.1, "m", "end_turn")
    out = []
    llm = pt.trace(Fake(t1, t2), write=out.append)
    assert llm.turn("s", [], [], 10) is t1 and llm.turn("s", [], [], 10) is t2
    text = "\n".join(out)
    assert "turn 1: stop=tool_use" in text and "reasoning: plan the query" in text and "call sql:" in text
    assert "turn 2: stop=end_turn" in text and "text: the answer is 4" in text and "(no tool call in this reply)" in text


def test_a_failed_call_is_reported_and_raised_again():
    out = []
    llm = pt.trace(Fake(ValueError("server said 500")), write=out.append)
    try:
        llm.turn("s", [], [], 10)
        raise AssertionError("expected the error to propagate")
    except ValueError:
        pass
    assert "the call failed: ValueError: server said 500" in out[0]


def test_long_text_is_clipped():
    assert len(pt.clip("x" * 5000)) < 800 and "more characters" in pt.clip("x" * 5000)
