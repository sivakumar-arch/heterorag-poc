"""The rendered-prompt probe finds the prompts a debug log holds and reads their quoting."""
from benchmark_v2 import probe_rendered as pr

LOG = ('time=1 level=INFO msg="other" x=1\n'
       'time=2 level=DEBUG source=routes.go:1 msg="chat request" images=0 prompt="<s>[SYSTEM_PROMPT]say \\"hi\\"\\nnow[/SYSTEM_PROMPT][INST]q[/INST]"\n'
       'time=3 level=DEBUG msg="chat request" images=0 prompt="second"\n')


def test_prompts_are_found_in_order_and_unquoted():
    ps = pr.prompts_in(LOG)
    assert ps == ['<s>[SYSTEM_PROMPT]say "hi"\nnow[/SYSTEM_PROMPT][INST]q[/INST]', "second"]


def test_a_log_without_prompts_gives_none():
    assert pr.prompts_in('time=1 level=INFO msg="x"\n') == []


TOOLS = [{"type": "function", "function": {"name": "sql", "description": "a < b & c",
          "parameters": {"properties": {"query": {"type": "string"}}, "type": "object", "required": ["query"]}}}]


def test_the_prompt_follows_the_template_for_a_first_request():
    p = pr.render_mistral([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}], TOOLS)
    assert p.startswith("[SYSTEM_PROMPT]S[/SYSTEM_PROMPT][AVAILABLE_TOOLS][{")
    assert p.endswith("[/AVAILABLE_TOOLS][INST]Q[/INST]")
    assert '"parameters":{"type":"object","required":["query"],"properties":{"query":{"type":"string"}}}' in p
    assert "a \\u003c b \\u0026 c" in p


def test_later_turns_write_calls_and_results_as_the_template_does():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "sql", "arguments": {"query": "select 1"}}}]},
            {"role": "tool", "content": "1 row"}]
    p = pr.render_mistral(msgs, TOOLS)
    assert "[INST]Q[/INST][TOOL_CALLS]sql[CALL_ID]0[ARGS]{\"query\":\"select 1\"}</s>[TOOL_RESULTS]{\"content\": 1 row}[/TOOL_RESULTS]" in p
    assert "[AVAILABLE_TOOLS]" not in p              # the user message is no longer one of the last two messages


def test_the_first_request_is_sent_raw_and_the_run_goes_on(monkeypatch):
    sent = []
    monkeypatch.setattr(pr, "generate_raw", lambda h, m, p, t, c: (sent.append(p), {"response": "[TOOL_CALLS]x", "eval_count": 3, "done_reason": "stop"})[1])

    class C:
        def __init__(self):
            self.calls = 0

        def chat(self, **kw):
            self.calls += 1
            return "reply"

    out, c = [], C()
    pr.trace_first_request(c, "http://h", "m", 2, 50, 100, write=out.append)
    kw = dict(model="m", messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], tools=TOOLS)
    assert c.chat(**kw) == "reply" and c.chat(**kw) == "reply" and c.calls == 2
    assert len(sent) == 2 and sent[0] == sent[1]                     # only the first request is probed, twice repeated
    assert sum(1 for line in out if "raw output" in line) == 2 and any("rendered prompt:" in line for line in out)
