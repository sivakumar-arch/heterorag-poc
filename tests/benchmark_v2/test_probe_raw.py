"""The raw probe prints the delivered reply and the streamed replay, and hands the delivered reply on unchanged."""
from benchmark_v2 import probe_raw as pr


class Client:
    """Stands in for ollama.Client: a reply as a dict, or a list of chunks when stream=True."""

    def __init__(self, reply, chunks, fail_stream=False):
        self.reply, self.chunks, self.fail_stream, self.calls = reply, chunks, fail_stream, []

    def chat(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            if self.fail_stream:
                raise RuntimeError("stream refused")
            return iter(self.chunks)
        return self.reply


EMPTY = {"message": {"role": "assistant", "content": ""}, "done_reason": "stop", "eval_count": 90, "prompt_eval_count": 2735}
CHUNKS = [{"message": {"content": "", "role": "assistant"}, "done": False},
          {"message": {"content": "", "role": "assistant"}, "done": True, "done_reason": "stop", "eval_count": 90}]


def test_delivered_reply_is_returned_unchanged_and_both_forms_are_printed():
    out = []
    c = pr.trace_client(Client(EMPTY, CHUNKS), write=out.append)
    assert c.chat(model="m", messages=[], stream=False) is EMPTY
    text = "\n".join(out)
    assert "eval_count=90" in text and "content (0 characters)" in text
    assert "the same request streamed: 2 chunks" in text and "chunks that carry text or a call: 0 of 2" in text


def test_the_replay_is_the_same_request_with_streaming_on():
    inner = Client(EMPTY, CHUNKS)
    c = pr.trace_client(inner, write=lambda s: None)
    c.chat(model="m", messages=[{"role": "user", "content": "q"}], tools=[{"x": 1}], options={"num_predict": 5})
    first, second = inner.calls
    assert not first.get("stream") and second["stream"] is True
    assert {k: v for k, v in second.items() if k != "stream"} == {k: v for k, v in first.items() if k != "stream"}


def test_a_failed_replay_is_reported_and_does_not_end_the_probe():
    out = []
    c = pr.trace_client(Client(EMPTY, CHUNKS, fail_stream=True), write=out.append)
    assert c.chat(model="m", messages=[]) is EMPTY
    assert any("the streamed replay failed: RuntimeError: stream refused" in s for s in out)


def test_stream_summary_joins_text_and_keeps_calls():
    s = pr.summarise_stream([{"message": {"content": "a"}}, {"message": {"content": "b", "tool_calls": [{"function": {"name": "sql"}}]}},
                             {"message": {}, "done": True, "done_reason": "stop", "eval_count": 7}])
    assert s["text"] == "ab" and s["tool_calls"][0]["function"]["name"] == "sql" and s["eval_count"] == 7
