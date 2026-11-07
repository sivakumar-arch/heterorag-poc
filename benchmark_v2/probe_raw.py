"""
benchmark_v2/probe_raw.py
=========================
Shows what the Ollama server sends back for the turns of an agent question, before the adapter reads it. It
exists to tell a model that stops after an announcement from a server that generated tokens and delivered
nothing: ``probe_turns`` shows what the adapter made of a reply, this shows the reply itself.

For every turn the normal request is sent as the adapter sends it, and the answer is printed whole (content,
reasoning, tool calls, finish reason, tokens generated). The identical request is then sent a second time with
streaming and every chunk is printed, so that text the server held back or dropped at the end of the stream
becomes visible. The second request is a diagnostic only: its reply is not used and nothing is written to the
results files. Use it on one question.

    python -c "import benchmark_v2.probe_raw as p;p.main(['--model','ollama:mistral-small3.2-nosys','--questions','data/questions_v2_dev_classD_cp1b.final.jsonl','--ids','CP1B-D05-001','--arm','plain','--cap','10000','--accept-text-final','--out','/tmp/probe_raw.json'])"

Everything after --model is passed to ``benchmark_v2.agent_baseline``.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import agent_baseline as ab


def plain(x):
    """A server reply (a pydantic object or a dict) as plain data."""
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if isinstance(x, dict):
        return x
    return dict(x) if hasattr(x, "keys") else {"value": str(x)}


def summarise_stream(chunks: list[dict]) -> dict:
    """What a streamed reply added up to: the text, the tool calls, the last chunk's counters."""
    text, calls, thinking, last = [], [], [], {}
    for c in chunks:
        m = c.get("message") or {}
        if m.get("content"):
            text.append(m["content"])
        if m.get("thinking"):
            thinking.append(m["thinking"])
        calls.extend(m.get("tool_calls") or [])
        if c.get("done"):
            last = c
    return {"chunks": len(chunks), "text": "".join(text), "thinking": "".join(thinking), "tool_calls": calls,
            "done_reason": last.get("done_reason"), "eval_count": last.get("eval_count"),
            "prompt_eval_count": last.get("prompt_eval_count")}


def trace_client(client, write=print):
    """Wrap ``client.chat`` so that each request is also replayed with streaming and both replies are printed."""
    original, count = client.chat, {"call": 0}

    def chat(*args, **kwargs):
        count["call"] += 1
        n = count["call"]
        r = original(*args, **kwargs)
        d = plain(r)
        m = plain(d.get("message") or {})
        write(f"=== request {n}: reply as delivered: done_reason={d.get('done_reason')!r} eval_count={d.get('eval_count')} "
              f"prompt_eval_count={d.get('prompt_eval_count')}")
        write(f"    content ({len(m.get('content') or '')} characters): {json.dumps(m.get('content'))}")
        if m.get("thinking"):
            write(f"    thinking: {json.dumps(m.get('thinking'))[:700]}")
        write(f"    tool_calls: {json.dumps(m.get('tool_calls'), default=str)[:700]}")
        try:
            chunks = [plain(c) for c in original(*args, **{**kwargs, "stream": True})]
            s = summarise_stream(chunks)
            write(f"--- the same request streamed: {s['chunks']} chunks, done_reason={s['done_reason']!r} "
                  f"eval_count={s['eval_count']}")
            write(f"    text ({len(s['text'])} characters): {json.dumps(s['text'])[:1200]}")
            if s["thinking"]:
                write(f"    thinking: {json.dumps(s['thinking'])[:700]}")
            write(f"    tool_calls: {json.dumps(s['tool_calls'], default=str)[:700]}")
            shown = [c for c in chunks if (c.get("message") or {}).get("content") or (c.get("message") or {}).get("tool_calls")]
            write(f"    chunks that carry text or a call: {len(shown)} of {len(chunks)}")
        except Exception as ex:                                            # a diagnostic must never end the probe
            write(f"--- the streamed replay failed: {type(ex).__name__}: {str(ex)[:300]}")
        return r

    client.chat = chat
    return client


def main(argv: list[str] | None = None, llm=None) -> int:
    from . import probe_turns
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="for example ollama:mistral-small3.2-nosys (default: HETERORAG_AGENT_MODEL)")
    args, rest = ap.parse_known_args(argv)
    llm = llm or ab.make_agent(args.model)
    trace_client(llm._client)
    code = ab.main(rest, llm=probe_turns.trace(llm))
    print(f"\nadapter counters: {getattr(llm, 'stats', None)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
