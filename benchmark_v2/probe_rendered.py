"""
benchmark_v2/probe_rendered.py
==============================
Shows what a local model writes for the exact prompt the server built, before the server reads it as a tool call.
``probe_raw`` showed that for some turns the server returns no text and no tool call although it counts generated
tokens. The server's own reading of the model output is the suspect, so this builds the prompt the chat template makes, sends it to /api/generate with raw=true, which skips the
chat template and the tool-call reading, and prints the model's output as it is. Nothing is written to the results files.

    python -m benchmark_v2.probe_rendered --model ollama:mistral-small3.2-nosys --repeats 3 --questions data/q.jsonl --ids ID --arm plain --cap 10000 --accept-text-final --out /tmp/p.json

The prompt for the first request of the agent run on the question is built here from the model's chat template
(Mistral Small 3.2, as printed by ``ollama show --template``; the server's own rendering option is not in Ollama 0.40.8),
then sent raw. Everything not named here is passed to ``benchmark_v2.agent_baseline``. As a fallback, ``--log``
reads the prompt text from a server log written with OLLAMA_DEBUG=1 when that log holds it.

The output is a fresh sample of the model for that prompt, so it can differ from the sample that returned nothing;
repeat it a few times.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

_PROMPT = re.compile(r'prompt="((?:[^"\\]|\\.)*)"')


def unquote(s: str) -> str:
    """A Go-quoted string body as text."""
    try:
        return json.loads('"' + s + '"')
    except ValueError:
        return s.encode("latin-1", "backslashreplace").decode("unicode_escape")


def prompts_in(log_text: str) -> list[str]:
    """The prompts the server logged for chat requests, oldest first."""
    out = []
    for line in log_text.splitlines():
        if "request" in line and ' prompt="' in line:
            m = _PROMPT.search(line)
            if m:
                out.append(unquote(m.group(1)))
    return out


def generate_raw(host: str, model: str, prompt: str, tokens: int, num_ctx: int) -> dict:
    body = json.dumps({"model": model, "prompt": prompt, "raw": True, "stream": False,
                       "options": {"num_predict": tokens, "num_ctx": num_ctx}}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def _post(host: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(host.rstrip("/") + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def _go_json(x) -> str:
    """Compact JSON the way Go's encoder writes it for the template: no spaces, and < > & written as \\u003c \\u003e \\u0026."""
    t = json.dumps(x, separators=(",", ":"), ensure_ascii=False)
    return t.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _tools_json(tools: list[dict]) -> str:
    out = []
    for t in tools:
        f = dict(t.get("function") or {})
        prm = dict(f.get("parameters") or {})
        prm = {k: prm[k] for k in ("type", "defs", "items", "required", "properties") if k in prm} | \
              {k: v for k, v in prm.items() if k not in ("type", "defs", "items", "required", "properties")}
        out.append({"type": t.get("type", "function"),
                    "function": {"name": f.get("name"), "description": f.get("description"), "parameters": prm}})
    return _go_json(out)


def render_mistral(messages: list[dict], tools: list[dict] | None) -> str:
    """The prompt the chat template of Mistral Small 3.2 in Ollama builds (template read with ``ollama show --template``).
    The beginning-of-text token is added by the server and is not part of this text. The tools are written where the
    template puts them, before the last user message (or the one before the last message). This is a re-implementation
    for a diagnostic: compare its length with the server's logged ``prompt_len`` before trusting it."""
    out = []
    n = len(messages)
    for i, m in enumerate(messages):
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            out.append(f"[SYSTEM_PROMPT]{content}[/SYSTEM_PROMPT]")
        elif role == "user":
            if tools and n - i <= 2:
                out.append(f"[AVAILABLE_TOOLS]{_tools_json(tools)}[/AVAILABLE_TOOLS]")
            out.append(f"[INST]{content}[/INST]")
        elif role == "assistant":
            if content:
                out.append(content + ("</s>" if n - i != 1 else ""))
            elif m.get("tool_calls"):
                for k, tc in enumerate(m["tool_calls"]):
                    fn = tc.get("function") or {}
                    out.append(f"[TOOL_CALLS]{fn.get('name')}[CALL_ID]{k}[ARGS]{_go_json(fn.get('arguments') or {})}")
                out.append("</s>")
        elif role == "tool":
            out.append('[TOOL_RESULTS]{"content": ' + content + "}[/TOOL_RESULTS]")
    return "".join(out)


def show_raw(host: str, model: str, prompt: str, repeats: int, tokens: int, num_ctx: int, write=print) -> None:
    write(f"prompt: {len(prompt)} characters")
    write("start of the prompt: " + json.dumps(prompt[:200]))
    write("end of the prompt:   " + json.dumps(prompt[-400:]))
    for i in range(repeats):
        r = generate_raw(host, model, prompt, tokens, num_ctx)
        write(f"--- raw output {i + 1}: eval_count={r.get('eval_count')} done_reason={r.get('done_reason')!r}")
        write("    " + json.dumps(r.get("response")))


def trace_first_request(client, host: str, model: str, repeats: int, tokens: int, num_ctx: int, write=print):
    """On the first chat request of an agent run, render the prompt from its messages and tools and send it raw."""
    original, done = client.chat, {"first": False}

    def chat(*args, **kwargs):
        if not done["first"]:
            done["first"] = True
            try:
                p = render_mistral(kwargs.get("messages") or [], kwargs.get("tools"))
                write(f"rendered prompt: {len(p)} characters (compare with prompt_len of the server's 'completion request' line)")
                show_raw(host, model, p, repeats, tokens, num_ctx, write)
            except Exception as ex:                                        # a diagnostic must never end the probe
                write(f"the raw probe failed: {type(ex).__name__}: {str(ex)[:300]}")
        return original(*args, **kwargs)

    client.chat = chat
    return client


def main(argv: list[str] | None = None, llm=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--log", help="a server log written with OLLAMA_DEBUG=1 (used when it holds the prompt text)")
    ap.add_argument("--model", required=True, help="name as Ollama knows it; an ollama: prefix is accepted")
    ap.add_argument("--index", type=int, default=0, help="with --log: which logged prompt to send (0: the first request)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--host", default=None)
    ap.add_argument("--num-ctx", type=int, default=32768)
    a, rest = ap.parse_known_args(argv)
    name = a.model.split("ollama:", 1)[-1]
    from heterorag.federation.llm_config import ollama_settings
    host = a.host or ollama_settings()["host"]
    if a.log:
        ps = prompts_in(open(a.log, errors="replace").read())
        print(f"prompts found in the log: {len(ps)}")
        if not ps:
            print("none: the log has no chat request with the prompt text")
            return 1
        show_raw(host, name, ps[a.index], a.repeats, a.tokens, a.num_ctx)
        return 0
    from . import agent_baseline as ab
    llm = llm or ab.make_agent("ollama:" + name)
    trace_first_request(llm._client, host, name, a.repeats, a.tokens, a.num_ctx)
    return ab.main(rest, llm=llm)


if __name__ == "__main__":
    sys.exit(main())
