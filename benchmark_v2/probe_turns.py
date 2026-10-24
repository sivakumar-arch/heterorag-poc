"""
benchmark_v2/probe_turns.py
===========================
Shows what a model actually returns on each turn of the agent loop: its reasoning, its text, the tool calls the
server extracted, and how the turn ended. It exists to tell a model that cannot use the tools from an adapter or
server that loses the calls, before a long run is trusted or repeated. It runs the normal agent runner on the
questions you select, so use it on one or two questions, never on a whole set.

    python -m benchmark_v2.probe_turns --model ollama:gpt-oss:20b --questions data/q.jsonl --ids CP1B-D01-001 \
        --arm python --out /tmp/probe.json

Everything after --model is passed to ``benchmark_v2.agent_baseline``. Nothing is written to the results files.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import agent_baseline as ab


def clip(text: str, n: int = 700) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + f" ... [{len(text) - n} more characters]"


def trace(llm, write=print):
    """Wrap ``llm.turn`` so that every reply is printed as it arrives."""
    original, count = llm.turn, {"turn": 0}

    def turn(system, messages, tools, max_tokens):
        count["turn"] += 1
        try:
            t = original(system, messages, tools, max_tokens)
        except Exception as ex:
            write(f"--- turn {count['turn']}: the call failed: {type(ex).__name__}: {clip(ex, 400)}")
            raise
        thought = "".join(b.get("thinking", "") for b in t.content if b.get("type") == "thinking")
        write(f"--- turn {count['turn']}: stop={t.stop_reason} prompt_tokens={t.prompt_tokens} "
              f"reply_tokens={t.reply_tokens} calls={[c.name for c in t.calls]}")
        if thought:
            write("reasoning: " + clip(thought))
        if t.text:
            write("text: " + clip(t.text))
        for c in t.calls:
            write(f"call {c.name}: " + clip(json.dumps(c.input, default=str), 400))
        if not t.calls:
            write("(no tool call in this reply)")
        return t

    llm.turn = turn
    return llm


def main(argv: list[str] | None = None, llm=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="for example ollama:gpt-oss:20b (default: HETERORAG_AGENT_MODEL)")
    args, rest = ap.parse_known_args(argv)
    llm = llm or ab.make_agent(args.model)
    code = ab.main(rest, llm=trace(llm))
    print(f"\nadapter counters: {getattr(llm, 'stats', None)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
