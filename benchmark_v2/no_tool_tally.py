"""
benchmark_v2/no_tool_tally.py
=============================
What did an agent run look like when it ended `no_tool`? The harness stops a run after two consecutive replies
without a tool call (the second follows the nudge "Use the tools, and finish with final_answer."). The result
file keeps those replies, so the last one can be read and classified. Descriptive only: nothing is rescored and
nothing is written.

Category of the LAST reply of every `no_tool` run, in this order (the first that applies):

  hidden         no text, but at least HIDDEN_MIN_TOKENS generated tokens: the server dropped a malformed tool call
  empty          no text and fewer generated tokens than that
  final_marker   the text mentions final_answer (a final answer the registered reader refused, or a mention)
  tool_as_text   the text looks like a tool call or a query written out instead of called (code fence, SELECT,
                 MATCH (, a "query"/"code"/"body" key, [TOOL_CALLS], <tool_call>, sql( cypher( es( python( )
  prose          anything else: the model wrote words and neither called a tool nor gave a final answer

The patterns were written after reading a handful of transcripts, so the categories describe these runs and are
not a registered measure. Each run is also split by how many tool calls it had made before it stopped: none
(the model never used a tool), 1 to 3, 4 or more.

    python -m benchmark_v2.no_tool_tally data/floor_mistral_small32_24b_python_text_1.json data/floor_mistral_small32_24b_plain_text_1.json
    python -m benchmark_v2.no_tool_tally data/floor_mistral_small32_24b_plain_text_1.json --show prose

Two further lines are printed for every file. "Repeated replies" lists last replies that occur word for word in
two or more runs (a canned reply). "Tokens beyond the visible text" tests whether a reply that has text also lost
content: for the replies with text that is not cut by the 600 character limit of the transcript, the number of
generated tokens per character is compared with the median of the same file; a reply with more than twice the
median ratio and at least TOKEN_SLACK more tokens than the median ratio explains is flagged. A flagged reply
generated much more than the harness received, which is what a dropped tool call after a sentence of text looks
like. The rule is fixed here before any Mistral number is read from it, and it flags candidates, not proof.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_v2.floor_compare import HIDDEN_MIN_TOKENS

TEXT_LIMIT = 600          # the transcript keeps at most this many characters of a reply
TOKEN_SLACK = 40
CATEGORIES = ("hidden", "empty", "final_marker", "tool_as_text", "prose")
_FINAL = re.compile(r"final[_ ]?answer", re.I)
_TOOLISH = re.compile(
    r"\[TOOL_CALLS\]|<tool_call>|```|\bSELECT\b|\bMATCH\s*\(|[\"'](?:query|code|body)[\"']\s*:|"
    r"\b(?:sql|cypher|es|python)\s*\(", re.I)


def last_text(row: dict) -> dict | None:
    for t in reversed(row.get("transcript") or []):
        if t.get("tool") == "_text":
            return t
    return None


def classify(entry: dict | None) -> str:
    if entry is None:
        return "empty"
    text = str(entry.get("text") or "").strip()
    if not text:
        return "hidden" if int(entry.get("reply_tokens") or 0) >= HIDDEN_MIN_TOKENS else "empty"
    if _FINAL.search(text):
        return "final_marker"
    if _TOOLISH.search(text):
        return "tool_as_text"
    return "prose"


def calls_made(row: dict) -> int:
    return sum(1 for t in (row.get("transcript") or []) if not str(t.get("tool", "_")).startswith("_"))


def bucket(n: int) -> str:
    return "none" if n == 0 else "1 to 3" if n <= 3 else "4 or more"


def tally(rows: list[dict]) -> tuple[Counter, dict, list[dict]]:
    """(category counts, {calls bucket: category counts}, one record per no_tool run)."""
    total: Counter = Counter()
    by_calls: dict[str, Counter] = defaultdict(Counter)
    recs = []
    for r in rows:
        if r.get("stop") != "no_tool":
            continue
        e = last_text(r)
        cat = classify(e)
        k = calls_made(r)
        total[cat] += 1
        by_calls[bucket(k)][cat] += 1
        recs.append({"question_id": r["question_id"], "template": r.get("template"), "category": cat, "calls": k,
                     "text": " ".join(str((e or {}).get("text") or "").split()),
                     "raw_chars": len(str((e or {}).get("text") or "")),
                     "tokens": int((e or {}).get("reply_tokens") or 0)})
    return total, by_calls, recs


def repeated(recs: list[dict], top: int = 3) -> list[tuple[str, int]]:
    c = Counter(x["text"] for x in recs if x["text"])
    return [(t, n) for t, n in c.most_common(top) if n >= 2]


def token_excess(recs: list[dict]) -> tuple[int, int, list[dict]]:
    """(replies usable for the test, median tokens per character, flagged records). Usable: text present, fewer
    than TEXT_LIMIT characters (so not cut), generated tokens recorded."""
    use = [x for x in recs if x["text"] and 0 < x["raw_chars"] < TEXT_LIMIT and x["tokens"] > 0]
    if len(use) < 5:
        return len(use), 0.0, []
    med = statistics.median(x["tokens"] / x["raw_chars"] for x in use)
    flagged = [x for x in use if x["tokens"] / x["raw_chars"] > 2 * med and x["tokens"] - med * x["raw_chars"] >= TOKEN_SLACK]
    return len(use), med, flagged


def render(name: str, rows: list[dict], total: Counter, by_calls: dict) -> str:
    n_stop = sum(total.values())
    out = [f"{name}: {len(rows)} runs, {n_stop} ended no_tool"]
    out.append(f"  {'last reply':14s} {'runs':>5s}   " + "   ".join(f"{b:>9s}" for b in ("none", "1 to 3", "4 or more")))
    out.append(f"  {'(tool calls made first)':14s}")
    for c in CATEGORIES:
        out.append(f"  {c:14s} {total[c]:>5d}   " + "   ".join(f"{by_calls[b][c]:>9d}" for b in ("none", "1 to 3", "4 or more")))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show", choices=CATEGORIES, help="print examples of this category")
    ap.add_argument("--limit", type=int, default=6, help="examples per file with --show")
    args = ap.parse_args(argv)
    for f in args.files:
        rows = json.loads(Path(f).read_text())["rows"]
        total, by_calls, recs = tally(rows)
        print(render(Path(f).name, rows, total, by_calls))
        rep = repeated(recs)
        print("  Repeated replies: " + ("; ".join(f"{n} runs: {t[:90]!r}" for t, n in rep) if rep else "none"))
        use, med, flagged = token_excess(recs)
        if use < 5:
            print(f"  Tokens beyond the visible text: too few usable replies ({use})")
        else:
            print(f"  Tokens beyond the visible text: {len(flagged)} of {use} usable replies flagged "
                  f"(median {med:.2f} tokens per character)")
            for x in flagged[: args.limit]:
                print(f"    {x['question_id']} {x['category']}: {x['tokens']} tokens for {x['raw_chars']} characters: {x['text'][:110]!r}")
        if args.show:
            for x in [x for x in recs if x["category"] == args.show][: args.limit]:
                print(f"\n  {x['question_id']} ({x['template']}, {x['calls']} calls first)\n    {x['text'][:260]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
