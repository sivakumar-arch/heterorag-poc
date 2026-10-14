"""
benchmark_v2/agent_audit.py
===========================
Reads the output of ``benchmark_v2.agent_baseline`` (development split only: the test split stores
no transcripts) and tells apart two kinds of failure before any comparison is drawn.

    harness   the run was cut short or the model was shown less than the store returned, for a
              reason that belongs to the evaluation set-up and not to the method: the turn budget,
              a reply cut off at the token limit, a tool result clipped for display, a result
              limited by the hard cap, an error in the tool itself.
    method    the agent finished and was wrong: an incomplete or an over-complete set, a wrong
              number, a wrong answer kind, an abstention where the question is answerable.

A row can carry both. The harness flags do not say the harness caused the failure; they say
where to look first. Changing the harness after seeing results is allowed only before the
comparison, only in the direction of fairness to the agent, and every change is logged in the
checkpoint document (the agent's score on a set it has been tuned on is not reported).

    python -m benchmark_v2.agent_audit data/agent_plain_pilot.json data/agent_python_pilot.json
    python -m benchmark_v2.agent_audit data/agent_plain_pilot.json --show D05-0003
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SHOW_ROWS = 30


def flags(row: dict) -> list[str]:
    """Harness conditions visible in the transcript of one run."""
    tr = row.get("transcript") or []
    out: list[str] = []
    if row.get("stop") == "max_turns":
        out.append("budget")
    if row.get("stop") == "no_tool":
        out.append("no_tool")
    if row.get("stop") == "error":
        out.append("error")
    if any(t.get("tool") == "_turn" and t.get("stop_reason") == "max_tokens" for t in tr):
        out.append("max_tokens")
    calls = [t for t in tr if t.get("tool") in ("sql", "cypher", "es", "python")]
    if any(t.get("exhausted") is False for t in calls):
        out.append("cap_hit")
    if any(t.get("rows", 0) > SHOW_ROWS for t in calls):
        out.append("display_clipped")
    if any(t.get("cut") for t in calls):
        out.append("output_cut")
    errs = [t for t in calls if t.get("error")]
    if errs:
        out.append(f"tool_errors={len(errs)}")
    if any("window" in (t.get("result") or "").lower() and "large" in (t.get("result") or "").lower() for t in errs):
        out.append("es_window")
    if any("longer than" in (t.get("result") or "") for t in errs):
        out.append("python_timeout")
    return out


def diagnosis(row: dict) -> str:
    """Method-level reading of a finished run (empty for a correct run)."""
    oc = row.get("outcome")
    if oc == "correct":
        return ""
    if oc == "no_answer":
        return "no answer given"
    if oc == "abstain_unexpected":
        return "abstained on an answerable question"
    p, r = row.get("precision"), row.get("recall")
    if p is None:
        return f"wrong {row.get('answer_kind') or 'answer'}"
    if p == 1.0 and r is not None and r < 1.0:
        return "incomplete set (all returned are right, some missing)"
    if r == 1.0 and p < 1.0:
        return "over-complete set (all gold found, extras returned)"
    if (row.get("f1") or 0) == 0:
        return "disjoint from gold"
    return "partly right and partly wrong"


def load(path: str) -> tuple[dict, list[dict]]:
    d = json.loads(Path(path).read_text())
    return d.get("meta", {}), d["rows"]


def render(meta: dict, rows: list[dict]) -> str:
    lines = [f"arm={meta.get('arm')}  model={meta.get('model')}  max_turns={meta.get('max_turns')}  "
             f"cap={meta.get('cap')}  n={len(rows)}", ""]
    hdr = f"{'question':10s} {'out':9s} {'stop':9s} {'turn':>4s} {'call':>4s} {'err':>3s} {'f1':>5s}  flags / reading"
    lines.append(hdr)
    for r in rows:
        if r["outcome"] == "correct" and not flags(r):
            continue
        f1 = "-" if r.get("f1") is None else f"{r['f1']:.2f}"
        reading = diagnosis(r)
        lines.append(f"{r['question_id']:10s} {r['outcome'][:9]:9s} {r['stop'][:9]:9s} {r['turns']:4d} "
                     f"{r['tool_calls']:4d} {r['tool_errors']:3d} {f1:>5s}  "
                     f"{', '.join(flags(r)) or '-'}{'  |  ' + reading if reading else ''}")
    c = Counter()
    for r in rows:
        if r["outcome"] != "correct":
            fl = [f.split("=")[0] for f in flags(r)]
            c.update(fl or ["(none: finished, wrong, no harness flag)"])
    lines += ["", "harness flags among the runs that were not correct (a run can carry several):"]
    lines += [f"  {k:45s} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])] or ["  none"]
    return "\n".join(lines)


def show(rows: list[dict], qid: str, width: int = 400) -> str:
    row = next((r for r in rows if r["question_id"] == qid), None)
    if row is None:
        return f"{qid}: not in this file"
    out = [f"{qid}  {row['text']}", f"outcome={row['outcome']} stop={row['stop']} kind={row.get('answer_kind')} "
           f"reason={row.get('reason')!r} got={row.get('got')!r}", ""]
    for i, t in enumerate(row.get("transcript") or [], 1):
        extra = {k: t[k] for k in ("rows", "exhausted", "shown", "chars", "cut", "stop_reason") if k in t}
        out.append(f"{i:2d}. {t['tool']}  {'ERROR ' if t.get('error') else ''}{extra or ''}")
        out.append(f"    in : {str(t.get('input', ''))[:width]}")
        if "result" in t:
            out.append(f"    out: {str(t['result'])[:width]}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show", help="print the whole transcript of this question id")
    args = ap.parse_args(argv)
    for path in args.files:
        meta, rows = load(path)
        print(f"== {path}")
        print(show(rows, args.show) if args.show else render(meta, rows))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
