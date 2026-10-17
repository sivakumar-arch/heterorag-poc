"""
benchmark_v2/inspect_rows.py
============================
Prints what happened inside selected rows of a ``run_slice`` output file: the plan the planner
wrote, the plan that was executed, the size estimates, every bind decision with its reason, the
queries, step errors and, for a plan that was rejected, the planner's replies (development split
only; the rows of the test split carry no replies).

    python -m benchmark_v2.inspect_rows data/slice_classD_cost_pilot.json --ids D01-002,D01-004
    python -m benchmark_v2.inspect_rows data/slice_classD_cost_pilot.json --not-correct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def steps_line(plan: dict | None) -> str:
    if not plan:
        return "none"
    return "  ".join(f"{s['id']}:{s['op']}"
                     + (f"/{s['store']}" if s.get("store") else "")
                     + (f"({','.join(s['inputs'])})" if s.get("inputs") else "")
                     + ("+bind" if s.get("bind") else "") for s in plan.get("steps", []))


def describe(r: dict, raw_chars: int = 900, query_chars: int = 160) -> str:
    out = [f"{r['question_id']}  {r['outcome']}  guarantee={r.get('guarantee')}  policy={r.get('bind_policy')}",
           f"  question: {r.get('text', '')}",
           f"  plan     : {steps_line(r.get('plan'))}"]
    if r.get("executed_plan") and r.get("executed_plan") != r.get("plan"):
        out.append(f"  executed : {steps_line(r['executed_plan'])}")
    if r.get("gold_plan"):
        out.append(f"  gold     : {steps_line({'steps': r['gold_plan']})}")
    if r.get("plan_abstain"):
        out.append(f"  planner  : {r['plan_abstain']}  errors={r.get('plan_errors')}  repairs={r.get('plan_repairs')}")
    for i, text in enumerate(r.get("plan_raw") or [], 1):
        if r.get("plan_abstain") or i == len(r["plan_raw"]):
            out.append(f"  reply {i}  : {' '.join(str(text).split())[:raw_chars]}")
            if not r.get("plan_abstain"):
                break
    if r.get("estimates"):
        out.append("  estimates: " + ", ".join(f"{k}={v['n']} ({v['basis']})" for k, v in r["estimates"].items()))
        for k, v in r["estimates"].items():
            if v.get("why"):
                out.append(f"    why {k} fell back: {' '.join(str(v['why']).split())[:260]}")
    for d in r.get("bind_decisions") or []:
        out.append(f"  decision : {d['action']:11s} {d['level']:13s} {d['step']} <- {d['source']}: {d['reason'][:110]}")
    for k, q in (r.get("queries") or {}).items():
        out.append(f"  query {k}  : {' '.join(str(q).split())[:query_chars]}")
    for e in r.get("step_errors") or []:
        out.append(f"  step error: {str(e)[:200]}")
    if r.get("reason"):
        out.append(f"  reason   : {str(r['reason'])[:200]}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--ids", help="comma-separated question ids")
    ap.add_argument("--not-correct", action="store_true", help="every row whose outcome is not 'correct'")
    ap.add_argument("--raw-chars", type=int, default=900)
    args = ap.parse_args(argv)
    rows = json.loads(Path(args.file).read_text())["rows"]
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["question_id"] in want]
    elif args.not_correct:
        rows = [r for r in rows if r["outcome"] != "correct"]
    if not rows:
        print("no rows selected", file=sys.stderr)
        return 2
    print("\n\n".join(describe(r, args.raw_chars) for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
