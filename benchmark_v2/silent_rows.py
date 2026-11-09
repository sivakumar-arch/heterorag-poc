"""
benchmark_v2/silent_rows.py
===========================
Print the silent answers (outcome wrong or partial_violated) of result files in a compact form, so that each one can
be read and classified by hand: the question, the template plan next to the plan that was run, and the final query of
every retrieve step. Used to classify the faults behind the numbers of ``layer_split``: a service that cannot answer a
sub-question, a wrong store, a wrong literal in the sub-question, or a wrong translation of a right sub-question.

    python -m benchmark_v2.silent_rows data/floor_mistral_small32_24b_cost_1.json data/floor_gptoss20b_cost_1.json

Reads result files only. ``--width`` sets the number of characters shown per query (default 200).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark_v2.repeat_summary import group


def plan_line(steps: list[dict] | None) -> str:
    """One line: id:store (or operator), the bind source in brackets, the inputs of a combine step in parentheses."""
    if not steps:
        return "(no plan)"
    parts = []
    for s in steps:
        label = s.get("store") or s.get("op", "?")
        if s.get("mode"):
            label += "." + s["mode"]
        bind = f"[bind {s['bind']['input']}]" if s.get("bind") else ""
        ins = f"({','.join(s['inputs'])})" if s.get("op") not in (None, "retrieve") and s.get("inputs") else ""
        parts.append(f"{s['id']}:{label}{bind}{ins}")
    return "  ".join(parts)


def one_line(text: object, width: int) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= width else s[:width - 3] + "..."


def render_row(name: str, r: dict, width: int) -> str:
    plan = r.get("plan")
    steps = plan.get("steps") if isinstance(plan, dict) else plan
    lines = [f"=== {name}  {r['question_id']}  [{r.get('template')}]  {r['outcome']}  plan_equal={r.get('plan_equal')}",
             f"  Q:    {one_line(r.get('text', ''), 260)}",
             f"  gold: {plan_line(r.get('gold_plan'))}",
             f"  ran:  {plan_line(steps)}"]
    queries = r.get("queries") or {}
    store_of = {s["id"]: s.get("store") for s in (steps or [])}
    sub_q = {s["id"]: s.get("question") for s in (steps or [])}
    for sid, q in queries.items():
        lines.append(f"  {sid} ({store_of.get(sid)}) asked: {one_line(sub_q.get(sid, ''), width)}")
        lines.append(f"       query: {one_line(q, width)}")
    if "got" in r:
        lines.append(f"  got:  {one_line(r['got'], 160)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--width", type=int, default=200)
    args = ap.parse_args(argv)
    shown = 0
    for f in args.files:
        try:
            rows = json.loads(Path(f).read_text())["rows"]
        except (OSError, ValueError, KeyError) as ex:
            print(f"error: cannot read {f}: {ex}", file=sys.stderr)
            return 2
        for r in rows:
            if group(r["outcome"]) == "silent":
                print(render_row(Path(f).name, r, args.width))
                shown += 1
    print(f"\n{shown} silent answers shown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
