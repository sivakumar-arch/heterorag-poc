"""
benchmark_v2/layer_split.py
===========================
Where do the silent errors come from, the plan or the translation? And how often is a bound input inexact?

Formal core v1 separates two assumptions: PC (the plan means what the question means) and TC (each translated
query computes what its step means). ``plan_equal`` compares only the STRUCTURE of the plan (stores, operators, entities,
binds); the wording of the sub-questions, including any literal the planner copied from the question, is ignored.
So a silent answer with an equal structure comes from the translation OR from a wrong sub-question wording; a
silent answer with a different structure is ambiguous: the plan may be wrong, or an equivalent plan may have been
translated wrongly. The same split among CORRECT answers shows how often a different structure is nevertheless
right, which tells how far "structure differs" can be read as "plan is wrong".

    python -m benchmark_v2.layer_split data/floor_*_cost_1.json data/cp1b_*_cost_1.json

Part B reads the step records of rows written since Batch 78: for every step that has a bind, whether the step it
is bound to ended with an exact result (formal core P3: an inexact bound input is the only case in which the
propagation rule needs the monotonicity assumption). Rows without step records are counted and skipped.
Reads result files only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark_v2.floor_compare import pct
from benchmark_v2.repeat_summary import group


def layer_counts(rows: list[dict]) -> dict[str, int]:
    c = {"silent": 0, "silent_plan_equal": 0, "silent_plan_differs": 0, "correct": 0, "correct_plan_differs": 0,
         "no_plan_flag": 0}
    for r in rows:
        eq = r.get("plan_equal")
        if eq is None:
            c["no_plan_flag"] += 1
            continue
        g = group(r["outcome"])
        if g == "silent":
            c["silent"] += 1
            c["silent_plan_equal" if eq else "silent_plan_differs"] += 1
        elif g == "correct":
            c["correct"] += 1
            c["correct_plan_differs"] += 0 if eq else 1
    return c


def bound_inputs(rows: list[dict]) -> dict[str, int]:
    c = {"rows_with_records": 0, "rows_without": 0, "bound_steps": 0, "inexact_input": 0, "silent_with_inexact": 0}
    for r in rows:
        recs = {x["step"]: x for x in r.get("step_records") or []}
        steps = (r.get("plan") or {}).get("steps") if isinstance(r.get("plan"), dict) else None
        if not recs or not steps:
            c["rows_without"] += 1
            continue
        c["rows_with_records"] += 1
        inexact = False
        for s in steps:
            b = s.get("bind")
            if not b:
                continue
            c["bound_steps"] += 1
            src = recs.get(b["input"])
            if src is not None and src["guarantee"] != "exact":
                c["inexact_input"] += 1
                inexact = True
        if inexact and group(r["outcome"]) == "silent":
            c["silent_with_inexact"] += 1
    return c


def render(name: str, rows: list[dict]) -> str:
    a, b = layer_counts(rows), bound_inputs(rows)
    lines = [f"{name}: {len(rows)} rows"]
    if a["silent"]:
        lines.append(f"  silent {a['silent']}: plan structure equal to the template {a['silent_plan_equal']}, "
                     f"plan differs {a['silent_plan_differs']}")
    else:
        lines.append("  silent 0")
    if a["correct"]:
        lines.append(f"  correct {a['correct']}: plan structure differs from the template in {pct(a['correct_plan_differs'], a['correct'])}")
    if a["no_plan_flag"]:
        lines.append(f"  {a['no_plan_flag']} rows carry no plan_equal flag (skipped)")
    if b["rows_with_records"]:
        lines.append(f"  bound steps {b['bound_steps']} in {b['rows_with_records']} rows with step records; "
                     f"inexact bound input in {b['inexact_input']}; silent answers with an inexact bound input {b['silent_with_inexact']}")
    else:
        lines.append(f"  no step records ({b['rows_without']} rows): bound inputs cannot be counted for this file")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    tot = {"silent": 0, "silent_plan_equal": 0, "silent_plan_differs": 0, "correct": 0, "correct_plan_differs": 0}
    for f in args.files:
        try:
            rows = json.loads(Path(f).read_text())["rows"]
        except (OSError, ValueError, KeyError) as ex:
            print(f"error: cannot read {f}: {ex}", file=sys.stderr)
            return 2
        print(render(Path(f).name, rows))
        for k, v in layer_counts(rows).items():
            if k in tot:
                tot[k] += v
    if len(args.files) > 1 and tot["silent"]:
        print(f"all files (files share questions, read as descriptive): silent {tot['silent']}, structure equal "
              f"{pct(tot['silent_plan_equal'], tot['silent'])}, structure differs {pct(tot['silent_plan_differs'], tot['silent'])}; "
              f"correct answers with a structure that differs {pct(tot['correct_plan_differs'], tot['correct'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
