"""
benchmark_v2/replay_checks.py
=============================
Apply the candidate checks of `natural_error_checks` to the queries already recorded in result files
(`run_slice` outputs), with no model call and no store access. For each file and each check it counts the rows
in which the check fires, split by the recorded outcome:

  correct      the check would reject a query of a correct answer: a harmless rejection (false alarm)
  silent       outcome `wrong` (an answer that is wrong, and was not flagged): a catch
  other        every other outcome (abstentions, flagged partial answers)

A rejected translation is retried by the pipeline, so a harmless rejection costs a repair, not an answer, but
the count is the quantity the F2 rule limits (at most twice the baseline rate of harmless rejections).

    python -m benchmark_v2.replay_checks data/floor_*_cost_1.json data/cp1b_*_cost_1.json
    python -m benchmark_v2.replay_checks data/floor_mistral_small32_24b_cost_1.json --show correct

Only the executed plan's queries are read. A row without queries or without a readable plan is skipped and
counted. Nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_v2.natural_error_checks import CHECKS, Signatures, candidate_errors
from heterorag.federation import Plan


def outcome_class(row: dict) -> str:
    o = row.get("outcome")
    return "correct" if o == "correct" else "silent" if o == "wrong" else "other"


def replay_rows(rows: list[dict], sigs: Signatures) -> tuple[dict, list[dict], int]:
    """(counts[check][class], firings, skipped). A firing is one (row, step, check) with the messages."""
    counts: dict[str, Counter] = {c: Counter() for c in CHECKS}
    firings: list[dict] = []
    skipped = 0
    for r in rows:
        plan_d = r.get("executed_plan") or r.get("plan")
        queries = r.get("queries") or {}
        if not plan_d or not queries:
            skipped += 1
            continue
        try:
            plan = Plan.from_dict(plan_d if "steps" in plan_d else {"steps": plan_d})
        except Exception:
            skipped += 1
            continue
        fired: set[str] = set()
        for sid, text in queries.items():
            st = plan.by_id().get(sid)
            if st is None or not text:
                continue
            for check, errs in candidate_errors(st, str(text), sigs).items():
                if errs:
                    fired.add(check)
                    firings.append({"question_id": r["question_id"], "outcome": r.get("outcome"), "step": sid,
                                    "check": check, "errors": errs, "query": str(text)})
        cls = outcome_class(r)
        for c in fired:
            counts[c][cls] += 1
    return counts, firings, skipped


def render(name: str, rows: list[dict], counts: dict, skipped: int) -> str:
    n = len(rows)
    cls = Counter(outcome_class(r) for r in rows)
    out = [f"{name}: {n} rows ({cls['correct']} correct, {cls['silent']} silent, {cls['other']} other)"
           + (f", {skipped} skipped (no queries or plan)" if skipped else "")]
    out.append(f"  {'check':6s} {'fires on correct':>17s} {'on silent':>10s} {'on other':>9s}")
    for c in CHECKS:
        k = counts[c]
        out.append(f"  {c:6s} {k['correct']:>8d} of {cls['correct']:<5d} {k['silent']:>4d} of {cls['silent']:<3d} {k['other']:>9d}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show", choices=["correct", "silent", "other"], help="print the firings of this class")
    args = ap.parse_args(argv)
    sigs = Signatures.poc()
    total: dict[str, Counter] = defaultdict(Counter)
    shown: list[dict] = []
    for f in args.files:
        rows = json.loads(Path(f).read_text())["rows"]
        counts, firings, skipped = replay_rows(rows, sigs)
        print(render(Path(f).name, rows, counts, skipped))
        by_id = {r["question_id"]: outcome_class(r) for r in rows}
        shown += [dict(x, file=Path(f).name) for x in firings if args.show and by_id[x["question_id"]] == args.show]
    for x in shown:
        print(f"\n[{x['file']}] {x['question_id']} step {x['step']} {x['check']} ({x['outcome']}): "
              f"{' | '.join(x['errors'])}\n    {' '.join(x['query'].split())[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
