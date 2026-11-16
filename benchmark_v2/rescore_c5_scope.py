"""
benchmark_v2/rescore_c5_scope.py
================================
Batch 99 check: does the narrowed scope of C5 and T2 (``plan.containment_applies``, decided 2026-11-16) change any
firing on saved result files? The scope was decided before this check from the code and the descriptors (a code read
of shape S1's plan); this script is a check that nothing changes, not a tuning step. If any row changes, the result
is reported and the work stops for a decision.

For every file and row it scores T2 and C5 under the wide scope (the rule before Batch 99: every same-entity bound
step) and under the narrow scope (a planner-written bind on the graph store is out), with the same row data, and
prints:

  * per file, the counts of T2 and C5 firings on silent, correct and other rows under both scopes;
  * the same-entity bound steps of the plans that ran, by store and bind origin, and how many are out of scope;
  * every row whose T2 or C5 result differs between the scopes, with the steps involved.

No model call, no store access. Exit status 0 if no row changes, 1 if any row changes, 2 on an input error.

    python -m benchmark_v2.rescore_c5_scope data/t0_qwen_cost_1.json data/t0_gptoss_cost_1.json \\
        data/t0_mistral_cost_1.json data/t0_gemma4_cost_1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from benchmark_v2.dev_pass import _plan, _with_origins, bind_origins, fired, in_scope
from benchmark_v2.replay_checks import outcome_class
from heterorag.federation.semantic_checks import Signatures

COMPARED = ("T2", "C5")


def bound_steps(row: dict) -> Counter:
    """(store, origin, in narrow scope) -> number of same-entity bound steps in the plan that ran."""
    plan = _with_origins(_plan(row), bind_origins(row))
    out: Counter = Counter()
    if plan is None:
        return out
    for s in plan.steps:
        if s.bind is not None and s.output == s.bind.entity:
            out[(s.store.value if s.store else "-", s.bind.origin, in_scope(s, "narrow"))] += 1
    return out


def compare(rows: list[dict], sigs: Signatures) -> tuple[dict[str, dict[str, Counter]], Counter, list[dict]]:
    counts = {sc: {d: Counter() for d in COMPARED} for sc in ("wide", "narrow")}
    steps: Counter = Counter()
    changed: list[dict] = []
    for r in rows:
        cls = outcome_class(r)
        res = {sc: fired(r, sigs, sc) for sc in ("wide", "narrow")}
        for sc, (fr, _) in res.items():
            for d in COMPARED:
                v = fr.get(d)
                counts[sc][d]["unscored_" + cls if v is None else (cls if v else "quiet_" + cls)] += 1
        steps.update(bound_steps(r))
        diff = [d for d in COMPARED if res["wide"][0].get(d) != res["narrow"][0].get(d)]
        if diff:
            changed.append({"question_id": r.get("question_id"), "class": cls, "detectors": diff,
                            "wide": {d: res["wide"][1].get(d) for d in diff},
                            "narrow": {d: res["narrow"][1].get(d) for d in diff}})
    return counts, steps, changed


def render(name: str, counts, steps: Counter, changed: list[dict]) -> str:
    out = [f"{name}"]
    for d in COMPARED:
        for sc in ("wide", "narrow"):
            k = counts[sc][d]
            out.append(f"  {d} {sc:6s}: fires on {k['silent']} silent, {k['correct']} correct, {k['other']} other rows"
                       f" (not scored: {k['unscored_silent'] + k['unscored_correct'] + k['unscored_other']})")
    if steps:
        out.append("  same-entity bound steps in the plans that ran (store, bind origin, in narrow scope): "
                   + ", ".join(f"{st}/{o}/{'in' if ins else 'OUT'} {n}" for (st, o, ins), n in sorted(steps.items())))
    else:
        out.append("  no same-entity bound step in any plan that ran")
    out.append(f"  rows whose T2 or C5 result changes: {len(changed)}")
    for c in changed:
        out.append(f"    {c['question_id']} ({c['class']}): {', '.join(c['detectors'])}; wide fired on "
                   f"{json.dumps(c['wide'])}, narrow on {json.dumps(c['narrow'])}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    sigs = Signatures.poc()
    n_changed = 0
    for f in args.files:
        try:
            rows = json.loads(Path(f).read_text())["rows"]
        except (OSError, ValueError, KeyError) as ex:
            print(f"{f}: cannot read rows ({type(ex).__name__}: {ex})", file=sys.stderr)
            return 2
        counts, steps, changed = compare(rows, sigs)
        print(render(Path(f).name, counts, steps, changed))
        n_changed += len(changed)
    if n_changed:
        print(f"RESULT: {n_changed} row(s) change between the scopes. Report and stop; the scope is not tuned.")
        return 1
    print("RESULT: no row changes between the scopes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
