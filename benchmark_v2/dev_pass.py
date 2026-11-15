"""
benchmark_v2/dev_pass.py
========================
The dev-pass analysis of the pre-registration (section 11), saved as a script so that every number can be
regenerated from the result files with no model call and no store access.

For each result file it scores, on the recorded rows only:

  S      the registered static set, C-free: R1 to R5 on the recorded query of every retrieve step
  R6     literal coverage (plan level, in-sample)
  R7     additive bind (step level, from the premise PM)
  S+     S together with R6 and R7
  T1     final answer is an empty set labelled exact (benchmark artefact where no gold set is empty)
  T2     a same-entity bound step returned more ids than its bound input (step records)
  T3     a retrieve step without a bind completed with zero ids (step records)
  C5     run-time containment counter `outside_bound` greater than zero (step records)

and the unions T1|T2|T3 and S+|T1|T2|T3|C5. Detection is counted on silent rows (outcome `wrong` or
`partial_violated`), false alarms on correct rows. A detector that needs step records or an answer digest is
reported as not scored on a file whose rows lack them.

    python -m benchmark_v2.dev_pass data/t0_*_cost_1.json
    python -m benchmark_v2.dev_pass data/t0_*_cost_1.json --show silent
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from benchmark_v2.replay_checks import outcome_class
from heterorag.federation import Plan
from heterorag.federation.semantic_checks import Signatures, candidate_errors, literal_coverage_errors

S_CHECKS = ("R1", "R2", "R3", "R4", "R5")
DETECTORS = ("S", "R6", "R7", "S+", "T1", "T2", "T3", "C5", "T1|T2|T3", "ALL")


def _plan(row: dict):
    d = row.get("executed_plan") or row.get("plan")
    if not d:
        return None
    try:
        return Plan.from_dict(d if "steps" in d else {"steps": d})
    except Exception:
        return None


def fired(row: dict, sigs: Signatures) -> tuple[dict[str, bool | None], dict[str, list[str]]]:
    """(detector -> fired or None when not scorable, detector -> reasons)."""
    plan, queries = _plan(row), row.get("queries") or {}
    why: dict[str, list[str]] = {}
    res: dict[str, bool | None] = {}
    s = r7 = False
    if plan is not None and queries:
        for sid, text in queries.items():
            st = plan.by_id().get(sid)
            if st is None or not text:
                continue
            found = candidate_errors(st, str(text), sigs)
            if any(found[c] for c in S_CHECKS):
                s = True
                why.setdefault("S", []).extend(f"{c} {sid}" for c in S_CHECKS if found[c])
            if found["R7"]:
                r7 = True
                why.setdefault("R7", []).append(sid)
        res["S"], res["R7"] = s, r7
    else:
        res["S"] = res["R7"] = None
    if queries:
        errs = literal_coverage_errors(row.get("text", ""), queries)
        res["R6"] = bool(errs)
        if errs:
            why["R6"] = errs
    else:
        res["R6"] = None
    res["S+"] = None if res["S"] is None else bool(res["S"] or res["R7"] or res["R6"])

    dg, recs = row.get("answer_digest"), row.get("step_records")
    res["T1"] = None if not dg else bool(dg.get("type") == "ids" and dg.get("size") == 0 and row.get("guarantee") == "exact")
    if recs and plan is not None:
        by = {r["step"]: r for r in recs}
        t2 = t3 = False
        for st in plan.steps:
            rec = by.get(st.id)
            if rec is None:
                continue
            if st.bind is None:
                if rec.get("status") == "complete" and rec.get("value_size") == 0:
                    t3 = True
                    why.setdefault("T3", []).append(st.id)
            elif st.output == st.bind.entity:
                src = by.get(st.bind.input)
                if src is not None and (rec.get("value_size") or 0) > (src.get("value_size") or 0):
                    t2 = True
                    why.setdefault("T2", []).append(st.id)
        res["T2"], res["T3"] = t2, t3
        res["C5"] = any((r.get("outside_bound") or 0) > 0 for r in recs)
    else:
        res["T2"] = res["T3"] = res["C5"] = None
    triv = [res[k] for k in ("T1", "T2", "T3")]
    res["T1|T2|T3"] = None if all(v is None for v in triv) else any(v for v in triv)
    allv = [res[k] for k in ("S+", "T1", "T2", "T3", "C5")]
    res["ALL"] = None if all(v is None for v in allv) else any(v for v in allv)
    return res, why


def score(rows: list[dict], sigs: Signatures) -> tuple[dict[str, Counter], Counter, list[dict]]:
    counts: dict[str, Counter] = {d: Counter() for d in DETECTORS}
    cls_n: Counter = Counter()
    detail: list[dict] = []
    for r in rows:
        cls = outcome_class(r)
        cls_n[cls] += 1
        res, why = fired(r, sigs)
        for d in DETECTORS:
            if res.get(d) is None:
                counts[d]["unscored_" + cls] += 1
            elif res[d]:
                counts[d][cls] += 1
        detail.append({"question_id": r["question_id"], "class": cls, "fired": [d for d in DETECTORS if res.get(d)], "why": why})
    return counts, cls_n, detail


def render(name: str, counts: dict[str, Counter], cls_n: Counter) -> str:
    out = [f"{name}: {sum(cls_n.values())} rows ({cls_n['correct']} correct, {cls_n['silent']} silent, {cls_n['other']} other)",
           f"  {'detector':10s} {'detects silent':>16s} {'false alarms (correct)':>24s}"]
    for d in DETECTORS:
        k = counts[d]
        note = ""
        if k["unscored_silent"] or k["unscored_correct"]:
            note = f"  (not scored on {k['unscored_silent']} silent, {k['unscored_correct']} correct rows)"
        out.append(f"  {d:10s} {k['silent']:>8d} of {cls_n['silent']:<4d} {k['correct']:>12d} of {cls_n['correct']:<5d}{note}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show", choices=["correct", "silent", "other"], help="print per-row detail for this class")
    ap.add_argument("--pooled", action="store_true", help="also print the totals over all files")
    args = ap.parse_args(argv)
    sigs = Signatures.poc()
    total: dict[str, Counter] = {d: Counter() for d in DETECTORS}
    total_n: Counter = Counter()
    for f in args.files:
        rows = json.loads(Path(f).read_text())["rows"]
        counts, cls_n, detail = score(rows, sigs)
        print(render(Path(f).name, counts, cls_n))
        for d in DETECTORS:
            total[d].update(counts[d])
        total_n.update(cls_n)
        if args.show:
            for x in detail:
                if x["class"] == args.show:
                    print(f"    {x['question_id']}: {', '.join(x['fired']) or 'nothing fired'}"
                          + (f"  {json.dumps(x['why'])}" if x["why"] else ""))
    if args.pooled and len(args.files) > 1:
        print(render("all files", total, total_n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
