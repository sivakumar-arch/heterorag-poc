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
  T2     a bound step in the C5 scope returned more ids than its bound input (step records)
  T3     a retrieve step without a bind completed with zero ids (step records)
  C5     run-time containment counter `outside_bound` greater than zero on a step in the C5 scope (step records)

and the unions T1|T2|T3 and S+|T1|T2|T3|C5. Detection is counted on silent rows (outcome `wrong` or
`partial_violated`), false alarms on correct rows. A detector that needs step records or an answer digest is
reported as not scored on a file whose rows lack them.

    python -m benchmark_v2.dev_pass data/t0_*_cost_1.json
    python -m benchmark_v2.dev_pass data/t0_*_cost_1.json --show silent

Scope of T2 and C5 (Batch 99, decided 2026-11-16): ``plan.containment_applies``; a planner-written bind on the graph
store is out of scope. ``--scope wide`` gives the rule before Batch 99 (every same-entity bound step). The origin of
each bind is read from the step records (``bind_origin``, Batch 99 and later) or, for older files, derived: a
bound step of the executed plan that is not bound in the planner's plan was bound by the bind planner.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from benchmark_v2.replay_checks import outcome_class
from heterorag.federation import BIND_ORIGIN_PLANNER, BIND_ORIGIN_POLICY, Plan, containment_applies
from heterorag.federation.semantic_checks import Signatures, candidate_errors, literal_coverage_errors

S_CHECKS = ("R1", "R2", "R3", "R4", "R5")
DETECTORS = ("S", "R6", "R7", "S+", "T1", "T2", "T3", "C5", "T1|T2|T3", "ALL")
SCOPES = ("narrow", "wide")


def _plan(row: dict):
    d = row.get("executed_plan") or row.get("plan")
    if not d:
        return None
    try:
        return Plan.from_dict(d if "steps" in d else {"steps": d}, trust_origin=True)
    except Exception:
        return None


def bind_origins(row: dict) -> dict[str, str]:
    """step id -> who wrote its bind, for every bound step of the plan that ran. Read from the step records when they
    carry it (Batch 99 on); otherwise derived: with an executed plan, a bound step that the planner's plan does not
    bind was bound by the bind planner (``bindplan`` never binds a step the planner already bound); without one,
    the plan that ran is the planner's."""
    plan = _plan(row)
    if plan is None:
        return {}
    recorded = {r["step"]: r["bind_origin"] for r in row.get("step_records") or [] if r.get("bind_origin")}
    raw = row.get("plan") or {}
    steps = raw.get("steps", []) if isinstance(raw, dict) else raw
    planner_bound = {str(d.get("id")) for d in steps if isinstance(d, dict) and d.get("bind")}
    has_executed = bool(row.get("executed_plan"))
    out: dict[str, str] = {}
    for s in plan.steps:
        if s.bind is None:
            continue
        if s.id in recorded:
            out[s.id] = recorded[s.id]
        elif has_executed and s.id not in planner_bound:
            out[s.id] = BIND_ORIGIN_POLICY
        else:
            out[s.id] = BIND_ORIGIN_PLANNER
    return out


def _with_origins(plan, origins: dict[str, str]):
    return plan if plan is None else Plan(tuple(
        replace(s, bind=replace(s.bind, origin=origins.get(s.id, s.bind.origin))) if s.bind is not None else s
        for s in plan.steps))


def in_scope(step, scope: str = "narrow") -> bool:
    """Whether T2 and C5 consider this step: the registered scope (``narrow``) or the rule before Batch 99."""
    if scope == "wide":
        return step.bind is not None and step.output == step.bind.entity
    return containment_applies(step)


def fired(row: dict, sigs: Signatures, scope: str = "narrow") -> tuple[dict[str, bool | None], dict[str, list[str]]]:
    """(detector -> fired or None when not scorable, detector -> reasons). ``scope`` applies to T2 and C5 only."""
    plan, queries = _plan(row), row.get("queries") or {}
    plan = _with_origins(plan, bind_origins(row))
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
            elif in_scope(st, scope):
                src = by.get(st.bind.input)
                if src is not None and (rec.get("value_size") or 0) > (src.get("value_size") or 0):
                    t2 = True
                    why.setdefault("T2", []).append(st.id)
        res["T2"], res["T3"] = t2, t3
        scoped = {st.id for st in plan.steps if st.bind is not None and in_scope(st, scope)}
        res["C5"] = any((r.get("outside_bound") or 0) > 0 for r in recs if r.get("step") in scoped)
        if res["C5"]:
            why["C5"] = [r["step"] for r in recs if r.get("step") in scoped and (r.get("outside_bound") or 0) > 0]
    else:
        res["T2"] = res["T3"] = res["C5"] = None
    triv = [res[k] for k in ("T1", "T2", "T3")]
    res["T1|T2|T3"] = None if all(v is None for v in triv) else any(v for v in triv)
    allv = [res[k] for k in ("S+", "T1", "T2", "T3", "C5")]
    res["ALL"] = None if all(v is None for v in allv) else any(v for v in allv)
    return res, why


def score(rows: list[dict], sigs: Signatures, scope: str = "narrow") -> tuple[dict[str, Counter], Counter, list[dict]]:
    counts: dict[str, Counter] = {d: Counter() for d in DETECTORS}
    cls_n: Counter = Counter()
    detail: list[dict] = []
    for r in rows:
        cls = outcome_class(r)
        cls_n[cls] += 1
        res, why = fired(r, sigs, scope)
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
    ap.add_argument("--scope", choices=SCOPES, default="narrow",
                    help="scope of T2 and C5: narrow (registered, Batch 99) or wide (the rule before Batch 99)")
    args = ap.parse_args(argv)
    sigs = Signatures.poc()
    total: dict[str, Counter] = {d: Counter() for d in DETECTORS}
    total_n: Counter = Counter()
    for f in args.files:
        rows = json.loads(Path(f).read_text())["rows"]
        counts, cls_n, detail = score(rows, sigs, args.scope)
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
