"""
benchmark_v2/validate_instances.py
==================================
Instance-level validation of generated questions against the live stores
(Benchmark spec 9b, decisions D14 and D15).

For every generated question the gold plan is executed on the live PostgreSQL, Neo4j and
Elasticsearch stores with the hand-written reference queries, and the typed answer is compared
with the gold answer computed from the oracle. No language model is involved. An instance
passes only if the stores reproduce the oracle's answer; a mismatch means a store, the oracle,
the virtual union or the text predicate differs for that instance. Mismatching instances are
dropped from the cleaned question file and counted in a report.

Unlike the generator, this module deliberately uses the system's plan validator, executor and
store clients: it is the integration check that they agree with the oracle.

    python -m benchmark_v2.validate_instances --questions data/questions_v2_dev.jsonl \
        --out data/validation_dev.json --clean data/questions_v2_dev.validated.jsonl

Batch 105b (decision 1 of 2026-11-28): each step is capped at the size of its output entity's table in the
oracle (``--oracle``, default data/oracle_stats.sqlite), not at one global value, so every gold plan runs
unrestricted. The store clients read one row past the limit to tell whether more existed, so a step that reaches
its cap holds more ids than the oracle has of that entity: it is reported as a data mismatch, named by step. A
question generated from another oracle (another fingerprint) is refused. The report file records the oracle path
and fingerprint and the caps used, with the outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from heterorag.federation import Executor, Plan, PlanError
from heterorag.federation.clients import live_clients

from . import union
from .questions import oracle_fingerprint, validator_caps
from .reference_queries import reference_queries

SHOW = 5          # example identifiers printed per mismatching dev instance


@dataclass
class Outcome:
    question_id: str
    template: str
    split: str
    status: str                  # match | mismatch | abstain | error
    missing: int = 0             # gold items the stores did not return
    extra: int = 0               # items the stores returned that gold does not contain
    guarantee: str = ""
    reason: str = ""
    calls: int = 0
    rows_read: int = 0
    examples: dict = field(default_factory=dict)
    cap_steps: list = field(default_factory=list)    # Batch 105b: steps that reached their cap (a data mismatch)


def _same_scalar(a, b) -> bool:
    return a == b


def n1_condition(rec: dict, plan: Plan, records: Mapping) -> str:
    """For an N1 control (Batch 102), the cell's condition on the stores, or "" when it holds: in the final cell
    every retrieve step returned something; in the intermediate cell the named step returned nothing and every
    other retrieve step returned something."""
    if rec.get("control") != "N1":
        return ""
    where = rec.get("empty_at")
    for st in plan.steps:
        if st.op.value != "retrieve":
            continue
        r = records.get(st.id)
        size = len(r.value) if r is not None and r.value is not None else None
        if size is None:
            return f"N1 condition: step {st.id} has no result"
        if where == st.id and size != 0:
            return f"N1 condition: step {st.id} should be empty on the stores, returned {size}"
        if where != st.id and size == 0:
            return f"N1 condition: step {st.id} should be non-empty on the stores, returned 0"
    if where != "final" and where not in {st.id for st in plan.steps}:
        return f"N1 condition: unknown empty_at {where!r}"
    return ""


def validate_record(rec: dict, executor: Executor) -> Outcome:
    base = dict(question_id=rec["question_id"], template=rec["template"], split=rec["split"])
    try:
        plan = Plan.from_dict({"steps": rec["plan"]})
        queries = reference_queries(rec["template"], rec["params"])
        run = executor.run(plan, queries)
    except (PlanError, KeyError, ValueError) as ex:
        return Outcome(**base, status="error", reason=f"{type(ex).__name__}: {ex}")
    except Exception as ex:                                   # connection or driver failure
        return Outcome(**base, status="error", reason=f"{type(ex).__name__}: {ex}")
    tot = run.totals()
    common = dict(**base, guarantee=run.answer.guarantee.value, calls=int(tot["calls"]),
                  rows_read=int(tot["rows_read"]))
    ans, gold = run.answer, rec["gold"]
    capped = sorted(sid for sid, r in run.records.items() if r.cap_hit)
    if capped:          # Batch 104b: never an abstention. Batch 105b: a data mismatch, named by step and cap
        steps = {st.id: st for st in plan.steps}
        what = ", ".join(f"{sid} ({steps[sid].output}, cap {executor.cap_for(steps[sid])})" for sid in capped)
        return Outcome(**common, status="mismatch", cap_steps=capped,
                       reason=f"data mismatch: step(s) {what} reached the oracle's number of ids of the entity")
    if ans.abstained or not ans.exact:
        errs = [r.error for r in run.records.values() if r.error]
        detail = "; ".join(errs)
        why = (f"{ans.reason}: {detail}" if detail else ans.reason or "") or f"answer not exact ({ans.guarantee.value})"
        return Outcome(**common, status="abstain", reason=why[:300])
    if rec["answer_type"] == "ids":
        got, want = set(ans.value), set(gold)
        if got == want:
            why = n1_condition(rec, plan, run.records)
            if why:
                return Outcome(**common, status="mismatch", reason=why)
            return Outcome(**common, status="match")
        return Outcome(**common, status="mismatch", missing=len(want - got), extra=len(got - want),
                       examples={"missing": sorted(want - got)[:SHOW], "extra": sorted(got - want)[:SHOW]})
    if rec["answer_type"] == "ranked":
        if list(ans.value) == list(gold):
            return Outcome(**common, status="match")
        got, want = set(ans.value), set(gold)
        return Outcome(**common, status="mismatch", missing=len(want - got), extra=len(got - want),
                       reason="order differs" if got == want else "",
                       examples={"got": list(ans.value)[:SHOW], "gold": list(gold)[:SHOW]})
    # scalar
    if _same_scalar(ans.value, gold):
        return Outcome(**common, status="match")
    return Outcome(**common, status="mismatch", reason="scalar differs",
                   examples={"got": ans.value, "gold": gold})


def validate(records: list[dict], executor: Executor) -> list[Outcome]:
    return [validate_record(r, executor) for r in records]


def render(outcomes: list[Outcome]) -> str:
    by: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by.setdefault(o.template, []).append(o)
    lines = [f"{'template':8s} {'n':>3s} {'match':>6s} {'mismatch':>9s} {'abstain':>8s} {'error':>6s} {'cap':>4s}"]
    for t in sorted(by):
        c = {k: sum(1 for o in by[t] if o.status == k) for k in ("match", "mismatch", "abstain", "error")}
        cap = sum(1 for o in by[t] if o.cap_steps)          # mismatches at a step's cap (included in mismatch)
        lines.append(f"{t:8s} {len(by[t]):3d} {c['match']:6d} {c['mismatch']:9d} {c['abstain']:8d} {c['error']:6d}"
                     f" {cap:4d}")
    bad = [o for o in outcomes if o.status != "match"]
    if bad:
        lines.append("")
        for o in bad:
            what = f"missing={o.missing} extra={o.extra}" if o.status == "mismatch" else ""
            show = o.examples if o.split == "dev" else {}       # no gold values for the test split
            lines.append(f"{o.question_id:10s} {o.status:9s} {what} {o.reason[:160]} {show if show else ''}")
    n = len(outcomes)
    ok = sum(1 for o in outcomes if o.status == "match")
    lines.append(f"\n{ok} of {n} instances reproduced by the stores")
    return "\n".join(lines)


def main(argv: list[str] | None = None, clients: Mapping | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate generated questions against the live stores")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--oracle", default="data/oracle_stats.sqlite", help="the oracle the questions were made from")
    ap.add_argument("--only", help="comma-separated template ids")
    ap.add_argument("--out", help="JSON report path")
    ap.add_argument("--clean", help="write only matching instances to this JSON lines file")
    args = ap.parse_args(argv)

    records = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        records = [r for r in records if r["template"] in wanted]
    if not records:
        print("no questions selected", file=sys.stderr)
        return 2
    if not Path(args.oracle).exists():
        print(f"oracle not found: {args.oracle}", file=sys.stderr)
        return 2
    db = union.open_union(str(args.oracle))
    try:
        fp, caps = oracle_fingerprint(db), validator_caps(db)
    finally:
        db.close()
    other = sorted({r.get("oracle") for r in records} - {fp}, key=str)
    if other:
        print(f"refused: questions made from another oracle (fingerprints {other}); this oracle is {fp}",
              file=sys.stderr)
        return 2
    top = max(caps.values())
    executor = Executor(clients if clients is not None else live_clients(), batch_size=1000, cap=top,
                        step_cap=lambda st: caps.get(st.output, top))
    print(f"oracle {args.oracle}: fingerprint {fp}; validator caps per entity {caps}")
    outcomes = validate(records, executor)
    print(render(outcomes))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"oracle_path": str(args.oracle), "oracle_fingerprint": fp,
                                              "validator_caps": caps, "outcomes": [asdict(o) for o in outcomes]},
                                             indent=1, sort_keys=True))
    if args.clean:
        keep = {o.question_id for o in outcomes if o.status == "match"}
        with open(args.clean, "w") as f:
            for r in records:
                if r["question_id"] in keep:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"{len(keep)} matching instances written to {args.clean}; "
              f"{len(records) - len(keep)} dropped")
    return 0 if all(o.status == "match" for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
