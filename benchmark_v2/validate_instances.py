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


def _same_scalar(a, b) -> bool:
    return a == b


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
    if ans.abstained or not ans.exact:
        errs = [r.error for r in run.records.values() if r.error]
        detail = "; ".join(errs)
        return Outcome(**common, status="abstain",
                       reason=(f"{ans.reason}: {detail}" if detail else ans.reason or "")[:300])
    if rec["answer_type"] == "ids":
        got, want = set(ans.value), set(gold)
        if got == want:
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
    lines = [f"{'template':8s} {'n':>3s} {'match':>6s} {'mismatch':>9s} {'abstain':>8s} {'error':>6s}"]
    for t in sorted(by):
        c = {k: sum(1 for o in by[t] if o.status == k) for k in ("match", "mismatch", "abstain", "error")}
        lines.append(f"{t:8s} {len(by[t]):3d} {c['match']:6d} {c['mismatch']:9d} {c['abstain']:8d} {c['error']:6d}")
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
    executor = Executor(clients if clients is not None else live_clients(), batch_size=1000, cap=200_000)
    outcomes = validate(records, executor)
    print(render(outcomes))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps([asdict(o) for o in outcomes], indent=1, sort_keys=True))
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
