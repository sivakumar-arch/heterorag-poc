"""
benchmark_v2/probe_check.py
===========================
Live check of the size probes (design B). For every unbound set step of the gold plans, the
store's cheap estimate is compared with the number of rows the step really returns (read up to
the cap), and the two times are compared. No language model is involved.

A probe is *correct* when
  * basis "count":  it equals the real row count and the read was exhausted;
  * basis "lower":  the real read was not exhausted (the cap truncates the step) and the probe
                    reported more than the cap;
  * basis "planner": reported separately; the store's own estimate is only a fallback.

    python -m benchmark_v2.probe_check --questions data/questions_v2_dev_classD_probe.jsonl \
        --out data/probe_check.json [--cap 10000]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from heterorag.federation import Plan
from heterorag.federation.guarantee import Kind
from heterorag.federation.plan import Op

from .reference_queries import reference_queries


def check_record(rec: dict, clients: Mapping, cap: int) -> list[dict]:
    plan = Plan.from_dict({"steps": rec["plan"]})
    queries = reference_queries(rec["template"], rec["params"])
    rows = []
    for s in plan.steps:
        if s.op is not Op.RETRIEVE or s.bind is not None or s.kind is not Kind.SET:
            continue
        client = clients[s.store]
        row = {"question_id": rec["question_id"], "template": rec["template"], "step": s.id,
               "store": s.store.value, "status": "", "probe_n": None, "basis": None, "actual": None,
               "probe_ms": None, "fetch_ms": None}
        try:
            t0 = time.perf_counter()
            est = client.estimate(s, queries[s.id], cap)
            row["probe_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            t0 = time.perf_counter()
            res = client.fetch(s, queries[s.id], {}, cap)
            row["fetch_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        except Exception as ex:
            row["status"] = f"error: {type(ex).__name__}: {str(ex)[:120]}"
            rows.append(row)
            continue
        row["actual"] = len(res.values) if res.exhausted else f">{cap}"
        if est is None:
            row["status"] = "no estimate"
        else:
            row["probe_n"], row["basis"] = est.n, est.basis
            if est.basis == "count":
                row["status"] = "ok" if (res.exhausted and est.n == len(res.values)) else "WRONG"
            elif est.basis == "lower":
                row["status"] = "ok" if (not res.exhausted and est.n > cap) else "WRONG"
            else:
                row["status"] = "planner estimate"
        rows.append(row)
    return rows


def render(rows: list[dict]) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["store"]].append(r)
    lines = [f"{'store':6s} {'steps':>5s} {'ok':>4s} {'wrong':>5s} {'planner':>7s} {'none':>4s} {'error':>5s} "
             f"{'probe ms (median)':>18s} {'fetch ms (median)':>18s}"]
    for st in sorted(by):
        rs = by[st]
        n = lambda f: sum(1 for r in rs if f(r["status"]))
        med = lambda k: (f"{statistics.median([r[k] for r in rs if r[k] is not None]):.1f}"
                         if any(r[k] is not None for r in rs) else "-")
        lines.append(f"{st:6s} {len(rs):5d} {n(lambda s: s == 'ok'):4d} {n(lambda s: s == 'WRONG'):5d} "
                     f"{n(lambda s: s == 'planner estimate'):7d} {n(lambda s: s == 'no estimate'):4d} "
                     f"{n(lambda s: s.startswith('error')):5d} {med('probe_ms'):>18s} {med('fetch_ms'):>18s}")
    bad = [r for r in rows if r["status"] not in ("ok", "planner estimate")]
    for r in bad:
        lines.append(f"{r['question_id']:9s} step {r['step']} {r['store']:5s} {r['status']}  "
                     f"probe={r['probe_n']} ({r['basis']}) actual={r['actual']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, clients: Mapping | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare the size probes with the real row counts")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--only", help="comma-separated template ids")
    ap.add_argument("--cap", type=int, default=10_000)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    recs = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    if args.only:
        want = set(args.only.split(","))
        recs = [r for r in recs if r["template"] in want]
    if not recs:
        print("no questions selected", file=sys.stderr)
        return 2
    if clients is None:
        from heterorag.federation.clients import live_clients
        clients = live_clients()
    rows = [x for r in recs for x in check_record(r, clients, args.cap)]
    print(render(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0 if all(r["status"] in ("ok", "planner estimate") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
