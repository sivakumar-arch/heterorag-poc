"""
benchmark_v2/cp1_report.py
==========================
Tables for checkpoint CP1 from the raw result files of ``run_slice`` (our pipeline) and
``agent_baseline`` (the free-form agent), so that no number is computed by hand.

A system is one name with one or more files; each file is one repeat over the same questions:

    python -m benchmark_v2.cp1_report \
        --system cost=data/cp1_cost_1.json,data/cp1_cost_2.json,data/cp1_cost_3.json \
        --system none=data/cp1_none_1.json,... \
        --system plain=data/cp1_plain_1.json,... --system python=data/cp1_python_1.json,... \
        --ours cost --agents plain,python --out data/cp1_report.md

Definitions (fixed before the CP1 run, 2026-10-17)

  Four mutually exclusive outcomes per run
    correct        the answer equals the gold answer;
    silent error   an answer that is wrong and was given as if it were right: for the pipeline an
                   answer labelled EXACT that is wrong (``wrong``) or a label that the answer
                   violates (``partial_violated``); for an agent any wrong answer;
    flagged        the pipeline's partial answer whose label holds (``partial_valid``): the user is
                   told it may be incomplete or too large. An agent has no such outcome;
    abstained      the system said it cannot answer, or an agent ended with no answer (turn budget).

  Reliable: correct plus flagged, the runs in which the user got a right answer or a partial answer
  that is honestly labelled as such. It is not a failure rate and not a pass rate: a partial answer
  is a result of its own. An agent cannot label a partial answer, so its reliable rate is its correct rate.

  Answer F1: set precision and recall of the returned ids against the gold ids; a scalar answer
  scores 1 if equal and 0 otherwise; an abstention scores 0. A flagged partial answer is scored on
  the set it returned (F1 "as flagged"); the strict F1 scores it 0 as well.
  Cost: model tokens at the price given on the command line (default Haiku 4.5, 1 and 5 USD per
  million input and output tokens); cost per correct answer is total cost over the correct runs.
  Unit of analysis: the question. Repeats are averaged inside a question; intervals are percentile
  bootstrap intervals over questions (paired when two systems are compared). Rates are reported
  in percent.

CP1 decision rule (written before results): reposition if the better agent arm (by F1 on the
whole set) is within 5 points of our F1 and its cost per correct answer is within 1.5 times ours.
Silent errors: "matches" means the 95 percent interval of the paired difference includes zero.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PRICE_IN, PRICE_OUT = 1.0, 5.0          # USD per million tokens (Haiku 4.5)
CATEGORIES = ("correct", "silent", "flagged", "abstained")
BOOT = 2000
SEED = 20261017


# ----------------------------------------------------------------------------- one run

def category(row: dict) -> str:
    oc = row.get("outcome")
    if oc == "correct":
        return "correct"
    if oc in ("wrong", "partial_violated"):
        return "silent"
    if oc == "partial_valid":
        return "flagged"
    if oc in ("abstain_expected", "abstain_unexpected", "no_answer"):
        return "abstained"
    raise ValueError(f"unknown outcome {oc!r} in {row.get('question_id')}")


def f1_of(row: dict, strict: bool = False) -> float:
    cat = category(row)
    if cat == "correct":
        return 1.0
    if cat == "abstained" or (strict and cat == "flagged"):
        return 0.0
    f1 = row.get("f1")
    return float(f1) if f1 is not None else 0.0          # a wrong scalar scores 0


def tokens(row: dict) -> tuple[int, int]:
    u = row["usage"]
    if "planner" in u:                                    # pipeline: planner and translator
        return (u["planner"]["prompt_tokens"] + u["translator"]["prompt_tokens"],
                u["planner"]["reply_tokens"] + u["translator"]["reply_tokens"])
    return u["prompt_tokens"], u["reply_tokens"]


def run_record(row: dict, price_in: float, price_out: float) -> dict:
    tin, tout = tokens(row)
    return {"qid": row["question_id"], "template": row["template"], "stratum": row.get("stratum") or "?",
            "cat": category(row), "f1": f1_of(row), "f1_strict": f1_of(row, strict=True),
            "cost": (tin * price_in + tout * price_out) / 1e6, "rows_read": row.get("rows_read", 0),
            "seconds": row.get("total_s", 0.0)}


# ----------------------------------------------------------------------------- one system

class System:
    """Per question: the means over its repeats."""

    def __init__(self, name: str, files: list[str], price_in: float = PRICE_IN, price_out: float = PRICE_OUT):
        self.name, self.files = name, files
        per_q: dict[str, list[dict]] = defaultdict(list)
        self.repeat_ids: list[list[str]] = []
        for f in files:
            rows = json.loads(Path(f).read_text())["rows"]
            ids = [r["question_id"] for r in rows]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{f}: duplicate question ids")
            self.repeat_ids.append(ids)
            for r in rows:
                per_q[r["question_id"]].append(run_record(r, price_in, price_out))
        base = set(self.repeat_ids[0])
        for f, ids in zip(files, self.repeat_ids):
            if set(ids) != base:
                raise ValueError(f"{f}: a different set of questions than {files[0]}")
        self.qids = sorted(per_q)
        self.repeats = len(files)
        self.meta = {qid: (runs[0]["template"], runs[0]["stratum"]) for qid, runs in per_q.items()}
        self.v: dict[str, dict] = {}
        for qid, runs in per_q.items():
            n = len(runs)
            d = {c: sum(r["cat"] == c for r in runs) / n for c in CATEGORIES}
            d.update(f1=sum(r["f1"] for r in runs) / n, f1_strict=sum(r["f1_strict"] for r in runs) / n,
                     cost=sum(r["cost"] for r in runs) / n, rows_read=sum(r["rows_read"] for r in runs) / n,
                     seconds=sum(r["seconds"] for r in runs) / n)
            self.v[qid] = d


# ----------------------------------------------------------------------------- statistics

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _quant(xs, q):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    i = q * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def boot(qids: list[str], stat, rng: random.Random, b: int = BOOT) -> tuple[float, float, float]:
    """Point estimate and 95 percent percentile interval of ``stat(list of question ids)``."""
    point = stat(qids)
    if len(qids) < 2:
        return point, float("nan"), float("nan")
    vals = [stat([qids[rng.randrange(len(qids))] for _ in qids]) for _ in range(b)]
    return point, _quant(vals, 0.025), _quant(vals, 0.975)


def rate(sys_: System, key: str):
    return lambda ids: 100.0 * _mean([sys_.v[q][key] for q in ids])


def cost_per_correct(sys_: System):
    def f(ids):
        c = sum(sys_.v[q]["correct"] for q in ids)
        return sum(sys_.v[q]["cost"] for q in ids) / c if c else float("nan")
    return f


def fmt(t, unit="%", dec=1):
    p, lo, hi = t
    if p != p:
        return "n/a"
    if lo != lo:
        return f"{p:.{dec}f}{unit}"
    return f"{p:.{dec}f}{unit} [{lo:.{dec}f}, {hi:.{dec}f}]"


# ----------------------------------------------------------------------------- tables

def table_overall(systems: list[System], qids: list[str], rng) -> list[str]:
    hdr = ("| system | repeats | correct | silent error | flagged | abstained | reliable (correct + flagged) | "
           "F1 as flagged | F1 strict | cost per question (USD) | cost per correct (USD) | seconds | rows read |")
    out = [hdr, "|" + "---|" * 13]
    for s in systems:
        row = [s.name, str(s.repeats)]
        for key in CATEGORIES:
            row.append(fmt(boot(qids, rate(s, key), rng)))
        row.append(fmt(boot(qids, lambda ids, s=s: 100.0 * _mean([s.v[q]["correct"] + s.v[q]["flagged"] for q in ids]), rng)))
        row.append(fmt(boot(qids, rate(s, "f1"), rng)))
        row.append(fmt(boot(qids, rate(s, "f1_strict"), rng)))
        row.append(fmt(boot(qids, lambda ids, s=s: _mean([s.v[q]["cost"] for q in ids]), rng), "", 4))
        row.append(fmt(boot(qids, cost_per_correct(s), rng), "", 4))
        row.append(f"{_mean([s.v[q]['seconds'] for q in qids]):.1f}")
        row.append(f"{_mean([s.v[q]['rows_read'] for q in qids]):.0f}")
        out.append("| " + " | ".join(row) + " |")
    return out


def table_by(systems: list[System], qids: list[str], group, title: str) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for q in qids:
        groups[group(systems[0], q)].append(q)
    out = [f"| {title} | n | " + " | ".join(f"{s.name} correct / silent / flagged / abstained" for s in systems) + " |",
           "|" + "---|" * (2 + len(systems))]
    for g in sorted(groups):
        ids = groups[g]
        cells = []
        for s in systems:
            cells.append(" / ".join(f"{100 * _mean([s.v[q][c] for q in ids]):.0f}" for c in CATEGORIES))
        out.append(f"| {g} | {len(ids)} | " + " | ".join(cells) + " |")
    return out


def paired(a: System, b: System, qids: list[str], key: str, rng) -> tuple:
    """a minus b in percentage points of ``key``, with a paired bootstrap interval."""
    return boot(qids, lambda ids: 100.0 * _mean([a.v[q][key] - b.v[q][key] for q in ids]), rng)


def table_paired(pairs: list[tuple[System, System]], qids: list[str], rng) -> list[str]:
    out = ["| A minus B | correct (points) | silent error (points) | F1 as flagged (points) | questions A better / equal / worse (correct) |",
           "|---|---|---|---|---|"]
    for a, b in pairs:
        w = sum(a.v[q]["correct"] > b.v[q]["correct"] for q in qids)
        l = sum(a.v[q]["correct"] < b.v[q]["correct"] for q in qids)
        out.append(f"| {a.name} minus {b.name} | {fmt(paired(a, b, qids, 'correct', rng), '')} | "
                   f"{fmt(paired(a, b, qids, 'silent', rng), '')} | {fmt(paired(a, b, qids, 'f1', rng), '')} | "
                   f"{w} / {len(qids) - w - l} / {l} |")
    return out


def decision(ours: System, agents: list[System], qids: list[str], rng) -> list[str]:
    best = max(agents, key=lambda a: _mean([a.v[q]["f1"] for q in qids]))
    gap = paired(ours, best, qids, "f1", rng)                         # ours minus best agent, points
    c_ours, c_agent = cost_per_correct(ours)(qids), cost_per_correct(best)(qids)
    ratio = c_agent / c_ours if c_ours == c_ours and c_ours > 0 else float("nan")
    within_f1 = gap[0] <= 5.0
    within_cost = ratio == ratio and ratio <= 1.5
    sil = paired(best, ours, qids, "silent", rng)                     # agent minus ours
    sil_match = sil[1] <= 0.0 <= sil[2] if sil[1] == sil[1] else False
    lines = [
        f"- better agent arm by F1 on the whole set: **{best.name}**",
        f"- F1 (as flagged), ours minus {best.name}: {fmt(gap, ' points')}; the agent is within 5 points: **{'yes' if within_f1 else 'no'}**",
        f"- cost per correct answer, {best.name} over ours: {ratio:.2f} (agent {c_agent:.4f} USD, ours {c_ours:.4f} USD); within 1.5 times: **{'yes' if within_cost else 'no'}**",
        f"- silent-error rate, {best.name} minus ours: {fmt(sil, ' points')}; the interval includes zero (the agent matches): **{'yes' if sil_match else 'no'}**",
    ]
    if within_f1 and within_cost:
        lines.append("- criterion met: the system is not the contribution; reposition on semantics, the cardinality finding and the benchmark."
                     + (" The silent-error rate also matches: consider a benchmark and analysis paper." if sil_match else ""))
    else:
        lines.append("- criterion not met: the pipeline differs from the agent on accuracy or on cost; continue to CP2.")
    return lines


# ----------------------------------------------------------------------------- main

def parse_system(spec: str) -> tuple[str, list[str]]:
    name, _, files = spec.partition("=")
    if not name or not files:
        raise argparse.ArgumentTypeError(f"expected NAME=file1,file2,... got {spec!r}")
    return name, [f for f in files.split(",") if f]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, type=parse_system)
    ap.add_argument("--ours", required=True, help="name of our pipeline system for the CP1 rule")
    ap.add_argument("--agents", required=True, help="comma-separated names of the agent systems")
    ap.add_argument("--price-in", type=float, default=PRICE_IN)
    ap.add_argument("--price-out", type=float, default=PRICE_OUT)
    ap.add_argument("--out", help="write the report (markdown) here as well")
    args = ap.parse_args(argv)

    systems = [System(n, f, args.price_in, args.price_out) for n, f in args.system]
    by_name = {s.name: s for s in systems}
    if len(by_name) != len(systems):
        print("system names must be distinct", file=sys.stderr)
        return 2
    agent_names = args.agents.split(",")
    for n in [args.ours, *agent_names]:
        if n not in by_name:
            print(f"unknown system {n!r}", file=sys.stderr)
            return 2
    qids = systems[0].qids
    for s in systems[1:]:
        if s.qids != qids:
            print(f"{s.name}: not the same questions as {systems[0].name} "
                  f"({len(set(qids) ^ set(s.qids))} differ)", file=sys.stderr)
            return 2
    ours, agents = by_name[args.ours], [by_name[n] for n in agent_names]
    rng = random.Random(SEED)

    L: list[str] = [f"# CP1 report: {len(qids)} questions, systems: " + ", ".join(f"{s.name} ({s.repeats} repeats)" for s in systems), ""]
    L += ["## Overall (percent of runs; 95 percent interval over questions)", ""] + table_overall(systems, qids, rng)
    L += ["", "## By template", ""] + table_by(systems, qids, lambda s, q: s.meta[q][0], "template")
    d04 = [q for q in qids if systems[0].meta[q][0] == "D04"]
    if d04:
        L += ["", "## D04 by stratum (bound-set band)", ""] + table_by(systems, d04, lambda s, q: s.meta[q][1], "stratum")
    pairs = [(ours, a) for a in agents] + [(a, b) for i, a in enumerate(agents) for b in agents[i + 1:]]
    pairs += [(ours, s) for s in systems if s not in agents and s is not ours]
    L += ["", "## Paired differences (percentage points; paired bootstrap over questions)", ""] + table_paired(pairs, qids, rng)
    L += ["", "## CP1 decision rule", ""] + decision(ours, agents, qids, rng)
    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
