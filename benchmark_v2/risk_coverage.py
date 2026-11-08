"""
benchmark_v2/risk_coverage.py
=============================
Selective risk against coverage for the systems of a comparison. A system that may abstain is judged by two
numbers together: how much of the question set it answers (coverage) and how often an answer it gives is
wrong (selective risk). Both are read from the recorded outcomes, nothing is rerun and nothing is rescored.

Outcomes of a row (as in `cp1_report.category`):

    correct           answered, equal to the truth (label exact)
    wrong             answered with label exact, not equal to the truth                      silent error
    partial_valid     answered with a weaker label (subset, superset, lower, upper) that holds   flagged, not an error
    partial_violated  answered with a weaker label that does NOT hold                           silent error
    abstain_*, no_answer   no answer

Two operating points per file:

    exact only   answered = correct + wrong                      (what a user who accepts only exact answers sees)
    all answers  answered = exact + partial_valid + partial_violated   (flagged partial answers accepted)

A system without guarantee labels (the agent arms) has no partial outcomes, so both points coincide. Intervals
are Clopper-Pearson. The point of the table is the pair (coverage, risk): a system that is right more often only
because it answers less is not comparable on accuracy alone.

    python -m benchmark_v2.risk_coverage data/floor_gptoss20b_cost_1.json data/floor_gptoss20b_python_text_1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark_v2.floor_compare import pct

EXACT = ("correct", "wrong")
PARTIAL = ("partial_valid", "partial_violated")
SILENT = ("wrong", "partial_violated")


def points(rows: list[dict]) -> dict:
    n = len(rows)
    c = {}
    for r in rows:
        c[r["outcome"]] = c.get(r["outcome"], 0) + 1
    g = lambda *names: sum(c.get(x, 0) for x in names)
    out = {"n": n}
    for label, names in (("exact only", EXACT), ("all answers", EXACT + PARTIAL)):
        answered = g(*names)
        silent = g(*[s for s in SILENT if s in names])
        correct = g("correct")
        out[label] = {"answered": answered, "silent": silent, "correct": correct,
                      "flagged": g("partial_valid") if "partial_valid" in names else 0}
    return out


def render(name: str, rows: list[dict]) -> str:
    p = points(rows)
    n = p["n"]
    lines = [f"{name}: {n} questions"]
    for label in ("exact only", "all answers"):
        q = p[label]
        a = q["answered"]
        cov = pct(a, n)
        risk = pct(q["silent"], a) if a else "n/a (no answers)"
        yld = pct(q["correct"], a) if a else "n/a"
        lines.append(f"  {label:12s} coverage {cov}")
        lines.append(f"  {'':12s} silent among answered {risk}")
        lines.append(f"  {'':12s} correct among answered {yld}; flagged partial answers {q['flagged']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    for f in args.files:
        rows = json.loads(Path(f).read_text())["rows"]
        print(render(Path(f).name, rows))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
