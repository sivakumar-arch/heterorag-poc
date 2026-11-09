"""
benchmark_v2/repeat_summary.py
==============================
The repeats of one system on the same questions (rule L: repeats 2 and 3 of a rung that passed the 50 percent
screen). Reads the result files only; nothing is rerun and nothing is rescored.

For the repeats of one system it prints:
    per repeat      correct, silent (wrong or partial_violated), flagged, abstained, with Clopper-Pearson intervals;
    pooled          the mean of the per-repeat rates, with a percentile interval from a question-level bootstrap
                    (4,000 resamples, seed 20261025; whole questions are resampled, so the repeated runs of a
                    question stay together and the interval is not too narrow);
    stability       how many questions were correct in every repeat, in some repeats, in none; how many questions
                    had a silent error in at least one repeat; how many changed outcome category between repeats.

    python -m benchmark_v2.repeat_summary data/floor_gptoss20b_cost_1.json data/floor_gptoss20b_cost_2.json data/floor_gptoss20b_cost_3.json

All files must hold the same question ids (the script stops otherwise).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from benchmark_v2.cp1_report import _mean, _quant
from benchmark_v2.floor_compare import BOOT, SEED, pct

SILENT = ("wrong", "partial_violated")
FLAGGED = ("partial_valid",)


def group(outcome: str) -> str:
    if outcome == "correct":
        return "correct"
    if outcome in SILENT:
        return "silent"
    if outcome in FLAGGED:
        return "flagged"
    return "abstained"


def load(files: list[str]) -> list[dict[str, str]]:
    """One dict per file: question id -> outcome. Raises ValueError if the question sets differ."""
    runs = []
    for f in files:
        rows = json.loads(Path(f).read_text())["rows"]
        runs.append({r["question_id"]: r["outcome"] for r in rows})
    ids = set(runs[0])
    for f, run in zip(files, runs):
        if set(run) != ids:
            raise ValueError(f"{f} does not hold the same question ids as {files[0]}")
    return runs


def pooled(runs: list[dict[str, str]], name: str, rng: random.Random) -> tuple[float, float, float]:
    """Mean over repeats of the share of questions in group `name`, with a question-level bootstrap interval."""
    qids = sorted(runs[0])
    value = {q: _mean([1.0 if group(r[q]) == name else 0.0 for r in runs]) for q in qids}
    stat = lambda ids: 100.0 * _mean([value[q] for q in ids])
    vals = [stat([qids[rng.randrange(len(qids))] for _ in qids]) for _ in range(BOOT)]
    return stat(qids), _quant(vals, 0.025), _quant(vals, 0.975)


def stability(runs: list[dict[str, str]]) -> dict[str, int]:
    qids = sorted(runs[0])
    k = len(runs)
    out = {"questions": len(qids), "correct_all": 0, "correct_some": 0, "correct_none": 0,
           "silent_any": 0, "changed_category": 0}
    for q in qids:
        gs = [group(r[q]) for r in runs]
        c = gs.count("correct")
        out["correct_all"] += c == k
        out["correct_none"] += c == 0
        out["correct_some"] += 0 < c < k
        out["silent_any"] += "silent" in gs
        out["changed_category"] += len(set(gs)) > 1
    return out


def render(files: list[str], runs: list[dict[str, str]]) -> str:
    n = len(runs[0])
    lines = [f"{len(runs)} repeats, {n} questions"]
    for f, run in zip(files, runs):
        g = [group(o) for o in run.values()]
        lines.append(f"  {Path(f).name}")
        for name in ("correct", "silent", "flagged", "abstained"):
            lines.append(f"      {name:9s} {pct(g.count(name), n)}")
    lines.append("  pooled over repeats (mean of per-repeat rates; question-level bootstrap interval)")
    for name in ("correct", "silent", "flagged", "abstained"):
        m, lo, hi = pooled(runs, name, random.Random(SEED))
        lines.append(f"      {name:9s} {m:.1f}% [{lo:.1f}, {hi:.1f}]")
    s = stability(runs)
    lines.append("  stability across repeats")
    lines.append(f"      correct in every repeat {s['correct_all']}, in some repeats {s['correct_some']}, in none {s['correct_none']}")
    lines.append(f"      questions with a silent error in at least one repeat {s['silent_any']}")
    lines.append(f"      questions whose outcome category changed between repeats {s['changed_category']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    try:
        runs = load(args.files)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(render(args.files, runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
