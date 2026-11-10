"""
benchmark_v2/silent_persistence.py
==================================
How stable are silent errors from one run to the next, with no change to the system? A before and after
comparison of any new check must be read against this: if a question with a silent error in one run is not
silent in another run of the SAME system, a check that "removes" such an error has to do better than that.

For the repeats of one system on the same questions it prints, for every ordered pair of runs (i, j):
    silent in run i       the questions whose outcome in run i is wrong or partial_violated
    still silent in j     of those, the questions that are silent again in run j
    became correct in j   ... correct in run j
    became abstained or flagged in j
and a pooled line over all ordered pairs. The pooled share that is NOT silent again is the spontaneous
conversion rate: the null reference for a catch rate measured between two runs.

The ordered pairs of one question are not independent (the same question appears in every pair, and an error
may be repeated in both directions), so the pooled shares carry a percentile interval from a QUESTION-CLUSTERED
bootstrap (4,000 resamples, seed 20261025; whole questions are resampled with all their pairs). The pair counts
are shown as counts only, without an interval.

    python -m benchmark_v2.silent_persistence data/floor_gptoss20b_cost_1.json data/floor_gptoss20b_cost_2.json data/floor_gptoss20b_cost_3.json

Reads result files only. Needs at least two files with the same question ids.
"""

from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path

import random

from benchmark_v2.cp1_report import _quant
from benchmark_v2.floor_compare import BOOT, SEED
from benchmark_v2.repeat_summary import group, load


def transitions(runs: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, int]]:
    out: dict[tuple[int, int], dict[str, int]] = {}
    for i, j in permutations(range(len(runs)), 2):
        c = {"silent": 0, "still_silent": 0, "correct": 0, "other": 0}
        for q, o in runs[i].items():
            if group(o) != "silent":
                continue
            c["silent"] += 1
            g = group(runs[j][q])
            c["still_silent" if g == "silent" else "correct" if g == "correct" else "other"] += 1
        out[(i, j)] = c
    return out


def pooled(tr: dict[tuple[int, int], dict[str, int]]) -> dict[str, int]:
    tot = {"silent": 0, "still_silent": 0, "correct": 0, "other": 0}
    for c in tr.values():
        for k in tot:
            tot[k] += c[k]
    return tot


def per_question(runs: list[dict[str, str]]) -> dict[str, tuple[int, int, int]]:
    """question -> (ordered pairs in which it is silent in the first run, of those not silent in the second,
    of those correct in the second)."""
    out = {}
    for q in runs[0]:
        s = ns = c = 0
        for i, j in permutations(range(len(runs)), 2):
            if group(runs[i][q]) != "silent":
                continue
            s += 1
            g = group(runs[j][q])
            ns += g != "silent"
            c += g == "correct"
        out[q] = (s, ns, c)
    return out


def clustered(runs: list[dict[str, str]], which: int) -> tuple[float, float, float, int]:
    """Pooled share (in percent) of the silent first-run cases that are not silent again (which = 1) or are correct
    (which = 2), with a question-clustered percentile interval, and the number of questions that contribute."""
    pq = per_question(runs)
    qids = sorted(q for q in pq if pq[q][0])
    if not qids:
        return float("nan"), float("nan"), float("nan"), 0
    ratio = lambda ids: 100.0 * sum(pq[q][which] for q in ids) / sum(pq[q][0] for q in ids)
    rng = random.Random(SEED)
    # resample from ALL questions, so that the number of questions with a silent error varies as it would in a new sample
    allq = sorted(pq)
    vals = []
    for _ in range(BOOT):
        ids = [allq[rng.randrange(len(allq))] for _ in allq]
        if sum(pq[q][0] for q in ids):
            vals.append(ratio(ids))
    return ratio(allq), _quant(vals, 0.025), _quant(vals, 0.975), len(qids)


def render(files: list[str], runs: list[dict[str, str]]) -> str:
    names = [Path(f).name for f in files]
    tr = transitions(runs)
    lines = [f"{len(runs)} runs, {len(runs[0])} questions"]
    for (i, j), c in tr.items():
        if c["silent"] == 0:
            lines.append(f"  {names[i]} -> {names[j]}: no silent errors in the first run")
            continue
        lines.append(f"  {names[i]} -> {names[j]}: silent in first {c['silent']}; still silent {c['still_silent']}, "
                     f"became correct {c['correct']}, became abstained or flagged {c['other']}")
    p = pooled(tr)
    if p["silent"]:
        _, lo, hi, nq = clustered(runs, 1)
        m1 = 100.0 * (p["silent"] - p["still_silent"]) / p["silent"]
        m2 = 100.0 * p["correct"] / p["silent"]
        _, lo2, hi2, _ = clustered(runs, 2)
        lines.append(f"  pooled over {p['silent']} ordered pairs from {nq} questions with a silent error "
                     f"(intervals: question-clustered bootstrap, 4,000 resamples)")
        lines.append(f"      still silent        {100.0 - m1:.1f}% ({p['still_silent']}/{p['silent']} pairs)")
        lines.append(f"      not silent (spontaneous conversion) {m1:.1f}% [{lo:.1f}, {hi:.1f}] ({p['silent'] - p['still_silent']}/{p['silent']} pairs)")
        lines.append(f"      of which correct    {m2:.1f}% [{lo2:.1f}, {hi2:.1f}] ({p['correct']}/{p['silent']} pairs)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    if len(args.files) < 2:
        print("error: at least two result files are needed", file=sys.stderr)
        return 2
    try:
        runs = load(args.files)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(render(args.files, runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
