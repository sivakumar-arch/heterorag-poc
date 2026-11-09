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
and a pooled line over all ordered pairs, with Clopper-Pearson intervals. The pooled share that is NOT silent
again is the spontaneous conversion rate: the null reference for a catch rate measured between two runs.

    python -m benchmark_v2.silent_persistence data/floor_gptoss20b_cost_1.json data/floor_gptoss20b_cost_2.json data/floor_gptoss20b_cost_3.json

Reads result files only. Needs at least two files with the same question ids.
"""

from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path

from benchmark_v2.floor_compare import pct
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
        lines.append("  pooled over ordered pairs (the pairs share questions, so read the intervals as optimistic)")
        lines.append(f"      still silent        {pct(p['still_silent'], p['silent'])}")
        lines.append(f"      not silent (spontaneous conversion) {pct(p['silent'] - p['still_silent'], p['silent'])}")
        lines.append(f"      of which correct    {pct(p['correct'], p['silent'])}")
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
