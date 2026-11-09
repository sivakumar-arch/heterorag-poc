"""
benchmark_v2/agreement_replay.py
================================
What would a two-run agreement rule have done on runs that already exist? Exploratory, no model call, nothing
rescored. The rule: answer only if two independent runs of the same system on the same question both answer
and give the same answer; otherwise abstain.

Result files record the outcome of each run, not always the answer, so the rule is bracketed:

    both correct                      the answers agree (both equal the gold)            -> answered, correct
    one correct, one silent           the answers differ                                  -> abstain
    one answers, one abstains         no agreement                                        -> abstain
    both silent                       the two wrong answers may be equal or different     -> worst case answered and
                                                                                             silent, best case abstain
    anything else (flagged partial)   treated as not agreeing                              -> abstain

For every unordered pair of runs it prints correct, silent (best case and worst case) and abstained, against the
single-run figures of the two runs. Cost: two runs per question. The numbers are on the dev questions the
mechanisms were looked at on: they show what is worth testing on fresh questions, they do not establish it.

    python -m benchmark_v2.agreement_replay data/floor_gptoss20b_cost_1.json data/floor_gptoss20b_cost_2.json data/floor_gptoss20b_cost_3.json
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

from benchmark_v2.floor_compare import pct
from benchmark_v2.repeat_summary import group, load


def pair_counts(a: dict[str, str], b: dict[str, str]) -> dict[str, int]:
    c = {"n": len(a), "correct": 0, "silent_worst": 0, "abstained_or_disagree": 0}
    for q in a:
        ga, gb = group(a[q]), group(b[q])
        if ga == "correct" and gb == "correct":
            c["correct"] += 1
        elif ga == "silent" and gb == "silent":
            c["silent_worst"] += 1
        else:
            c["abstained_or_disagree"] += 1
    return c


def single(run: dict[str, str]) -> dict[str, int]:
    g = [group(o) for o in run.values()]
    return {"n": len(g), "correct": g.count("correct"), "silent": g.count("silent")}


def render(files: list[str], runs: list[dict[str, str]]) -> str:
    names = [Path(f).name for f in files]
    lines = [f"{len(runs)} runs, {len(runs[0])} questions; rule: answer only if two runs agree"]
    for i, s in enumerate(single(r) for r in runs):
        lines.append(f"  single run {names[i]}: correct {pct(s['correct'], s['n'])}, silent {pct(s['silent'], s['n'])}")
    tot = {"n": 0, "correct": 0, "silent_worst": 0}
    for i, j in combinations(range(len(runs)), 2):
        c = pair_counts(runs[i], runs[j])
        lines.append(f"  pair {i + 1}+{j + 1}: correct {pct(c['correct'], c['n'])}; silent at most "
                     f"{pct(c['silent_worst'], c['n'])} (best case 0); abstained or disagree {c['abstained_or_disagree']}")
        for k in tot:
            tot[k] += c[k]
    lines.append(f"  pooled over pairs: correct {pct(tot['correct'], tot['n'])}; silent at most "
                 f"{pct(tot['silent_worst'], tot['n'])} (pairs share questions, intervals optimistic)")
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
