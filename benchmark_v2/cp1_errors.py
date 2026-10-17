"""
benchmark_v2/cp1_errors.py
==========================
Where do the failures of one pipeline system sit? Reads the repeat files of a ``run_slice`` system
(CP1) and lists, per question, what happened in each repeat, so that systematic failures (the same
question in every repeat) are told apart from intermittent ones (sampling variance), and a silent
error under an EXACT label (``wrong``) from a violated label (``partial_violated``).

    python -m benchmark_v2.cp1_errors data/cp1_cost_1.json data/cp1_cost_2.json data/cp1_cost_3.json

Letters per repeat: C correct, W wrong (answer labelled EXACT and wrong), V label violated,
P flagged partial, A abstained.
This is analysis only: it reads result files and changes none.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

LETTER = {"correct": "C", "wrong": "W", "partial_violated": "V", "partial_valid": "P",
          "abstain_expected": "A", "abstain_unexpected": "A"}


def load(paths: list[str]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        for r in json.loads(Path(p).read_text())["rows"]:
            by[r["question_id"]].append(r)
    return by


def render(by: dict[str, list[dict]]) -> str:
    n = max(len(v) for v in by.values())
    lines = [f"{'question':12s} {'template':8s} {'stratum':16s} repeats  plan_equal  guarantee(s)"]
    kinds, silent_q = Counter(), []
    for qid in sorted(by):
        rs = by[qid]
        letters = "".join(LETTER.get(r["outcome"], "?") for r in rs)
        if letters == "C" * len(rs):
            continue
        kinds.update(c for c in letters if c != "C")
        if any(c in letters for c in "WV"):
            silent_q.append((qid, letters))
        lines.append(f"{qid:12s} {rs[0]['template']:8s} {str(rs[0].get('stratum') or '?'):16s} {letters:8s} "
                     f"{''.join('=' if r.get('plan_equal') else 'x' for r in rs):10s}  "
                     f"{','.join(sorted({str(r.get('guarantee')) for r in rs}))}")
    total = sum(len(v) for v in by.values())
    lines += ["", f"runs: {total} over {len(by)} questions ({n} repeats)",
              "runs that were not correct, by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())),
              f"questions with a silent error (W or V) in at least one repeat: {len(silent_q)}",
              f"  in every repeat (systematic): {sum(1 for _, l in silent_q if all(c in 'WV' for c in l))}",
              f"  in some repeats (intermittent): {sum(1 for _, l in silent_q if not all(c in 'WV' for c in l))}",
              "plan_equal: = same structure as the annotated plan, x different (not a fault by itself)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    print(render(load(args.files)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
