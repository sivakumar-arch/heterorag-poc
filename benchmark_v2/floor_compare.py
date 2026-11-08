"""
benchmark_v2/floor_compare.py
=============================
One rung of the capability-floor ladder (gate G0, rule L): the pipeline against the two free-form agent
arms on the same questions, with the rule-L1 verdict and the pre-specified count of replies whose generated
tokens never reached the harness. All numbers come from the result files; nothing is computed by hand.

    python -m benchmark_v2.floor_compare \
        --pipeline data/floor_mistral_small32_24b_cost_1.json \
        --python   data/floor_mistral_small32_24b_python_text_1.json \
        --plain    data/floor_mistral_small32_24b_plain_text_1.json

Method (fixed before the Mistral agent arms were run, 2026-10-25 and 2026-11-07)
  outcomes     as in cp1_report: correct, silent error (wrong or label violated), flagged, abstained;
  intervals    Clopper-Pearson (exact) for every rate;
  paired       difference in percentage points over questions, percentile bootstrap, 4,000 resamples, seed 20261025;
  McNemar      exact two-sided, on the discordant pairs of `correct`;
  better arm   the agent arm with more correct answers;
  rule L1      the pipeline is at least 80 percent correct AND exceeds the better arm by at least 15 points
               with a paired interval that excludes zero. A pipeline below 50 percent correct is a floor effect
               and the result is inconclusive.
  hidden       a question whose last reply had 20 or more generated tokens (`reply_tokens`) and no text and no
               call; reported as a count and a share of the questions of the arm; above 10 percent in an arm,
               a raw replay of a stated sample is run and reported as a limitation (not added to the scores).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from benchmark_v2.cp1_report import CATEGORIES, System, _mean, _quant, category, exact_interval

BOOT, SEED = 4000, 20261025
HIDDEN_MIN_TOKENS = 20


def pct(k: int, n: int) -> str:
    lo, hi = exact_interval(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] ({k}/{n})"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for discordant counts b (first only) and c (second only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(a: dict, b: dict, qids: list[str], rng: random.Random) -> tuple[float, float, float]:
    """a minus b, percentage points, over the question-level values in the dicts a and b."""
    stat = lambda ids: 100.0 * _mean([a[q] - b[q] for q in ids])
    vals = [stat([qids[rng.randrange(len(qids))] for _ in qids]) for _ in range(BOOT)]
    return stat(qids), _quant(vals, 0.025), _quant(vals, 0.975)


def hidden_replies(rows: list[dict]) -> tuple[int, int]:
    """(questions with such a reply as the LAST transcript entry, questions with one anywhere)."""
    last = anywhere = 0
    for r in rows:
        tr = r.get("transcript") or []
        empty = [t.get("tool") == "_text" and not str(t.get("text") or "").strip()
                 and int(t.get("reply_tokens") or 0) >= HIDDEN_MIN_TOKENS for t in tr]
        anywhere += any(empty)
        last += bool(empty) and empty[-1]
    return last, anywhere


def hidden_list(rows: list[dict]) -> list[dict]:
    """The questions whose LAST transcript entry is a hidden reply, with where in the run it happened:
    `calls_before` tool calls had been made, and `replies` replies without a call had been seen (the harness
    nudges once, so 2 means the run ended after the nudge). A hidden reply with calls_before 0 and replies 1
    or 2 is a reply to the first request, which a raw replay can reproduce exactly."""
    out = []
    for r in rows:
        tr = r.get("transcript") or []
        if not tr:
            continue
        t = tr[-1]
        if t.get("tool") == "_text" and not str(t.get("text") or "").strip() \
                and int(t.get("reply_tokens") or 0) >= HIDDEN_MIN_TOKENS:
            out.append({"question_id": r["question_id"], "template": r.get("template"),
                        "reply_tokens": int(t["reply_tokens"]),
                        "calls_before": sum(1 for x in tr if not str(x.get("tool", "_")).startswith("_")),
                        "replies": sum(1 for x in tr if x.get("tool") == "_text")})
    return out


def verdict_l1(pipe_k: int, n: int, gap: tuple[float, float, float]) -> list[str]:
    lo_p = 100 * exact_interval(pipe_k, n)[0]
    level = 100 * pipe_k / n
    out = []
    if level < 50:
        return [f"pipeline {level:.1f}% correct is below 50%: floor effect, inconclusive (rule L)."]
    out.append(f"pipeline level: {level:.1f}% correct (lower limit {lo_p:.1f}); at least 80%: "
               f"{'yes' if level >= 80 else 'NO'}")
    point, lo, hi = gap
    out.append(f"gap to the better arm: {point:+.1f} points [{lo:+.1f}, {hi:+.1f}]; at least 15 with the interval "
               f"excluding zero: {'yes' if (point >= 15 and lo > 0) else 'NO'}")
    out.append("L1: " + ("MET" if level >= 80 and point >= 15 and lo > 0 else "NOT MET"))
    return out


def compare(pipe: str, py: str, plain: str) -> str:
    systems = {"pipeline": System("pipeline", [pipe]), "python": System("python", [py]),
               "plain": System("plain", [plain])}
    qids = systems["pipeline"].qids
    for s in systems.values():
        if s.qids != qids:
            raise ValueError(f"{s.name}: not the same questions as the pipeline "
                             f"({len(set(qids) ^ set(s.qids))} differ)")
    n = len(qids)
    k = {name: {c: round(sum(s.v[q][c] for q in qids)) for c in CATEGORIES} for name, s in systems.items()}
    out = [f"{n} questions, one repeat per system.", "", "| system | correct | silent error | flagged | abstained |",
           "|---|---|---|---|---|"]
    for name in systems:
        out.append(f"| {name} | " + " | ".join(pct(k[name][c], n) for c in CATEGORIES) + " |")
    better = max(("python", "plain"), key=lambda a: (k[a]["correct"], -k[a]["silent"]))
    rng = random.Random(SEED)
    out += ["", f"Better agent arm (more correct answers): {better}", ""]
    gap_better = None
    for a in ("python", "plain"):
        cor = paired_bootstrap({q: systems["pipeline"].v[q]["correct"] for q in qids},
                               {q: systems[a].v[q]["correct"] for q in qids}, qids, rng)
        sil = paired_bootstrap({q: systems["pipeline"].v[q]["silent"] for q in qids},
                               {q: systems[a].v[q]["silent"] for q in qids}, qids, rng)
        b = sum(systems["pipeline"].v[q]["correct"] > 0.5 and systems[a].v[q]["correct"] < 0.5 for q in qids)
        c = sum(systems[a].v[q]["correct"] > 0.5 and systems["pipeline"].v[q]["correct"] < 0.5 for q in qids)
        out.append(f"pipeline minus {a}: correct {cor[0]:+.1f} [{cor[1]:+.1f}, {cor[2]:+.1f}] points; "
                   f"silent {sil[0]:+.1f} [{sil[1]:+.1f}, {sil[2]:+.1f}]; McNemar on correct: pipeline only {b}, "
                   f"{a} only {c}, exact p = {mcnemar_exact(b, c):.4g}")
        if a == better:
            gap_better = cor
    out += ["", "Rule L1 against the better arm:"] + ["  " + line for line in verdict_l1(k["pipeline"]["correct"], n, gap_better)]
    out += ["", "Replies whose generated tokens never reached the harness (reply_tokens >= "
            f"{HIDDEN_MIN_TOKENS}, no text, no call):"]
    for name, path in (("python", py), ("plain", plain)):
        rows = json.loads(Path(path).read_text())["rows"]
        last, anywhere = hidden_replies(rows)
        trig = "ABOVE the 10% trigger: raw replay of a stated sample required" if 100 * last / n > 10 else "below the 10% trigger"
        out.append(f"  {name}: last reply {last}/{n} = {100 * last / n:.1f}% ({trig}); anywhere in the run {anywhere}/{n}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("pipeline", "python", "plain"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--list-hidden", action="store_true",
                    help="also list the questions whose last reply is hidden, per agent arm")
    args = ap.parse_args(argv)
    try:
        print(compare(args.pipeline, args.python, args.plain))
        if args.list_hidden:
            for name, path in (("python", args.python), ("plain", args.plain)):
                items = hidden_list(json.loads(Path(path).read_text())["rows"])
                first = [i for i in items if i["calls_before"] == 0]
                print(f"\nHidden last replies, {name} arm: {len(items)} questions, {len(first)} with no tool call "
                      f"before it (reply to the first request, reproducible by a raw replay)")
                for i in items:
                    print(f"  {i['question_id']}  {i['template']}  reply_tokens={i['reply_tokens']}  "
                          f"calls_before={i['calls_before']}  replies_without_call={i['replies']}")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
