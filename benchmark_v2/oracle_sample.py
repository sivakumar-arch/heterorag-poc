"""Select the oracle-check sample from a generated question file (Batch 106).

The pre-registration hand-checks 78 questions against reference queries before any model run (section 3; the
split of the handoff of 2026-11-15, section 9 item 7; the H1 and H4 constraints of decision 2026-11-16):

- first-contact, 8 per shape (48): H1 to H4 checked by a second outside person (32), S1 and S2 by Seema
  Aarella (16);
  - H4: two questions per usable year, covering every threshold rank and every wording variant present at
    least once;
  - H1: five over-cap questions from five different years, and three within-cap questions;
  - other shapes: spread as evenly as possible over their strata (ranks for S1 and H3), Claude's choice; which
    strata get an extra question when the count does not divide evenly is decided by the seeded stream, not by
    name order (decision 4 of 2026-11-28, Batch 107), and the order is printed;
- N1, 10: five with an empty final answer and five with an empty intermediate step, checked by Siva;
- REPEAT, 20: five each from D01, D05 and D07 and all five of D04, checked by Siva; spread over strata.

The selection is seeded and the seed is fixed before generation (``SEED``). Each shape has its own random
generator, derived from the seed and the shape, so adding or removing a shape never changes another shape's
selection. A shape with fewer questions than its quota gives all it has, and the shortfall is reported; nothing
is topped up from another shape. The script reads the question file only: it runs no query and touches no store.

Usage:
    python -m benchmark_v2.oracle_sample --questions FILE --out SAMPLE.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261128          # fixed before generation (Batch 106); recorded in the v1.8 queue

OUTSIDE_2 = "second outside person"
SEEMA = "Seema Aarella"
SIVA = "Siva"

# shape -> (quota, checker). N1 is the pair of templates D01E and D05E, selected by cell (``empty_at``).
QUOTAS = {"H1": (8, OUTSIDE_2), "H2": (8, OUTSIDE_2), "H3": (8, OUTSIDE_2), "H4": (8, OUTSIDE_2),
          "S1": (8, SEEMA), "S2": (8, SEEMA),
          "N1": (10, SIVA),
          "D01": (5, SIVA), "D05": (5, SIVA), "D07": (5, SIVA), "D04": (5, SIVA)}
N1_TEMPLATES = ("D01E", "D05E")
H1_OVER, H1_WITHIN = 5, 3
N1_PER_CELL = 5


class Refused(Exception):
    pass


def shape_of(rec: dict) -> str:
    return "N1" if rec.get("template") in N1_TEMPLATES else rec.get("template", "")


def rng_for(seed: int, shape: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{seed}:{shape}".encode()).hexdigest()[:8], 16))


def _shuffled(recs: list[dict], rng: random.Random) -> list[dict]:
    out = sorted(recs, key=lambda r: r["question_id"])
    rng.shuffle(out)
    return out


def spread(recs: list[dict], n: int, rng: random.Random, by: str = "stratum",
           orders: dict | None = None, label: str = "") -> list[dict]:
    """n questions, round-robin over the values of ``by``, each value's queue in seeded order. The order of the
    values, which decides which of them get an extra question when n does not divide evenly, is drawn from the same
    seeded stream (Batch 107; before it, name order) and recorded in ``orders[label]``."""
    queues: dict[str, list[dict]] = defaultdict(list)
    for r in _shuffled(recs, rng):
        queues[str(r.get(by, "all"))].append(r)
    out: list[dict] = []
    names = sorted(queues)
    rng.shuffle(names)
    if orders is not None and label:
        orders[label] = list(names)
    while len(out) < n and any(queues.values()):
        for k in names:
            if queues[k] and len(out) < n:
                out.append(queues[k].pop(0))
    return out


def select_h1(recs: list[dict], rng: random.Random, notes: list[str]) -> list[dict]:
    over = [r for r in _shuffled(recs, rng) if r.get("stratum") == "over-cap"]
    within = [r for r in _shuffled(recs, rng) if r.get("stratum") == "within-cap"]
    chosen, years = [], set()
    for r in over:
        y = r["params"].get("year")
        if y not in years and len(chosen) < H1_OVER:
            chosen.append(r)
            years.add(y)
    if len(chosen) < H1_OVER:
        notes.append(f"H1: {len(chosen)} over-cap questions from distinct years, {H1_OVER} wanted")
    w = within[:H1_WITHIN]
    if len(w) < H1_WITHIN:
        notes.append(f"H1: {len(w)} within-cap questions, {H1_WITHIN} wanted")
    return chosen + w


def _covers(combo: tuple[dict, ...], ranks: set, variants: set) -> tuple[int, int]:
    return (len({r["params"].get("rank") for r in combo} & ranks),
            len({r["params"].get("variant") for r in combo} & variants))


def select_h4(recs: list[dict], n: int, rng: random.Random, notes: list[str]) -> list[dict]:
    """Two per year; among all such choices, those covering every rank and every variant present; one of them by
    seeded draw. If none covers all, the best coverage is taken and the gap reported."""
    by_year: dict = defaultdict(list)
    for r in sorted(recs, key=lambda r: r["question_id"]):
        by_year[r["params"].get("year")].append(r)
    per_year = max(1, n // max(1, len(by_year))) if by_year else 0
    per_year = min(per_year, 2)
    options = []
    for y in sorted(by_year):
        k = min(per_year, len(by_year[y]))
        if k < per_year:
            notes.append(f"H4: year {y} has {len(by_year[y])} question(s), {per_year} wanted")
        options.append(list(itertools.combinations(by_year[y], k)))
    ranks = {r["params"].get("rank") for r in recs}
    variants = {r["params"].get("variant") for r in recs}
    combos = [sum(c, ()) for c in itertools.product(*options)] if options else []
    if not combos:
        return []
    scored = [(_covers(c, ranks, variants), c) for c in combos]
    best = max(s for s, _ in scored)
    good = [c for s, c in scored if s == best]
    if best != (len(ranks), len(variants)):
        notes.append(f"H4: no choice covers every rank and variant; best covers {best[0]} of {len(ranks)} ranks "
                     f"and {best[1]} of {len(variants)} variants")
    return list(rng.choice(good))


def select_n1(recs: list[dict], rng: random.Random, notes: list[str], orders: dict | None = None) -> list[dict]:
    out = []
    for cell in ("final", "intermediate"):
        cell_recs = [r for r in recs if (r.get("empty_at") == "final") == (cell == "final")]
        got = spread(cell_recs, N1_PER_CELL, rng, by="template", orders=orders, label=f"N1 {cell}")
        if len(got) < N1_PER_CELL:
            notes.append(f"N1 {cell}: {len(got)} questions, {N1_PER_CELL} wanted")
        out += got
    return out


def check_input(records: list[dict]) -> None:
    if any(r.get("split") == "test" for r in records):
        raise Refused("the file holds test-split questions")
    fps = {r.get("oracle") for r in records}
    if len(fps) > 1:
        raise Refused(f"the file mixes oracle fingerprints: {sorted(map(str, fps))}")
    ids = Counter(r.get("question_id") for r in records)
    dup = sorted(q for q, k in ids.items() if k > 1)
    if dup:
        raise Refused(f"question ids appear more than once: {dup[:5]}")


def select(records: list[dict], seed: int = SEED, orders: dict | None = None) -> tuple[list[dict], list[str]]:
    """The oracle-check sample: records with ``oracle_check`` (checker, shape, seed) added, and the notes. ``orders``,
    if given, receives the seeded order of the strata per shape (the tie-break)."""
    check_input(records)
    by_shape: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_shape[shape_of(r)].append(r)
    chosen: list[dict] = []
    notes: list[str] = []
    for shape, (quota, checker) in QUOTAS.items():
        recs, rng = by_shape.get(shape, []), rng_for(seed, shape)
        if shape == "H1":
            got = select_h1(recs, rng, notes)
        elif shape == "H4":
            got = select_h4(recs, quota, rng, notes)
        elif shape == "N1":
            got = select_n1(recs, rng, notes, orders)
        else:
            got = spread(recs, quota, rng, orders=orders, label=shape)
        if len(got) < quota:
            notes.append(f"{shape}: {len(got)} of {quota} selected (shortfall {quota - len(got)}, not topped up)")
        for r in got:
            chosen.append({**r, "oracle_check": {"checker": checker, "shape": shape, "seed": seed}})
    unknown = sorted(set(by_shape) - set(QUOTAS))
    if unknown:
        notes.append(f"shapes not in the oracle sample: {unknown}")
    return chosen, notes


def summary(chosen: list[dict]) -> list[str]:
    lines = []
    by_checker = Counter(r["oracle_check"]["checker"] for r in chosen)
    lines.append(f"selected {len(chosen)}: " + ", ".join(f"{c} {k}" for c, k in sorted(by_checker.items())))
    by_shape: dict[str, list[dict]] = defaultdict(list)
    for r in chosen:
        by_shape[r["oracle_check"]["shape"]].append(r)
    for shape in QUOTAS:
        recs = by_shape.get(shape, [])
        strata = dict(sorted(Counter(r.get("stratum", "all") for r in recs).items()))
        extra = ""
        if shape == "H4":
            extra = (f"; years {dict(sorted(Counter(r['params'].get('year') for r in recs).items()))}, ranks "
                     f"{sorted({r['params'].get('rank') for r in recs})}, variants "
                     f"{sorted({r['params'].get('variant') for r in recs})}")
        if shape == "H1":
            over = [r for r in recs if r.get("stratum") == "over-cap"]
            extra = f"; over-cap years {sorted(r['params'].get('year') for r in over)}"
        lines.append(f"  {shape}: {len(recs)} {strata}{extra}")
    return lines


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--questions", required=True, help="the generated question file (JSON lines)")
    ap.add_argument("--out", required=True, help="the sample, JSON lines, one record per selected question")
    ap.add_argument("--seed", type=int, default=SEED, help=f"fixed before generation (default {SEED})")
    args = ap.parse_args(argv)
    if "test" in Path(args.questions).name.lower():
        print(f"refused: the file name names a test split: {args.questions}", file=sys.stderr)
        return 2
    records = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    orders: dict = {}
    try:
        chosen, notes = select(records, args.seed, orders)
    except Refused as ex:
        print(f"refused: {ex}", file=sys.stderr)
        return 2
    with open(args.out, "w") as f:
        for r in chosen:
            f.write(json.dumps(r) + "\n")
    print(f"questions {args.questions}: {len(records)} records, sha256 {file_sha256(args.questions)}")
    print(f"seed {args.seed}; oracle {records[0].get('oracle') if records else '-'}")
    for line in summary(chosen):
        print(line)
    for shape, order in orders.items():
        if len(order) > 1:
            print(f"  stratum order (seeded tie-break) {shape}: {order}")
    for n in notes:
        print(f"note: {n}")
    print(f"written {args.out}: sha256 {file_sha256(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
