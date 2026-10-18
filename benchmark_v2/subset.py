"""
benchmark_v2/subset.py
======================
Derives a question file for a run from a generated one, without changing any question.

* ``--exclude FILE`` (repeatable) drops every question whose template and text, or whose template
  and parameter values, also occur in FILE (the same instance in another paraphrase is the same
  question). This keeps questions used for tuning (the baseline audit and planner pilots) out of a
  comparison. Text is compared after collapsing whitespace.
* ``--id-prefix P`` is put in front of every question id, so that two generated files, which both
  number their questions D01-001, D01-002, ..., cannot be confused in a results file.
* ``--limit N`` keeps the first N questions of each template that remain, in file order.

    python -m benchmark_v2.subset --questions data/questions_v2_dev_classD_cp1.jsonl \
        --exclude data/questions_v2_dev_classD_probe.jsonl --id-prefix CP1- \
        --out data/questions_v2_dev_classD_cp1.eligible.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _key(rec: dict) -> tuple[str, str]:
    return rec["template"], " ".join(rec["text"].split())


def _pkey(rec: dict) -> tuple[str, str, str]:
    """Template and parameter values: the same instance worded with another paraphrase has the same key."""
    return "params", rec["template"], json.dumps(rec.get("params"), sort_keys=True, default=str)


def read(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def subset(recs: list[dict], excluded: set[tuple[str, str]], prefix: str = "", limit: int = 0):
    """(kept records, number dropped per template as excluded, ids seen twice)"""
    kept, dropped, per = [], Counter(), defaultdict(int)
    for r in recs:
        if _key(r) in excluded or (r.get("params") is not None and _pkey(r) in excluded):
            dropped[r["template"]] += 1
            continue
        if limit and per[r["template"]] >= limit:
            continue
        per[r["template"]] += 1
        kept.append(dict(r, question_id=prefix + r["question_id"]))
    ids = Counter(r["question_id"] for r in kept)
    return kept, dropped, sorted(i for i, n in ids.items() if n > 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--id-prefix", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    used = [r for f in args.exclude for r in read(f)]
    excluded = {_key(r) for r in used} | {_pkey(r) for r in used if r.get("params") is not None}
    recs = read(args.questions)
    kept, dropped, dup = subset(recs, excluded, args.id_prefix, args.limit)
    if dup:
        print(f"duplicate question ids: {dup}", file=sys.stderr)
        return 2
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    by = Counter(r["template"] for r in kept)
    print(f"{len(recs)} read, {sum(dropped.values())} dropped as already used {dict(dropped)}, "
          f"{len(kept)} written to {args.out}: {dict(sorted(by.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
