"""
benchmark_v2/pool_counts.py
===========================
Batch 100 (build item 1): count the candidate pool of each REPEAT template on the oracle, with the oracle's
identity recorded, after excluding the questions already used (CP1, CP1b, the probe, ...).

For each template the sampler is run without a size limit (every eligible parameter set), duplicates are dropped,
the gold answer is computed and ``check_gold`` applied with the generator's thresholds, and the candidates whose
key appears in an exclusion file are removed. Reported per template: candidates, distinct, passing ``check_gold``
(with the reject reasons), excluded, and the remaining pool; then the oracle fingerprint (the generator's
``oracle_fingerprint``) and the sha256 of the oracle file. Exclusion files are question files (JSON lines,
one record per question, as written by the generator); a file whose records name another oracle fingerprint is
reported.

Exclusion key per template: the parameters without the stratum, except D05, whose sampler draws one random word
per (tag, size band), so a used (tag, band) pair is excluded whatever word it had. The 2026-11-10 pool figures
(D01 3,954, D05 2,453, D07 49, D04 5) were measured by an ad hoc script that was not saved (its D01 candidate list
held exactly 4,000 entries, a truncated list); this script is the recorded method from now on, and a difference to
those figures is reported, not reconciled by hand.

Test split guard (Batch 101): the test split is never read. An exclusion file whose name contains ``test`` is
refused before it is opened, and a file that turns out to hold a record of the test split (``"split": "test"``)
stops the script with no count printed. Exclusion rule registered 2026-11-18: every dev candidate file,
``data/questions_v2_dev*.jsonl``.

Read-only: the oracle is opened read-only (``union.open_union``, which installs the views); nothing is written unless --out is given.

    python -m benchmark_v2.pool_counts --db data/oracle_stats.sqlite --exclude data/questions_v2_dev*.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from statistics import median
from collections import Counter
from pathlib import Path

from benchmark_v2 import union
from benchmark_v2.catalogue_d import TEMPLATES
from benchmark_v2.catalogue_fc import PROBE_TEMPLATES as FC_PROBE_TEMPLATES
from benchmark_v2.catalogue_fc import TEMPLATES as FC_TEMPLATES
from benchmark_v2.catalogue_n1 import TEMPLATES as N1_TEMPLATES
from benchmark_v2.questions import H_BENCH, T_INDEPENDENT, Ctx, check_gold, needs_restriction, oracle_fingerprint

UNLIMITED = 10**9
# D05: one random word per (tag, band); the band is the stratum. Others: the parameters without the stratum.
# D05 and the N1 templates choose the word or the threshold from the tag (and a random draw), so a used
# (tag, stratum) is excluded whatever word or threshold it had.
KEY_FIELDS = {"D05": ("tag", "stratum"), "D01E": ("tag", "stratum"), "D05E": ("tag", "stratum"),
              # first-contact (Batch 104): the draw caps are per tag, so a used tag is excluded
              "S1": ("tag", "stratum"), "S2": ("tag_a",), "H2": ("tag",),
              "H1": ("badge",), "H3": ("word",), "H4": ("year", "rank"), "H4P": ("year", "rank")}
# Thresholds that are not a real restriction (decision 4 of 2026-11-26: report how often they occur).
ARTIFICIAL = {"S1": lambda p: p.get("number", 1) <= 0,     # "at least 0" (excluded since Batch 104b)
              "H3": lambda p: p.get("number", 3) <= 2}     # "under 2": reputation 1 only


def key(template_id: str, params: dict, stratum: str | None = None) -> tuple:
    fields = KEY_FIELDS.get(template_id)
    p = dict(params, stratum=stratum if stratum is not None else params.get("stratum"))
    if fields:
        return tuple((f, json.dumps(p.get(f), sort_keys=True)) for f in fields)
    return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in p.items() if k != "stratum"))


class SplitRefused(ValueError):
    """An exclusion file is, or holds, the test split."""


def refuse_test_names(files: list[str]) -> list[str]:
    """Exclusion files whose name contains ``test``; checked before any file is opened."""
    return [f for f in files if "test" in Path(f).name.lower()]


def excluded_keys(files: list[str], fingerprint: str) -> tuple[dict[str, set], list[str]]:
    named = refuse_test_names(files)
    if named:
        raise SplitRefused(f"exclusion file names a test split, not opened: {named}")
    keys: dict[str, set] = {}
    notes: list[str] = []
    for f in files:
        prints: Counter = Counter()
        n = 0
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") == "test":
                raise SplitRefused(f"{f} holds a test-split record; stopped, nothing counted")
            keys.setdefault(r["template"], set()).add(key(r["template"], r.get("params", {}), r.get("stratum")))
            prints[r.get("oracle")] += 1
            n += 1
        other = {p: c for p, c in prints.items() if p != fingerprint}
        notes.append(f"{f}: {n} questions" + (f"; OTHER ORACLE FINGERPRINT {other}" if other else ""))
    return keys, notes


def count(template, ctx: Ctx, excluded: set, t_ind: int, h_cap: int) -> dict:
    cands = template.sample(ctx, UNLIMITED)
    seen, distinct = set(), []
    for p in cands:
        k = key(template.id, p)
        if k not in seen:
            seen.add(k)
            distinct.append(p)
    reasons: Counter = Counter()
    passing, golds = [], {}
    for p in distinct:
        gold = template.gold(ctx, p)
        why = check_gold(template, p, gold, t_ind, h_cap)
        if why:
            reasons[why.split(":")[0]] += 1
        else:
            passing.append(p)
            golds[id(p)] = gold
    pool = [p for p in passing if key(template.id, p) not in excluded]
    out = {"template": template.id, "candidates": len(cands), "distinct": len(distinct),
           "passing_check_gold": len(passing), "rejected": dict(reasons),
           "excluded": len(passing) - len(pool), "pool": len(pool),
           "pool_by_stratum": dict(Counter(p.get("stratum", "all") for p in pool))}
    steps: dict[str, list[int]] = {}
    for p in pool:                                   # class F (Batch 103): step sizes and restriction share
        for sid, n in golds[id(p)].steps.items():
            steps.setdefault(sid, []).append(n)
    if steps:
        out["step_sizes"] = {sid: {"min": min(v), "median": int(median(v)), "max": max(v)}
                             for sid, v in sorted(steps.items())}
        out["needs_restriction"] = sum(bool(needs_restriction(template, golds[id(p)], h_cap)) for p in pool)
    if template.cls == "F":                          # Batch 104: per stratum, NUMBER and the realised answer size
        per: dict[str, dict[str, list]] = {}
        for p in pool:
            d = per.setdefault(p.get("stratum", "all"), {"number": [], "answer_size": []})
            v = golds[id(p)].value
            d["answer_size"].append(len(v) if isinstance(v, list) else v)
            if "number" in p:
                d["number"].append(p["number"])
        summary = lambda v: {"min": min(v), "median": median(v), "max": max(v)} if v else None
        out["per_stratum"] = {s: {k: summary(v) for k, v in d.items()} for s, d in sorted(per.items())}
        if template.id in ARTIFICIAL:
            out["artificial_threshold"] = sum(bool(ARTIFICIAL[template.id](p)) for p in pool)
    out["notes"] = dict(sorted(ctx.notes.items()))
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/oracle_stats.sqlite")
    ap.add_argument("--exclude", nargs="*", default=[], help="question files (JSON lines) whose questions are used")
    ap.add_argument("--templates", nargs="*", default=[t.id for t in TEMPLATES],
                    help="default: the REPEAT templates; D01E and D05E count the N1 supply per cell")
    ap.add_argument("--seed", type=int, default=20261116, help="only the D05 word choice depends on it")
    ap.add_argument("--t-independent", type=int, default=T_INDEPENDENT)
    ap.add_argument("--h-cap", type=int, default=H_BENCH)
    ap.add_argument("--out", help="also write the result as JSON")
    args = ap.parse_args(argv)
    path = Path(args.db)
    if not path.exists():
        print(f"oracle not found: {path}", file=sys.stderr)
        return 2
    named = refuse_test_names(args.exclude)
    if named:
        print(f"refused: exclusion file names a test split, not opened: {named}", file=sys.stderr)
        return 2
    db = union.open_union(str(path))          # read-only, with the oracle views (temporary, per connection)
    fp = oracle_fingerprint(db)
    try:
        keys, notes = excluded_keys(args.exclude, fp)
    except SplitRefused as ex:
        print(f"refused: {ex}", file=sys.stderr)
        return 2
    by_id = {t.id: t for t in list(TEMPLATES) + list(N1_TEMPLATES) + list(FC_TEMPLATES) + list(FC_PROBE_TEMPLATES)}
    unknown = [t for t in args.templates if t not in by_id]
    if unknown:
        print(f"unknown templates: {unknown}", file=sys.stderr)
        return 2
    results = [count(by_id[t], Ctx(db, args.seed, args.t_independent, args.h_cap), keys.get(t, set()),
                     args.t_independent, args.h_cap) for t in args.templates]
    out = {"oracle_path": str(path), "oracle_fingerprint": fp, "oracle_sha256": file_sha256(path),
           "thresholds": {"T": args.t_independent, "H": args.h_cap}, "seed": args.seed,
           "exclusion_files": notes, "templates": results}
    print(f"oracle {path}: fingerprint {fp}, sha256 {out['oracle_sha256']}")
    print(f"thresholds T {args.t_independent}, H {args.h_cap}; seed {args.seed} (D05 word choice only)")
    for n in notes:
        print(f"exclude {n}")
    for r in results:
        print(f"{r['template']}: candidates {r['candidates']}, distinct {r['distinct']}, passing check_gold "
              f"{r['passing_check_gold']}, excluded {r['excluded']}, POOL {r['pool']}")
        if r["rejected"]:
            print(f"    rejected by check_gold: {r['rejected']}")
        print(f"    pool by stratum: {r['pool_by_stratum']}")
        for st, d in (r.get("per_stratum") or {}).items():
            print(f"    {st}: answer size {d['answer_size']}" + (f", NUMBER {d['number']}" if d["number"] else ""))
        if "artificial_threshold" in r:
            print(f"    artificial thresholds: {r['artificial_threshold']} of {r['pool']}")
        if r.get("notes"):
            print(f"    sampler notes: {r['notes']}")
        if "step_sizes" in r:
            print(f"    step sizes (min/median/max): "
                  + ", ".join(f"{k} {v['min']}/{v['median']}/{v['max']}" for k, v in r["step_sizes"].items())
                  + f"; exact only through a bind-planner restriction: {r['needs_restriction']} of {r['pool']}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
