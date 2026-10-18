"""
benchmark_v2/questions.py
=========================
Question templates, gold answers and the generator for Benchmark v2.

A *template* is a natural-language question with placeholders (three paraphrases), a sampler
that draws parameter values in strata, a gold function that computes the answer over the
virtual union V (``benchmark_v2.union``), and a plan annotation (which stores are needed, in
which dependency order). An *instance* is a template with concrete parameters. The generator
draws instances, computes gold, checks size and class rules, and writes JSON lines.

Rules enforced here (spec decisions D12, D13, D15):

* identifier-set answers have 1 to MAX_SET entities;
* ranked answers have exactly k entities and no tie at the k-th place;
* class P requires every independent sub-result to have at most T entities;
* class D (decision D12) requires an independent formulation to need a sub-result above T, or
  to be impossible because a store lacks the predicate (a *dependent* instance). Executing it
  exactly must still be possible: a set that is injected into a store (a bind) has at most H
  entities, the configurable hard cap on one retrieve step (decision D25, benchmark value 10,000);
* instances are unique per template, and drawn round-robin across strata;
* generation is deterministic for a given oracle file and seed.

Templates are assigned to the development or test split in their definition, before any
system is run. The generator writes only the development split unless ``--include-test`` is
given, and never prints gold values, only sizes.

Like the oracle, this module does not import the ``heterorag`` package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import union

T_INDEPENDENT = 200      # decision D12: largest independent sub-result for class P
H_BENCH = 10_000         # hard cap on the rows of one retrieve step (benchmark setting)
MAX_SET = 50             # decision D15: largest identifier-set answer
GENERATOR_VERSION = "2.1"      # 2.1: D01 paraphrase 3 says "questions tagged" (CP1 diagnosis: "regression questions" was ambiguous)

# English stop words of the Elasticsearch "_english_" list that are 5 or more letters long
# (shorter words are excluded by the length rule anyway).
STOPWORDS = frozenset({"their", "there", "these"})


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Bind:
    input: str                       # id of the step whose output is injected
    entity: str                      # entity type of the injected identifiers
    param: str = ""


@dataclass(frozen=True)
class Step:
    """One step of the annotated gold plan. The fields are identical to the system's
    ``heterorag.federation.plan.Step`` (Interface Specification v2.1, decision D18); a test
    asserts it. This module does not import the ``heterorag`` package, so the schema is
    repeated here with plain strings."""
    id: str
    op: str                          # retrieve | intersect | union | subtract | count
    store: str | None = None         # sql | graph | doc (retrieve only)
    mode: str | None = None          # filter | search (doc store only)
    question: str = ""               # sub-question template (retrieve only)
    inputs: tuple[str, ...] = ()     # ids of steps whose output this step consumes
    bind: Bind | None = None
    output: str = ""                 # entity type produced ("number" for scalars)
    kind: str = "set"                # set | ranked | scalar
    k: int | None = None             # filled per instance for ranked steps
    must_exist: bool = False


@dataclass
class Gold:
    value: Any                       # sorted list | ranked list | number
    intermediates: dict[str, int] = field(default_factory=dict)   # store -> size of its independent sub-result
    # Class D. ``independent`` is False when no formulation without a bind exists (a store lacks
    # the predicate); ``bound`` then holds the size of every set that has to be injected.
    bound: dict[str, int] = field(default_factory=dict)
    independent: bool = True


@dataclass(frozen=True)
class Template:
    id: str
    cls: str                         # S | P | D | A | N
    split: str                       # dev | test
    answer_type: str                 # ids | ranked | scalar
    entity: str                      # entity type of the answer (user, question, answer, tag, ...)
    paraphrases: tuple[str, str, str]
    plan: tuple[Step, ...]
    sample: Callable[["Ctx", int], list[dict]]
    gold: Callable[["Ctx", dict], Gold]
    k_param: str | None = None       # name of the parameter holding k for ranked answers

    @property
    def services(self) -> list[str]:
        return sorted({s.store for s in self.plan if s.store})


class Ctx:
    """Read-only access to V plus caches shared by samplers and gold functions."""

    def __init__(self, db: sqlite3.Connection, seed: int, t_independent: int = T_INDEPENDENT,
                 h_cap: int = H_BENCH):
        self.db = db
        self.rng = random.Random(seed)
        self.t_independent = t_independent       # class P / D boundary
        self.h_cap = h_cap                       # hard cap on one retrieve step
        self._cache: dict[str, Any] = {}

    def rows(self, sql: str, *args) -> list[tuple]:
        return self.db.execute(sql, args).fetchall()

    def col(self, sql: str, *args) -> list:
        return [r[0] for r in self.rows(sql, *args)]

    def one(self, sql: str, *args):
        r = self.db.execute(sql, args).fetchone()
        return None if r is None else r[0]

    def cached(self, key: str, fn: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # ---- shared derived structures
    def title_postings(self) -> dict[str, list[int]]:
        """word -> sorted question ids whose title contains the word (whole word).
        Only alphabetic words of 5 or more letters that are not stop words."""
        def build():
            post: dict[str, set[int]] = {}
            for pid, title in self.db.execute("SELECT post_id, title FROM v_question_docs"):
                for w in set(union.tokens(title)):
                    if w.isalpha() and len(w) >= 5 and w not in STOPWORDS:
                        post.setdefault(w, set()).add(pid)
            return {w: sorted(s) for w, s in post.items()}
        return self.cached("title_postings", build)

    def title_words(self) -> dict[int, frozenset[str]]:
        """question id -> the indexed words of its title (inverse of ``title_postings``)."""
        def build():
            inv: dict[int, set[str]] = {}
            for w, ids in self.title_postings().items():
                for i in ids:
                    inv.setdefault(i, set()).add(w)
            return {i: frozenset(ws) for i, ws in inv.items()}
        return self.cached("title_words", build)

    def tag_members(self) -> dict[str, list[int]]:
        """tag -> sorted question ids, for every tag (one query, cached)."""
        def build():
            out: dict[str, list[int]] = {}
            for qid, tag in self.db.execute("SELECT question_id, tag FROM v_tagged ORDER BY question_id"):
                out.setdefault(tag, []).append(qid)
            return out
        return self.cached("tag_members", build)

    def tag_sizes(self) -> dict[str, int]:
        return self.cached("tag_sizes", lambda: dict(self.rows(
            "SELECT tag, count(*) FROM v_tagged GROUP BY tag")))

    def tagged(self, tag: str) -> list[int]:
        return self.col("SELECT question_id FROM v_tagged WHERE tag = ? ORDER BY question_id", tag)

    def tags_of(self, qid: int) -> list[str]:
        return self.col("SELECT tag FROM v_tagged WHERE question_id = ? ORDER BY tag", qid)


def pick(items: list, n: int, rng: random.Random) -> list:
    return rng.sample(items, min(len(items), n))


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

@dataclass
class Report:
    template: str
    candidates: int = 0
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    strata: dict[str, int] = field(default_factory=dict)
    sizes: list[int] = field(default_factory=list)

    def reject(self, why: str) -> None:
        self.rejected[why] = self.rejected.get(why, 0) + 1


def _canonical(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def check_gold(t: Template, params: dict, gold: Gold, t_ind: int = T_INDEPENDENT,
               h_cap: int = H_BENCH) -> str | None:
    """Return the reason an instance is unacceptable, or None."""
    v = gold.value
    if t.answer_type == "ids":
        if not isinstance(v, list) or not (1 <= len(v) <= MAX_SET):
            n = 0 if not isinstance(v, list) else len(v)
            return "set size 0, the answer is empty" if n == 0 else f"set size above {MAX_SET}"
        if len(set(v)) != len(v):
            return "duplicate identifiers in gold"
    elif t.answer_type == "ranked":
        k = params.get(t.k_param or "k")
        if v is None:
            return "tie at the k-th place or fewer than k+1 candidates"
        if not isinstance(v, list) or len(v) != k or len(set(v)) != k:
            return "ranked list is not k distinct entities"
    elif t.answer_type == "scalar":
        if v is None:
            return "scalar gold is missing"
    if t.cls == "P":
        if not gold.intermediates:
            return "class P needs recorded intermediate cardinalities"
        if max(gold.intermediates.values()) > t_ind:
            return f"independent sub-result above T={t_ind}"
    if t.cls in ("D", "A"):
        if not gold.intermediates:
            return "class D needs recorded intermediate cardinalities"
        if gold.independent:
            if max(gold.intermediates.values()) <= t_ind:
                return f"every independent sub-result at most T={t_ind}: the instance is class P"
            if min(gold.intermediates.values()) > h_cap:
                return f"no sub-result within H={h_cap}: no set can be bound exactly"
        else:
            if not gold.bound:
                return "a dependent instance needs the sizes of its bound sets"
            if max(gold.bound.values()) > h_cap:
                return f"bound set above H={h_cap}"
    return None


def generate_template(t: Template, ctx: Ctx, per_template: int, oversample: int = 8, progress=None):
    """Instances for one template: oversample candidates, validate, take round-robin by stratum."""
    rep = Report(t.id)
    say = progress or (lambda msg: None)
    say(f"{t.id}: sampling candidates")
    candidates = t.sample(ctx, per_template * oversample)
    say(f"{t.id}: {len(candidates)} candidates, computing gold")
    seen: set[str] = set()
    by_stratum: dict[str, list[tuple[dict, Gold]]] = {}
    for params in candidates:
        rep.candidates += 1
        if rep.candidates % 100 == 0:
            say(f"{t.id}: {rep.candidates}/{len(candidates)} candidates")
        key = _canonical(params)
        if key in seen:
            rep.reject("duplicate parameters")
            continue
        seen.add(key)
        gold = t.gold(ctx, params)
        why = check_gold(t, params, gold, ctx.t_independent, ctx.h_cap)
        if why:
            rep.reject(why)
            continue
        by_stratum.setdefault(params.get("stratum", "all"), []).append((params, gold))
    chosen: list[tuple[dict, Gold]] = []
    names = sorted(by_stratum)
    while len(chosen) < per_template and any(by_stratum.values()):
        for s in names:
            if by_stratum[s] and len(chosen) < per_template:
                chosen.append(by_stratum[s].pop(0))
    rep.accepted = len(chosen)
    for params, gold in chosen:
        rep.strata[params.get("stratum", "all")] = rep.strata.get(params.get("stratum", "all"), 0) + 1
        n = len(gold.value) if isinstance(gold.value, list) else 1
        rep.sizes.append(n)
    return chosen, rep


def render(t: Template, params: dict) -> list[str]:
    fmt = {k: v for k, v in params.items() if k != "stratum"}
    return [p.format(**fmt) for p in t.paraphrases]


def oracle_fingerprint(db: sqlite3.Connection) -> str:
    meta = dict(db.execute("SELECT key, value FROM oracle_meta"))
    return hashlib.sha256(meta.get("files", "").encode()).hexdigest()[:16]


def build_records(templates: list[Template], db: sqlite3.Connection, seed: int, per_template: int,
                 t_independent: int = T_INDEPENDENT, h_cap: int = H_BENCH, progress=None):
    fingerprint = oracle_fingerprint(db)
    records, reports = [], []
    for t in sorted(templates, key=lambda x: x.id):
        # one RNG per template, so adding a template never changes another's instances
        tseed = int(hashlib.sha256(f"{seed}:{t.id}".encode()).hexdigest()[:8], 16)
        ctx = Ctx(db, tseed, t_independent, h_cap)
        chosen, rep = generate_template(t, ctx, per_template, progress=progress)
        reports.append(rep)
        for i, (params, gold) in enumerate(chosen, 1):
            texts = render(t, params)
            records.append({
                "question_id": f"{t.id}-{i:03d}",
                "template": t.id,
                "class": t.cls,
                "split": t.split,
                "stratum": params.get("stratum", "all"),
                "text": texts[(i - 1) % 3],
                "paraphrases": texts,
                "params": {k: v for k, v in params.items() if k != "stratum"},
                "answer_type": t.answer_type,
                "entity": t.entity,
                "gold": gold.value,
                "services": t.services,
                "plan": [step_record(s, params.get(t.k_param or "k")) for s in t.plan],
                "intermediates": gold.intermediates,
                "independent": gold.independent,
                "bound": gold.bound,
                "thresholds": {"T": t_independent, "H": h_cap},
                "oracle": fingerprint,
                "seed": seed,
                "generator": GENERATOR_VERSION,
            })
    return records, reports


def step_record(s: Step, k: int | None) -> dict:
    """A plan step as stored in a question record (same keys as the system's plan JSON)."""
    d: dict[str, Any] = {"id": s.id, "op": s.op, "output": s.output, "kind": s.kind}
    if s.store:
        d["store"] = s.store
    if s.mode:
        d["mode"] = s.mode
    if s.question:
        d["question"] = s.question
    if s.inputs:
        d["inputs"] = list(s.inputs)
    if s.bind:
        d["bind"] = {"input": s.bind.input, "entity": s.bind.entity, "param": s.bind.param}
    if s.kind == "ranked" and k is not None:
        d["k"] = k
    if s.must_exist:
        d["must_exist"] = True
    return d


def render_report(reports: list[Report], requested: int) -> str:
    lines = [f"{'template':8s} {'asked':>5s} {'got':>4s} {'cand':>5s}  gold size (min/med/max)   strata   rejected"]
    for r in reports:
        sz = (f"{min(r.sizes)}/{int(statistics.median(r.sizes))}/{max(r.sizes)}" if r.sizes else "-")
        flag = "" if r.accepted >= requested else "   <-- SHORT"
        lines.append(f"{r.template:8s} {requested:5d} {r.accepted:4d} {r.candidates:5d}  {sz:22s}  "
                     f"{dict(sorted(r.strata.items()))}  {r.rejected or ''}{flag}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def all_templates() -> list[Template]:
    from . import catalogue_d, catalogue_sp
    return list(catalogue_sp.TEMPLATES) + list(catalogue_d.TEMPLATES)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate Benchmark v2 questions with gold answers")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True, help="JSON lines output file")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--per-template", type=int, default=5)
    ap.add_argument("--t-independent", type=int, default=T_INDEPENDENT, help="class P / D boundary T")
    ap.add_argument("--h-cap", type=int, default=H_BENCH, help="hard cap H on one retrieve step")
    ap.add_argument("--only", help="comma-separated template ids")
    ap.add_argument("--include-test", action="store_true",
                    help="also generate the frozen test split (leakage guard: off by default)")
    args = ap.parse_args(argv)

    templates = all_templates()
    if args.only:
        wanted = set(args.only.split(","))
        templates = [t for t in templates if t.id in wanted]
    if not args.include_test:
        templates = [t for t in templates if t.split == "dev"]
    if not templates:
        print("no templates selected", file=sys.stderr)
        return 2

    db = union.open_union(args.oracle)
    try:
        records, reports = build_records(templates, db, args.seed, args.per_template,
                                          args.t_independent, args.h_cap,
                                          progress=lambda m: print(m, file=sys.stderr, flush=True))
    finally:
        db.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(render_report(reports, args.per_template))
    short = [r.template for r in reports if r.accepted < args.per_template]
    print(f"\n{len(records)} questions from {len(templates)} templates written to {args.out}"
          + (f"; SHORT: {short}" if short else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
