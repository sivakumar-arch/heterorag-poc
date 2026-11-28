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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from . import union

T_INDEPENDENT = 200      # decision D12: largest independent sub-result for class P
H_BENCH = 10_000         # hard cap on the rows of one retrieve step (benchmark setting)
MAX_SET = 50             # decision D15: largest identifier-set answer
GENERATOR_VERSION = "2.4"      # 2.4: validator caps per entity from the oracle, holder sets and bands for H1,
#   thresholds that must change the answer (Batch 105b). 2.3: step sizes and hashes in every record of a template with step sets; class F
                               #      rules, quotas and group caps (Batch 103); no question of an existing template changes
                               # 2.2: N1 correct-empty controls D01E and D05E (Batch 102); no other template changes
                               # 2.1: D01 paraphrase 3 says "questions tagged" (CP1 diagnosis: "regression questions" was ambiguous)

# English stop words of the Elasticsearch "_english_" list that are 5 or more letters long
# (shorter words are excluded by the length rule anyway).
STOPWORDS = frozenset({"their", "there", "these"})

# The validator runs every gold plan on the live stores without restriction (``validate_instances``). Batch 104b
# rejected at generation a question whose unrestricted gold plan has a step above the validator's cap (decision 4
# of 2026-11-26, second round). Batch 105b (decision 1 of 2026-11-28): the cap is computed from the oracle, per
# step, as the size of the table of that step's output entity; a store step that reaches it holds more ids than
# the oracle and is a data mismatch. On a consistent oracle no gold step can exceed its cap, so the generation
# rule is a guard.
ENTITY_TABLES: dict[str, str] = {
    "user": "SELECT count(*) FROM v_users",
    "question": "SELECT count(*) FROM v_questions",
    "answer": "SELECT count(*) FROM v_answers",
    "tag": "SELECT count(*) FROM tags",
}


def validator_caps(db: sqlite3.Connection) -> dict[str, int]:
    """entity type -> number of ids of that type in the oracle (the validator's cap for a step with that output)."""
    return {e: int(db.execute(q).fetchone()[0]) for e, q in ENTITY_TABLES.items()}

# N1 correct-empty controls (pre-registration section 3, Batch 102): the two cells of an N1 template.
N1_FINAL = "N1 final"                # both sub-results non-empty, their combination empty
N1_INTERMEDIATE = "N1 intermediate"  # the template's ``empty_step`` returns nothing


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
    # Class F (Batch 103): the size of every retrieve step's exact result, by step id (a bound step with its
    # exact bound input). Used by the step rule of ``check_gold`` and reported per shape.
    steps: dict[str, int] = field(default_factory=dict)
    # Batch 104: further fields copied into the question record (H2: the exclusive-reading gold).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Template:
    id: str
    cls: str                         # S | P | D | A | N | F (first-contact shapes, Batch 103)
    split: str                       # dev | test
    answer_type: str                 # ids | ranked | scalar
    entity: str                      # entity type of the answer (user, question, answer, tag, ...)
    paraphrases: tuple[str, str, str]
    plan: tuple[Step, ...]
    sample: Callable[["Ctx", int], list[dict]]
    gold: Callable[["Ctx", dict], Gold]
    k_param: str | None = None       # name of the parameter holding k for ranked answers
    # N1 control templates only: the id of the step that is empty in the ``N1_INTERMEDIATE`` cell.
    # A template with ``empty_step`` set has an empty gold answer by design; every other template
    # still rejects an empty gold (``check_gold``).
    empty_step: str | None = None
    # Batch 103. ``step_sets(ctx, params)`` gives the exact id set of every retrieve step; it is called only for
    # the chosen instances, and the record keeps each step's size and a hash of its sorted ids (``step_sizes``,
    # ``step_hashes``), which the pattern key compares with the step records of a run.
    step_sets: Callable[["Ctx", dict], dict[str, Any]] | None = None
    # Selection (Batch 103). ``quota``: (stratum, weight) pairs; the instances are shared between the strata in
    # proportion to the weights (largest remainder) and a stratum short of supply is not topped up from another.
    # ``group_caps``: (parameter, cap) pairs; at most ``cap`` chosen instances share a value of the parameter;
    # ``cap`` is an int or a {stratum: int} mapping.
    quota: tuple[tuple[str, int], ...] = ()
    group_caps: tuple[tuple[str, Any], ...] = ()
    # Batch 104b: (short stratum, donor stratum) pairs; with a quota, a stratum short of supply is filled from its
    # donor, and the shortfall and the fill are recorded in the sampler notes. Without a pair, no top-up.
    fallback: tuple[tuple[str, str], ...] = ()
    # Batch 105b. ``share_caps``: (parameter, value, fraction, unit) tuples; if more than ``fraction`` of the
    # distinct ``unit`` values among the accepted candidates have ``parameter == value``, at most
    # floor(fraction * per_template) chosen instances have that value (decision 2 of 2026-11-28: single-holder
    # badges in H1). ``condition_step``: the step whose result is the answer without the threshold condition; a
    # candidate whose answer is not strictly smaller is rejected (decision 3 of 2026-11-28).
    share_caps: tuple[tuple[str, Any, float, str], ...] = ()
    condition_step: str | None = None

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
        self.notes: Counter = Counter()          # sampler notes (Batch 104: rank collisions), printed in the report

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

    def validator_caps(self) -> dict[str, int] | None:
        """The validator's cap per output entity, from the oracle (None without an oracle connection)."""
        if self.db is None:
            return None
        return self.cached("validator_caps", lambda: validator_caps(self.db))

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
    step_sizes: dict[str, list[int]] = field(default_factory=dict)   # class F: step id -> sizes of the chosen
    restricted: int = 0              # class F: chosen instances exact only through a bind-planner restriction
    notes: dict[str, int] = field(default_factory=dict)   # sampler notes (``Ctx.notes``)
    removed: list[int] = field(default_factory=list)     # Batch 106: items removed by the threshold, per chosen
    marks: dict[str, dict] = field(default_factory=dict)  # Batch 106: stratum -> {"n": chosen, mark: count}

    def reject(self, why: str) -> None:
        self.rejected[why] = self.rejected.get(why, 0) + 1


def _canonical(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def id_hash(ids) -> str:
    """Hash of an identifier set: sha256 of the sorted string forms joined by newlines, first 16 hex digits. The
    executor computes the same hash for each step record (``value_hash``); a test asserts that they agree."""
    return hashlib.sha256("\n".join(sorted(str(x).strip() for x in ids)).encode()).hexdigest()[:16]


# Batch 105b / 106 (decision 0 of 2026-11-28 and the H3 decision of the same day): thresholds whose share is
# reported per stratum, in the pool counts and in the generator report at the freeze.
THRESHOLD_MARKS = {"H3": (2, 102)}


def removed_by_condition(t: Template, gold: Gold) -> int | None:
    """How many items the threshold condition removes from the answer: the size of ``condition_step`` minus the
    answer size (Batch 106; at least 1 for an accepted question, by the rule of Batch 105b). None when the template
    has no condition step or the answer is not a set."""
    if t.condition_step is None or not isinstance(gold.value, list):
        return None
    without = gold.steps.get(t.condition_step)
    return None if without is None else without - len(gold.value)


def needs_restriction(t: Template, gold: Gold, h_cap: int = H_BENCH) -> list[str]:
    """Retrieve steps whose exact result is above H: the gold is exact only if the bind planner restricts them."""
    return sorted(sid for sid, n in gold.steps.items() if n > h_cap)


def _check_f(t: Template, gold: Gold, h_cap: int) -> str | None:
    """Class F, first-contact shapes (Batch 103, approved 2026-11-26): the registered cap rules, step by step, and
    no lower threshold T. Every bound step's input and output are within H; a step above H is allowed only as an
    input of an intersection whose other input is within H, or as the subtrahend of a difference whose first input
    is within H (the bind planner can then restrict it); both inputs of a union are within H; at least one retrieve
    step is within H."""
    sizes = gold.steps
    retrieve = [s for s in t.plan if s.op == "retrieve"]
    if not sizes or any(s.id not in sizes for s in retrieve):
        return "class F needs the size of every retrieve step"
    if not any(sizes[s.id] <= h_cap for s in retrieve):
        return f"no retrieve step within H={h_cap}"
    by = {s.id: s for s in t.plan}
    size = lambda sid: sizes.get(sid, 0)
    for s in retrieve:
        if s.bind is not None:
            if size(s.bind.input) > h_cap:
                return f"bound input of step {s.id} above H={h_cap}"
            if size(s.id) > h_cap:
                return f"bound step {s.id} returns more than H={h_cap}"
    for s in t.plan:
        if s.op == "union" and any(by[i].op == "retrieve" and size(i) > h_cap for i in s.inputs):
            return f"an input of union {s.id} is above H={h_cap}: the bind planner never restricts a union input"
    for s in retrieve:
        if s.bind is not None or size(s.id) <= h_cap:
            continue
        ok = False
        for c in t.plan:
            if s.id not in c.inputs:
                continue
            others = [i for i in c.inputs if i != s.id]
            if c.op == "intersect" and others and all(by[i].op != "retrieve" or size(i) <= h_cap for i in others):
                ok = True
            if c.op == "subtract" and c.inputs[1:] == (s.id,) and size(c.inputs[0]) <= h_cap:
                ok = True
        if not ok:
            return f"step {s.id} above H={h_cap} and not restrictable"
    return None


def empty_at(t: Template, stratum: str | None) -> str | None:
    """Where an N1 instance is empty: "final", the id of the empty step, or None (not an N1 template)."""
    if t.empty_step is None:
        return None
    return "final" if stratum == N1_FINAL else t.empty_step


def _check_n1(t: Template, params: dict, gold: Gold, t_ind: int, h_cap: int) -> str | None:
    """N1 correct-empty control: the gold is empty and the cell's condition holds on the oracle.

    final:        every sub-result is non-empty (only the combination is empty), and the class D
                  rule holds (some sub-result above T, some within H);
    intermediate: the sub-result of the template's ``empty_step`` is empty and every other
                  sub-result is within (T, H], so the empty step is the only empty one and the
                  other side is read exactly."""
    if t.answer_type != "ids":
        return "N1 control needs an identifier-set answer"
    v = gold.value
    if not isinstance(v, list) or v:
        return "N1 control: the gold answer is not empty"
    sizes = gold.intermediates
    if not sizes:
        return "class D needs recorded intermediate cardinalities"
    stratum = params.get("stratum")
    if stratum == N1_FINAL:
        if min(sizes.values()) < 1:
            return "N1 final: a sub-result is empty"
        if max(sizes.values()) <= t_ind:
            return f"every independent sub-result at most T={t_ind}: the instance is class P"
        if min(sizes.values()) > h_cap:
            return f"no sub-result within H={h_cap}: no set can be bound exactly"
        return None
    if stratum == N1_INTERMEDIATE:
        store = next((s.store for s in t.plan if s.id == t.empty_step), None)
        if sizes.get(store) != 0:
            return f"N1 intermediate: the {store} sub-result is not empty"
        others = [n for s, n in sizes.items() if s != store]
        if not others or not all(t_ind < n <= h_cap for n in others):
            return f"N1 intermediate: another sub-result is not within (T={t_ind}, H={h_cap}]"
        return None
    return f"N1 control: unknown stratum {stratum!r}"


def _not_validatable(t: Template, gold: Gold, caps: Mapping[str, int]) -> str | None:
    """The first step of the unrestricted gold plan whose size exceeds the validator's cap for its output entity."""
    entity = {s.id: s.output for s in t.plan}
    by_store: dict[str, set] = {}
    for s in t.plan:
        if s.op == "retrieve" and s.store:
            by_store.setdefault(s.store, set()).add(s.output)
    sized = [(sid, n, {entity.get(sid)}) for sid, n in [*gold.steps.items(), *gold.bound.items()]]
    sized += [(store, n, by_store.get(store, set())) for store, n in gold.intermediates.items()]
    for sid, n, ents in sized:
        limits = [caps[e] for e in ents if e in caps]
        if limits and n > min(limits):
            return f"not validatable: step {sid} above the validator cap"
    return None


def check_gold(t: Template, params: dict, gold: Gold, t_ind: int = T_INDEPENDENT,
               h_cap: int = H_BENCH, caps: Mapping[str, int] | None = None) -> str | None:
    """Return the reason an instance is unacceptable, or None. ``caps`` (entity -> validator cap, see
    ``validator_caps``) enables the not-validatable rule; without it the rule is not applied."""
    if t.empty_step is not None:
        return _check_n1(t, params, gold, t_ind, h_cap)
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
    if caps is not None:
        why = _not_validatable(t, gold, caps)
        if why:
            return why
    if t.condition_step is not None and isinstance(v, list):
        without = gold.steps.get(t.condition_step)
        if without is None:
            return f"the answer without the threshold (step {t.condition_step}) is not recorded"
        if len(v) >= without:
            return "the threshold removes nothing from the answer"
    if t.cls == "F":
        if t.answer_type == "scalar" and not (isinstance(v, int) and v >= 1):
            return "class F count must be at least 1"
        if gold.extra.get("forms_agree") is False:
            return "the two forms of the gold plan give different sets"
        return _check_f(t, gold, h_cap)
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
    rep.notes = dict(ctx.notes)
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
        why = check_gold(t, params, gold, ctx.t_independent, ctx.h_cap, ctx.validator_caps())
        if why:
            rep.reject(why)
            continue
        by_stratum.setdefault(params.get("stratum", "all"), []).append((params, gold))
    chosen = select(t, by_stratum, per_template, ctx.notes)
    rep.notes = dict(ctx.notes)
    rep.accepted = len(chosen)
    for params, gold in chosen:
        rep.strata[params.get("stratum", "all")] = rep.strata.get(params.get("stratum", "all"), 0) + 1
        n = len(gold.value) if isinstance(gold.value, list) else 1
        rep.sizes.append(n)
        for sid, n_step in gold.steps.items():
            rep.step_sizes.setdefault(sid, []).append(n_step)
        rep.restricted += bool(needs_restriction(t, gold, ctx.h_cap))
        r = removed_by_condition(t, gold)
        if r is not None:
            rep.removed.append(r)
        if t.id in THRESHOLD_MARKS:
            m = rep.marks.setdefault(params.get("stratum", "all"), {"n": 0})
            m["n"] += 1
            for mark in THRESHOLD_MARKS[t.id]:
                m[mark] = m.get(mark, 0) + (params.get("number") == mark)
    return chosen, rep


def quota_split(quota: tuple[tuple[str, int], ...], n: int) -> dict[str, int]:
    """Share n between the strata in proportion to the weights, by largest remainder (ties by stratum name)."""
    total = sum(w for _, w in quota)
    exact = {s: n * w / total for s, w in quota}
    out = {s: int(x) for s, x in exact.items()}
    for s in sorted(exact, key=lambda s: (-(exact[s] - out[s]), s))[:n - sum(out.values())]:
        out[s] += 1
    return out


def select(t: Template, by_stratum: dict[str, list], per_template: int,
           notes: Counter | None = None) -> list[tuple[dict, Gold]]:
    """Take instances in candidate order. Without a quota: round-robin over the strata in name order. With a quota:
    each stratum up to its share, no top-up. Group caps skip a candidate whose parameter value is already used up."""
    used: dict[tuple[str, Any], int] = {}
    share_limit: dict[tuple[str, Any], int] = {}     # Batch 105b: (parameter, value) -> at most this many chosen
    for name, value, fraction, unit in t.share_caps:
        units = {_canonical(p.get(unit)) for queue in by_stratum.values() for p, _ in queue}
        hit = {_canonical(p.get(unit)) for queue in by_stratum.values() for p, _ in queue if p.get(name) == value}
        if units and len(hit) > fraction * len(units):
            share_limit[(name, value)] = int(fraction * per_template)
            if notes is not None:
                notes[f"{t.id} share cap {name}={value}: {len(hit)} of {len(units)} {unit}s, "
                      f"at most {share_limit[(name, value)]} of {per_template}"] += 1

    def fits(params: dict) -> bool:
        for name, cap in t.group_caps:
            c = cap.get(params.get("stratum"), 0) if isinstance(cap, dict) else cap
            if used.get((name, _canonical(params.get(name))), 0) >= c:
                return False
        for (name, value), limit in share_limit.items():
            if params.get(name) == value and used.get(("share", name, value), 0) >= limit:
                return False
        return True

    def take(params: dict) -> None:
        for name, _ in t.group_caps:
            k = (name, _canonical(params.get(name)))
            used[k] = used.get(k, 0) + 1
        for (name, value) in share_limit:
            if params.get(name) == value:
                used[("share", name, value)] = used.get(("share", name, value), 0) + 1

    def next_fitting(stratum: str):
        queue = by_stratum[stratum]
        while queue:
            item = queue.pop(0)
            if fits(item[0]):
                take(item[0])
                return item
        return None

    chosen: list[tuple[dict, Gold]] = []
    if t.quota:
        wants = quota_split(t.quota, per_template)
        got: dict[str, int] = {}
        for stratum, want in sorted(wants.items()):
            got[stratum] = 0
            while got[stratum] < want and by_stratum.get(stratum):
                item = next_fitting(stratum)
                if item is None:
                    break
                chosen.append(item)
                got[stratum] += 1
        for short, donor in t.fallback:
            deficit = wants.get(short, 0) - got.get(short, 0)
            if deficit <= 0:
                continue
            filled = 0
            while filled < deficit and by_stratum.get(donor):
                item = next_fitting(donor)
                if item is None:
                    break
                chosen.append(item)
                filled += 1
            if notes is not None:
                notes[f"{t.id} shortfall of {short}"] += deficit
                notes[f"{t.id} shortfall of {short} filled from {donor}"] += filled
        return chosen
    names = sorted(by_stratum)
    while len(chosen) < per_template and any(by_stratum.values()):
        for s in names:
            if by_stratum[s] and len(chosen) < per_template:
                item = next_fitting(s)
                if item is not None:
                    chosen.append(item)
    return chosen


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
                # Batch 105: a template that fixes the wording variant per instance (H4's rotation) sets
                # params["variant"]; otherwise the variant follows the question number, as before.
                "text": texts[params["variant"] if "variant" in params else (i - 1) % 3],
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
            where = empty_at(t, params.get("stratum"))
            if where is not None:
                records[-1].update(control="N1", empty_at=where)
            if t.step_sets is not None:
                sets = t.step_sets(ctx, params)
                records[-1]["step_sizes"] = {sid: len(v) for sid, v in sorted(sets.items())}
                records[-1]["step_hashes"] = {sid: id_hash(v) for sid, v in sorted(sets.items())}
            if t.cls == "F":
                records[-1]["needs_restriction"] = needs_restriction(t, gold, h_cap)
                if isinstance(gold.value, list):
                    records[-1]["answer_size"] = len(gold.value)
            if gold.extra:
                records[-1].update({k: v for k, v in gold.extra.items() if k != "forms_agree"})
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
        if r.step_sizes:
            steps = ", ".join(f"{sid} {min(v)}/{int(statistics.median(v))}/{max(v)}"
                              for sid, v in sorted(r.step_sizes.items()))
            lines.append(f"{'':8s} step sizes (min/med/max): {steps}; exact only through a bind-planner "
                         f"restriction: {r.restricted} of {r.accepted}")
        if r.removed:                                # Batch 106: thresholds that remove only one or two items
            one, two = r.removed.count(1), r.removed.count(2)
            lines.append(f"{'':8s} threshold removes 1 item: {one}, 2 items: {two}, either: {one + two} of "
                         f"{len(r.removed)}")
        for st, m in sorted(r.marks.items()):          # Batch 106: NUMBER marks per stratum (H3: 2 and 102)
            lines.append(f"{'':8s} {st}: " + ", ".join(f"NUMBER = {k}: {v} of {m['n']}"
                                                      for k, v in m.items() if k != "n"))
        if r.notes:
            lines.append(f"{'':8s} sampler notes: {dict(sorted(r.notes.items()))}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def all_templates() -> list[Template]:
    from . import catalogue_d, catalogue_fc, catalogue_n1, catalogue_sp
    return (list(catalogue_sp.TEMPLATES) + list(catalogue_d.TEMPLATES) + list(catalogue_n1.TEMPLATES)
            + list(catalogue_fc.TEMPLATES) + list(catalogue_fc.PROBE_TEMPLATES))


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
        # Batch 105: a probe template (split "probe", outside the fresh set) is generated only when named in --only
        templates = [t for t in templates if t.split == "dev" or (t.split == "probe" and args.only)]
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
