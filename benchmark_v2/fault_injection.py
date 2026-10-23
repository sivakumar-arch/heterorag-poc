"""
benchmark_v2/fault_injection.py
===============================
Fault injection into native queries: what do the step validator, the executor and the guarantee
labels catch when the translated query is wrong?

The pipeline claims that a language model proposes and code disposes. The CP1 and CP1b runs show
no silent error after the fixes, but zero events is an absence, not a rate. This module measures
the rate directly. For a question whose gold plan and hand-written reference queries are known,
one step's query is replaced by a deliberately faulty variant (a mutant) and the plan is run on
the live stores. No language model is involved.

Each mutant gets two independent verdicts.

    static    does the step validator reject it before execution? (In the pipeline a rejected
              query goes back to the translator once.)
    dynamic   what happens when it is executed regardless? The typed answer is classified as in
              the pipeline runs: no effect (still the gold answer), store error or abstention,
              flagged with a guarantee label that holds, flagged with a label that is violated,
              or silent (labelled EXACT and different from gold).

The final verdict follows from both: a mutant is *rejected* if the validator stops it, *harmless*
if it passes and does not change the answer, *stopped* if execution abstains, *flagged* if the
answer carries a label other than EXACT, and *silent* if it passes, runs, claims EXACT and is
wrong. A *violated* label is a flagged answer whose label is false (SUBSET with an item that is
not in gold): it is as unsound as a silent error and is counted with it as ``unsound``.

Faults are of two groups. Structural faults break something the catalogue knows about (an unknown
table, the wrong entity returned, a changed bind, a fetch limit added). Semantic faults keep the
query valid against the catalogue and change its meaning (a flipped comparison, a dropped
predicate, a reversed relationship, a changed constant). The validator is built to see only the
first group; whether anything else catches the second is what the measurement shows.

Limits, stated so they are not forgotten. The faults are generic mutations of the reference
queries, not a sample of the faults a model makes; the share of each group among real
translations is a separate measurement (the CP1 diagnosis found case of values, a split path and
a misread phrase). Faults that violate the case rule cannot be injected because the reference
queries are exact-match ground truth and are checked with the rule off. Write statements are
checked statically and never executed.

Run (live stores, no model):

    python -m benchmark_v2.fault_injection --questions data/questions_v2_dev_classD_cp1.validated.jsonl \\
        --out data/fault_cp1.json --report data/fault_cp1.md

Use questions that are already consumed by diagnosis (the CP1 set, the S and P dev set), not the
CP1b set, so that any validator improvement made after reading these results can still be
confirmed on questions never used for it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from heterorag.federation import Executor, Plan, PlanError, Store
from heterorag.federation.contract import Catalogue
from heterorag.federation.guarantee import Kind
from heterorag.federation.plan import Step
from heterorag.federation.step_validator import StepValidator

from .cp1_report import exact_interval
from .reference_queries import reference_queries
from .run_slice import classify

CAP = 10_000
BATCH = 1000

STRUCTURAL, SEMANTIC = "structural", "semantic"
FINALS = ("harmless", "rejected", "stopped", "flagged", "violated", "silent", "unrun_pass")


# ----------------------------------------------------------------------------- fault operators

@dataclass(frozen=True)
class Fault:
    id: str
    group: str
    stores: tuple
    what: str
    fn: Callable[[str, Step, Catalogue], "str | None"]
    run: bool = True                     # False: checked statically only, never executed

    def apply(self, query: str, step: Step, cat: Catalogue) -> "str | None":
        if step.store not in self.stores:
            return None
        out = self.fn(query, step, cat)
        return out if out is not None and out != query else None


_SQL, _GR, _DOC = Store.SQL, Store.GRAPH, Store.DOC


def _sub_first(pattern: str, repl: Callable | str, text: str, flags=re.I) -> "str | None":
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    return new if n else None


def _where_region(q: str) -> tuple[int, int] | None:
    m = re.search(r"\bWHERE\b", q, re.I)
    if not m:
        return None
    end = re.search(r"\b(ORDER\s+BY|LIMIT|GROUP\s+BY|RETURN)\b", q[m.end():], re.I)
    return m.end(), (m.end() + end.start() if end else len(q))


# ---- SQL

def sql_unknown_table(q, step, cat):
    return _sub_first(r"\bFROM\s+(\w+)", lambda m: f"FROM {m.group(1)}_x", q)


def sql_unknown_column(q, step, cat):
    return _sub_first(r"\bWHERE\s+(\w+)", lambda m: f"WHERE {m.group(1)}_x", q)


def sql_wrong_id(q, step, cat):
    if step.kind is Kind.SCALAR:
        return None
    t = re.search(r"\bFROM\s+(\w+)", q, re.I)
    cols = cat.sql.tables.get(t.group(1)) if t else None
    other = next((c for c in cols or () if c not in ("id", "user_id", "post_id", "tag_name")), None)
    if other is None:
        return None
    return _sub_first(r"^(\s*SELECT\s+(?:DISTINCT\s+)?)(\w+)", lambda m: m.group(1) + other, q)


def sql_limit_changed(q, step, cat):
    if re.search(r"\bLIMIT\s+\d+", q, re.I):
        return _sub_first(r"\bLIMIT\s+(\d+)", lambda m: f"LIMIT {int(m.group(1)) + 3}", q)
    return q.rstrip().rstrip(";") + " LIMIT 5"


def sql_write_appended(q, step, cat):
    return q.rstrip().rstrip(";") + "; DELETE FROM users"


def sql_flip_comparison(q, step, cat):
    reg = _where_region(q)
    if not reg:
        return None
    a, b = reg
    flip = {">=": "<", "<=": ">", "<>": "=", ">": "<=", "<": ">=", "=": "<>"}
    m = re.search(r">=|<=|<>|>|<|=", q[a:b])
    return None if not m else q[:a + m.start()] + flip[m.group(0)] + q[a + m.end():]


def _scale_last_number(q: str, region: tuple[int, int] | None) -> "str | None":
    a, b = region if region else (0, len(q))
    nums = list(re.finditer(r"(?<![\w.'\"])\d+(?![\w.'\"])", q[a:b]))
    if not nums:
        return None
    m = nums[-1]
    return q[:a + m.start()] + str(int(m.group(0)) * 2 + 1) + q[a + m.end():]


def sql_scale_number(q, step, cat):
    return _scale_last_number(q, _where_region(q))


def sql_string_suffix(q, step, cat):
    return _sub_first(r"'((?:[^']|'')*)'", lambda m: f"'{m.group(1)}x'", q)


def sql_drop_predicate(q, step, cat):
    reg = _where_region(q)
    if not reg:
        return None
    a, b = reg
    body = q[a:b]
    parts = re.split(r"\bAND\b", body, flags=re.I)
    if len(parts) > 1:
        return (q[:a] + "AND".join(parts[:-1]).rstrip() + " " + q[b:]).strip()
    return (q[:a - len("WHERE")].rstrip() + " " + q[b:]).strip()


# ---- Cypher

def cy_unknown_label(q, step, cat):
    return _sub_first(r"\(\s*\w*\s*:\s*(\w+)", lambda m: m.group(0) + "x", q)


def cy_unknown_relationship(q, step, cat):
    return _sub_first(r"\[\s*\w*\s*:\s*(\w+)", lambda m: m.group(0) + "_X", q)


def cy_wrong_entity(q, step, cat):
    if step.kind is Kind.SCALAR:
        return None
    ret = re.search(r"\bRETURN\s+(DISTINCT\s+)?(\w+)\.(\w+)", q, re.I)
    if not ret:
        return None
    var = ret.group(2)
    nodes = re.findall(r"\(\s*(\w*)\s*:\s*(\w+)", q)
    own = next((lab for v, lab in nodes if v == var), None)
    other = next(((v, lab) for v, lab in nodes if lab != own and v != var), None)
    if other is None:
        return None
    v, lab = other
    if not v:                                                 # an anonymous node: give it a name
        q = q.replace(f"(:{lab}", f"(zz:{lab}", 1)
        v = "zz"
    prop = "name" if lab == "Tag" else "id"
    ret = re.search(r"\bRETURN\s+(DISTINCT\s+)?(\w+)\.(\w+)", q, re.I)
    return q[:ret.start()] + f"RETURN {ret.group(1) or ''}{v}.{prop}" + q[ret.end():]


def cy_reverse_direction(q, step, cat):
    return _sub_first(r"-(\[[^\]]*\])->", lambda m: f"<-{m.group(1)}-", q)


def cy_swap_relationship(q, step, cat):
    m = re.search(r"\[\s*\w*\s*:\s*(\w+)", q)
    if not m:
        return None
    rels = sorted(cat.graph.relationships)
    if m.group(1) not in rels or len(rels) < 2:
        return None
    nxt = rels[(rels.index(m.group(1)) + 1) % len(rels)]
    return q[:m.start(1)] + nxt + q[m.end(1):]


def cy_string_suffix(q, step, cat):
    return _sub_first(r'"((?:[^"\\]|\\.)*)"', lambda m: f'"{m.group(1)}x"', q)


def cy_scale_number(q, step, cat):
    lim = re.search(r"\bLIMIT\b", q, re.I)                      # a fetch limit is a structural fault, not a constant
    return _scale_last_number(q, (0, lim.start()) if lim else None)


def cy_drop_filter(q, step, cat):
    new = _sub_first(r"\s*\{[^{}]*\}", "", q)
    return new if new is not None else _sub_first(r"\s+WHERE\s+.*?(?=\s+RETURN\b)", "", q, re.I | re.S)


# ---- Elasticsearch

def _es(q: str) -> "dict | None":
    try:
        d = json.loads(q)
        return d if isinstance(d, dict) else None
    except ValueError:
        return None


def _dump(d: dict) -> str:
    return json.dumps(d)


def _filters(d: dict) -> "list | None":
    try:
        f = d["query"]["bool"]["filter"]
        return f if isinstance(f, list) else None
    except (KeyError, TypeError):
        return None


def es_unknown_field(q, step, cat):
    """A text field first (the likelier slip), the doc_type field when the query has no text field."""
    return (_sub_first(r'"(title|body)"', lambda m: f'"{m.group(1)}x"', q)
            or _sub_first(r'"doc_type"', '"doc_typex"', q))


def es_drop_doc_type(q, step, cat):
    d = _es(q)
    f = _filters(d) if d else None
    if not f:
        return None
    keep = [c for c in f if not (isinstance(c, dict) and any(k in ("term", "terms") and "doc_type" in v
                                                              for k, v in c.items() if isinstance(v, dict)))]
    if len(keep) == len(f):
        return None
    d["query"]["bool"]["filter"] = keep
    return _dump(d)


def es_extra_size(q, step, cat):
    d = _es(q)
    if d is None:
        return None
    d["size"] = 5
    return _dump(d)


def es_operator_or(q, step, cat):
    return _sub_first(r'"operator"\s*:\s*"and"', '"operator": "or"', q)


def es_string_suffix(q, step, cat):
    d = _es(q)
    f = _filters(d) if d else None
    for c in f or ():
        m = c.get("match") if isinstance(c, dict) else None
        if isinstance(m, dict):
            for field, spec in m.items():
                if isinstance(spec, dict) and isinstance(spec.get("query"), str):
                    spec["query"] += "x"
                    return _dump(d)
    return None


def es_drop_match(q, step, cat):
    d = _es(q)
    f = _filters(d) if d else None
    if not f:
        return None
    keep = [c for c in f if not (isinstance(c, dict) and ("match" in c or "match_phrase" in c))]
    if len(keep) == len(f) or not keep:
        return None
    d["query"]["bool"]["filter"] = keep
    return _dump(d)


# ---- bind (all stores)

def bind_renamed(q, step, cat):
    if step.bind is None:
        return None
    n = step.bind.name
    if step.store is Store.SQL:
        return q.replace(f"%({n})s", "%(zz)s")
    if step.store is Store.GRAPH:
        return re.sub(rf"\${n}\b", "$zz", q)
    return q.replace(f"$bind:{n}", "$bind:zz")


def cy_write_appended(q, step, cat):
    return q.rstrip().rstrip(";") + " DETACH DELETE x"


FAULTS: tuple[Fault, ...] = (
    Fault("sql_unknown_table", STRUCTURAL, (_SQL,), "table name does not exist", sql_unknown_table),
    Fault("sql_unknown_column", STRUCTURAL, (_SQL,), "unqualified column in WHERE does not exist", sql_unknown_column),
    Fault("sql_wrong_id", STRUCTURAL, (_SQL,), "first selected item is not the identifier", sql_wrong_id),
    Fault("sql_limit_changed", STRUCTURAL, (_SQL,), "a fetch limit is added or changed", sql_limit_changed),
    Fault("sql_write_appended", STRUCTURAL, (_SQL,), "a write statement is appended", sql_write_appended, run=False),
    Fault("cy_unknown_label", STRUCTURAL, (_GR,), "node label does not exist", cy_unknown_label),
    Fault("cy_unknown_relationship", STRUCTURAL, (_GR,), "relationship type does not exist", cy_unknown_relationship),
    Fault("cy_wrong_entity", STRUCTURAL, (_GR,), "returns the identifier of another entity type", cy_wrong_entity),
    Fault("cy_write_appended", STRUCTURAL, (_GR,), "a write clause is appended", cy_write_appended, run=False),
    Fault("es_unknown_field", STRUCTURAL, (_DOC,), "field does not exist", es_unknown_field),
    Fault("es_drop_doc_type", STRUCTURAL, (_DOC,), "doc_type restriction removed", es_drop_doc_type),
    Fault("es_extra_size", STRUCTURAL, (_DOC,), "a size key is added", es_extra_size),
    Fault("bind_renamed", STRUCTURAL, (_SQL, _GR, _DOC), "bind placeholder renamed", bind_renamed),
    Fault("sql_flip_comparison", SEMANTIC, (_SQL,), "first comparison in WHERE inverted", sql_flip_comparison),
    Fault("sql_scale_number", SEMANTIC, (_SQL,), "last numeric constant in WHERE doubled plus one", sql_scale_number),
    Fault("sql_string_suffix", SEMANTIC, (_SQL,), "first string constant gets a suffix", sql_string_suffix),
    Fault("sql_drop_predicate", SEMANTIC, (_SQL,), "last predicate (or the whole WHERE) removed", sql_drop_predicate),
    Fault("cy_reverse_direction", SEMANTIC, (_GR,), "first relationship direction reversed", cy_reverse_direction),
    Fault("cy_swap_relationship", SEMANTIC, (_GR,), "relationship replaced by another valid type", cy_swap_relationship),
    Fault("cy_string_suffix", SEMANTIC, (_GR,), "first string constant gets a suffix", cy_string_suffix),
    Fault("cy_scale_number", SEMANTIC, (_GR,), "last numeric constant doubled plus one", cy_scale_number),
    Fault("cy_drop_filter", SEMANTIC, (_GR,), "first property filter (or WHERE) removed", cy_drop_filter),
    Fault("es_operator_or", SEMANTIC, (_DOC,), "match operator and replaced by or", es_operator_or),
    Fault("es_string_suffix", SEMANTIC, (_DOC,), "query text gets a suffix", es_string_suffix),
    Fault("es_drop_match", SEMANTIC, (_DOC,), "text condition removed", es_drop_match),
)
FAULT_BY_ID = {f.id: f for f in FAULTS}


# ----------------------------------------------------------------------------- one mutant

def _dynamic(plan: Plan, queries: Mapping[str, str], rec: dict, executor: Executor) -> dict:
    """Run the plan with these queries and classify the typed answer against gold."""
    try:
        run = executor.run(plan, queries)
    except Exception as ex:                                      # infrastructure, not a result
        return {"dyn": "infra", "detail": f"{type(ex).__name__}: {ex}"[:200]}
    ans = run.answer
    errs = [r.error for r in run.records.values() if r.error]
    rows = int(run.totals()["rows_read"])
    if ans.abstained:
        kind = "premise_guard" if ans.reason == "premise_failed" else ("store_error" if errs else "incomplete")
        return {"dyn": kind, "detail": (errs[0] if errs else ans.reason or "")[:200], "rows_read": rows}
    outcome = classify(ans, rec)                                 # correct | wrong | partial_valid | partial_violated
    if outcome == "correct":
        return {"dyn": "no_effect", "rows_read": rows, "guarantee": ans.guarantee.value}
    if outcome == "wrong":
        return {"dyn": "silent", "rows_read": rows, "guarantee": ans.guarantee.value}
    return {"dyn": "flagged_holds" if outcome == "partial_valid" else "flagged_violated",
            "rows_read": rows, "guarantee": ans.guarantee.value}


def final_verdict(static_rejected: bool, dyn: str) -> str:
    if static_rejected:
        return "rejected"
    if dyn == "no_effect":
        return "harmless"
    if dyn == "silent":
        return "silent"
    if dyn == "flagged_holds":
        return "flagged"
    if dyn == "flagged_violated":
        return "violated"
    if dyn == "not_run":
        return "unrun_pass"                                      # never executed and not rejected: a validator hole
    return "stopped"                                             # store error, premise guard, incomplete


def inject_record(rec: dict, executor: Executor, validator: StepValidator, cat: Catalogue,
                  faults: tuple[Fault, ...] = FAULTS, per_fault: dict | None = None, cap_per_fault: int | None = None,
                  rng: random.Random | None = None) -> list[dict]:
    """All applicable mutants of one question. ``per_fault`` counts mutants already taken per fault id
    (shared across questions) so that ``cap_per_fault`` bounds the load on the live stores."""
    rows: list[dict] = []
    try:
        plan = Plan.from_dict({"steps": rec["plan"]})
        base = reference_queries(rec["template"], rec["params"])
    except (PlanError, KeyError, ValueError) as ex:
        return [{"question_id": rec["question_id"], "fault": None, "final": "setup_error",
                 "detail": f"{type(ex).__name__}: {ex}"[:200]}]
    steps = plan.by_id()
    order = list(faults)
    if rng:
        rng.shuffle(order)
    for fault in order:
        for sid, query in sorted(base.items()):
            step = steps[sid]
            if per_fault is not None and cap_per_fault is not None and per_fault.get(fault.id, 0) >= cap_per_fault:
                continue
            mutant = fault.apply(query, step, cat)
            if mutant is None:
                continue
            if per_fault is not None:
                per_fault[fault.id] = per_fault.get(fault.id, 0) + 1
            errs = validator.check(step, mutant)
            row = {"question_id": rec["question_id"], "template": rec["template"], "step": sid, "store": step.store.value,
                   "fault": fault.id, "group": fault.group, "static_rejected": bool(errs),
                   "static_errors": errs[:3], "query": mutant[:400]}
            row.update(_dynamic(plan, {**base, sid: mutant}, rec, executor) if fault.run else {"dyn": "not_run"})
            row["final"] = final_verdict(bool(errs), row["dyn"]) if row["dyn"] != "infra" else "infra"
            rows.append(row)
    return rows


def baseline_problems(rec: dict, validator: StepValidator) -> list[str]:
    """The unmodified reference queries must pass the validator, or no verdict on a mutant means anything."""
    plan = Plan.from_dict({"steps": rec["plan"]})
    base = reference_queries(rec["template"], rec["params"])
    return [f"{rec['question_id']}/{sid}: {e}" for sid, q in sorted(base.items()) for e in validator.check(plan.by_id()[sid], q)]


# ----------------------------------------------------------------------------- report

def _count(rows: list[dict], key: Callable[[dict], str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        c = out.setdefault(key(r), {f: 0 for f in FINALS} | {"n": 0, "unprotected_silent": 0, "rejected_harmless": 0})
        c["n"] += 1
        c[r["final"]] += 1
        c["unprotected_silent"] += r.get("dyn") == "silent"
        c["rejected_harmless"] += r["final"] == "rejected" and r.get("dyn") == "no_effect"
    return out


def _pct(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = exact_interval(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def _table(title: str, groups: dict[str, dict[str, int]]) -> list[str]:
    head = ("| " + title + " | mutants | harmless | rejected by validator | stopped at run time | flagged, label holds | "
            "flagged, label violated | silent (EXACT and wrong) | unsound (silent + violated) | silent without the validator |")
    out = [head, "|" + "---|" * 10]
    for k in sorted(groups):
        c = groups[k]
        effective = c["n"] - c["harmless"] - c["rejected_harmless"]
        out.append(f"| {k} | {c['n']} | {c['harmless']} | {c['rejected']} | {c['stopped']} | {c['flagged']} | {c['violated']} | "
                   f"{c['silent']} | {_pct(c['silent'] + c['violated'], effective)} of {effective} | {c['unprotected_silent']} |")
    return out


def render(rows: list[dict]) -> str:
    res = [r for r in rows if r["final"] not in ("infra", "setup_error")]
    bad = [r for r in rows if r["final"] in ("infra", "setup_error")]
    lines = ["# Fault injection report", "",
             f"{len(res)} mutants over {len({r['question_id'] for r in res})} questions "
             f"({len(bad)} rows excluded as infrastructure or setup errors).", "",
             "Unsound rate = (silent + violated) over effective mutants (not harmless, and not rejected mutants that would have been harmless). The interval is the exact "
             "95 percent interval over mutants; mutants of one question are not independent, so it is narrower than it should be.", ""]
    lines += _table("group", _count(res, lambda r: r["group"])) + [""]
    lines += _table("fault", _count(res, lambda r: r["fault"])) + [""]
    lines += _table("store", _count(res, lambda r: r["store"])) + [""]
    lines += _table("template", _count(res, lambda r: r["template"])) + [""]
    holes = [r for r in res if r["final"] == "unrun_pass"]
    if holes:
        lines += [f"WARNING: {len(holes)} write mutants were not rejected by the validator and were not run: " +
                  ", ".join(sorted({r['fault'] for r in holes})), ""]
    sil = [r for r in res if r["final"] in ("silent", "violated")]
    if sil:
        lines += ["## Mutants that were not caught (first 25)", "", "| question | step | fault | final | guarantee | query |", "|---|---|---|---|---|---|"]
        for r in sil[:25]:
            lines.append(f"| {r['question_id']} | {r['step']} | {r['fault']} | {r['final']} | {r.get('guarantee', '')} | "
                         f"{r['query'][:110].replace('|', '/')} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None, clients: Mapping | None = None, catalogue: Catalogue | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="validated questions (JSON lines); use consumed questions, not CP1b")
    ap.add_argument("--only", help="comma-separated template ids")
    ap.add_argument("--faults", help="comma-separated fault ids (default: all)")
    ap.add_argument("--max-questions", type=int, help="use at most this many questions per template")
    ap.add_argument("--per-fault", type=int, help="at most this many mutants per fault (bounds the load on the stores)")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", help="JSON rows")
    ap.add_argument("--report", help="markdown report")
    args = ap.parse_args(argv)

    records = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    if args.only:
        records = [r for r in records if r["template"] in set(args.only.split(","))]
    if args.max_questions:
        seen: dict[str, int] = {}
        kept = []
        for r in records:
            seen[r["template"]] = seen.get(r["template"], 0) + 1
            if seen[r["template"]] <= args.max_questions:
                kept.append(r)
        records = kept
    if not records:
        print("no questions selected", file=sys.stderr)
        return 2
    faults = FAULTS
    if args.faults:
        want = args.faults.split(",")
        unknown = [w for w in want if w not in FAULT_BY_ID]
        if unknown:
            print(f"unknown fault ids: {unknown}", file=sys.stderr)
            return 2
        faults = tuple(FAULT_BY_ID[w] for w in want)

    if catalogue is None:
        from heterorag.federation.contract import poc_catalogue
        catalogue = poc_catalogue()
    if clients is None:
        from heterorag.federation.clients import live_clients
        clients = live_clients()
    validator = StepValidator(catalogue, case_insensitive_text=False)
    executor = Executor(clients, batch_size=BATCH, cap=CAP)

    problems = [p for r in records for p in baseline_problems(r, validator)]
    if problems:
        print("the unmodified reference queries fail the validator; no verdict would mean anything:", file=sys.stderr)
        print("\n".join(problems[:10]), file=sys.stderr)
        return 3

    rng, taken = random.Random(args.seed), {}
    rows: list[dict] = []
    for i, rec in enumerate(records, 1):
        rows += inject_record(rec, executor, validator, catalogue, faults, taken, args.per_fault, rng)
        if i % 10 == 0:
            print(f"{i} of {len(records)} questions, {len(rows)} mutants", file=sys.stderr)
    text = render(rows)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"meta": {"questions": args.questions, "seed": args.seed, "per_fault": args.per_fault,
                                                       "faults": [f.id for f in faults]}, "rows": rows}, indent=1))
    if args.report:
        Path(args.report).write_text(text + "\n")
    return 0 if not any(r["final"] == "infra" for r in rows) else 75


if __name__ == "__main__":
    sys.exit(main())
