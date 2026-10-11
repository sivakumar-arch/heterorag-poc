"""
benchmark_v2/run_slice.py
=========================
Vertical slice (Interface Specification v2.1, section 12): run the model-driven pipeline
(planner, translator, executor) on validated benchmark questions against the live stores and
score every stage separately, so that a failure is attributed to a stage.

Per question:

* ``plan_valid``      the planner returned a plan that passes the plan validator;
* ``plan_equal``      the plan has the same structure as the annotated plan (operators, stores,
                      modes, entity types, kinds, k, dependencies; step names and sub-question
                      wording are ignored);
* ``translation_ok``  every retrieve step produced a query that passed the static checks;
* ``outcome``         correct, wrong, partial_valid, partial_violated, abstain_expected or
                      abstain_unexpected (spec section 7);
* cost: model calls, tokens and seconds per role, repairs, rows read, native calls, wall time.

Models are selected per role with HETERORAG_PLANNER_MODEL and HETERORAG_TRANSLATOR_MODEL
(default claude-haiku-4-5-20251001, decision D23). Use only questions that passed
``validate_instances``; questions of the test split are scored but never printed with gold.

    python -m benchmark_v2.run_slice --questions data/questions_v2_dev.validated.jsonl \
        --templates S09,P01,P02 --limit 6 --out data/slice_dev.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from heterorag.federation import Executor, Plan, PlanError
from heterorag.federation.guarantee import Guarantee, holds
from heterorag.federation.plan import Op

CAP = 200_000          # same hard cap as the instance validator; the benchmark value is set later


# ----------------------------------------------------------------------------- plan equality

def signature(plan: Plan) -> tuple:
    """Structure of the plan below its final step. Names and sub-question wording are ignored;
    inputs of intersect and union are unordered, inputs of subtract are ordered."""
    by = plan.by_id()
    memo: dict[str, tuple] = {}

    def sig(sid: str) -> tuple:
        if sid in memo:
            return memo[sid]
        s = by[sid]
        ins = tuple(sig(i) for i in s.inputs)
        if s.op in (Op.INTERSECT, Op.UNION):
            ins = tuple(sorted(ins, key=repr))
        bind = (s.bind.entity,) if s.bind else ()
        memo[sid] = (s.op.value, s.store.value if s.store else None, s.mode.value if s.mode else None,
                     s.output, s.kind.value, s.k, bind, ins)
        return memo[sid]

    return sig(plan.sink().id)


def plans_equal(a: Plan | None, b: Plan) -> bool:
    return a is not None and signature(a) == signature(b)


# ----------------------------------------------------------------------------- outcome

def _value(ans, answer_type: str):
    return list(ans.value) if answer_type == "ranked" else (set(ans.value) if answer_type == "ids" else ans.value)


def classify(answer, rec: dict) -> str:
    """Outcome of one question (spec section 7). Class N (expected abstention) is not built yet,
    so every abstention is unexpected unless the record says otherwise."""
    if answer.abstained:
        return "abstain_expected" if rec.get("expect_abstain") else "abstain_unexpected"
    at, gold = rec["answer_type"], rec["gold"]
    truth = list(gold) if at == "ranked" else (set(gold) if at == "ids" else gold)
    got = _value(answer, at)
    if answer.guarantee is Guarantee.EXACT:
        return "correct" if got == truth else "wrong"
    return "partial_valid" if holds(answer.guarantee, got, truth) else "partial_violated"


# ----------------------------------------------------------------------------- one question

def run_one(rec: dict, federation) -> dict:
    t0 = time.perf_counter()
    out = federation.answer(rec["text"])
    gold_plan = Plan.from_dict({"steps": rec["plan"]})
    totals = out.run.totals() if out.run else {}
    errors = {k: t.errors for k, t in out.translations.items() if t.errors}
    row = {
        "question_id": rec["question_id"], "template": rec["template"], "class": rec.get("class"),
        "split": rec["split"], "text": rec["text"],
        "plan_valid": out.plan is not None,
        "plan_equal": plans_equal(out.plan, gold_plan),
        "plan_abstain": out.plan_result.abstain,
        "plan_errors": out.plan_result.errors,
        "plan_repairs": out.plan_result.repairs,
        "translation_ok": bool(out.translations) and not errors,
        "translation_errors": errors,
        "translation_repairs": sum(max(0, t.attempts - 1) for t in out.translations.values()),
        "outcome": classify(out.answer, rec),
        "answer_type": out.answer.type, "guarantee": out.answer.guarantee.value,
        "reason": out.answer.reason,
        "plan": out.plan.to_dict() if out.plan else None,
        "queries": out.queries,
        "usage": {"planner": out.planner_usage.to_dict(), "translator": out.translator_usage.to_dict()},
        "rows_read": int(totals.get("rows_read", 0)), "native_calls": int(totals.get("calls", 0)),
        "store_ms": round(float(totals.get("wall_ms", 0.0)), 1),
        "total_s": round(time.perf_counter() - t0, 2),
        "step_errors": [r.error for r in out.run.records.values() if r.error] if out.run else [],
    }
    if rec["split"] == "dev":
        row["gold_plan"] = rec["plan"]
        if out.answer.type != "abstain" and row["outcome"] in ("wrong", "partial_violated"):
            row["got"] = out.answer.value if out.answer.type != "ids" else out.answer.value[:20]
    return row


# ----------------------------------------------------------------------------- report

def summarise(rows: list[dict]) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["template"]].append(r)
    head = (f"{'template':8s} {'n':>3s} {'plan_ok':>7s} {'plan_eq':>7s} {'tr_ok':>5s} {'correct':>7s} "
            f"{'wrong':>5s} {'partial':>7s} {'abstain':>7s} {'repairs':>7s} {'tok_in':>7s} {'tok_out':>7s} {'sec':>6s}")
    lines = [head]
    tot = defaultdict(float)

    def line(name, rs):
        n = len(rs)
        c = lambda f: sum(1 for r in rs if f(r))
        tin = sum(r["usage"]["planner"]["prompt_tokens"] + r["usage"]["translator"]["prompt_tokens"] for r in rs)
        tout = sum(r["usage"]["planner"]["reply_tokens"] + r["usage"]["translator"]["reply_tokens"] for r in rs)
        rep = sum(r["plan_repairs"] + r["translation_repairs"] for r in rs)
        sec = sum(r["total_s"] for r in rs) / n
        return (f"{name:8s} {n:3d} {c(lambda r: r['plan_valid']):7d} {c(lambda r: r['plan_equal']):7d} "
                f"{c(lambda r: r['translation_ok']):5d} {c(lambda r: r['outcome'] == 'correct'):7d} "
                f"{c(lambda r: r['outcome'] == 'wrong'):5d} {c(lambda r: r['outcome'].startswith('partial')):7d} "
                f"{c(lambda r: r['outcome'].startswith('abstain')):7d} {rep:7d} {tin:7d} {tout:7d} {sec:6.1f}")

    for t in sorted(by):
        lines.append(line(t, by[t]))
    lines.append(line("all", rows))
    return "\n".join(lines)


def failures(rows: list[dict]) -> str:
    out = []
    for r in rows:
        if r["outcome"] == "correct":
            continue
        why = (r["plan_abstain"] or "") if not r["plan_valid"] else ""
        detail = r["plan_errors"] or r["translation_errors"] or r["step_errors"] or r["reason"] or ""
        out.append(f"{r['question_id']:9s} {r['outcome']:19s} plan_equal={str(r['plan_equal']):5s} {why} {str(detail)[:160]}")
    return "\n".join(out)


# ----------------------------------------------------------------------------- main

def build_federation(env=None):
    from heterorag.federation.clients import live_clients
    from heterorag.federation.contract import poc_catalogue
    from heterorag.federation.llm_config import RoleModels, make_provider
    from heterorag.federation.pipeline import Federation
    from heterorag.federation.planner import Planner
    from heterorag.federation.plan import Store
    from heterorag.federation.translator import Translator
    from heterorag.layer1.poc_descriptors import (CONTENT_SERVICE_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR,
                                                  USER_ACTIVITY_DESCRIPTOR)
    models = RoleModels.from_env(env)
    desc = {Store.SQL: USER_ACTIVITY_DESCRIPTOR, Store.GRAPH: KNOWLEDGE_GRAPH_DESCRIPTOR,
            Store.DOC: CONTENT_SERVICE_DESCRIPTOR}
    pm, tm = make_provider(models.planner, "planner"), make_provider(models.translator, "translator")
    fed = Federation(Planner(pm, desc), Translator(tm, desc, poc_catalogue()),
                     Executor(live_clients(env), cap=CAP))
    return fed, models


def main(argv: list[str] | None = None, federation=None, models=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="validated questions (JSON lines)")
    ap.add_argument("--templates", help="comma-separated template ids (default: all in the file)")
    ap.add_argument("--ids", help="comma-separated question ids")
    ap.add_argument("--limit", type=int, default=0, help="at most this many questions per template")
    ap.add_argument("--out", required=True, help="JSON output with per-question rows and run metadata")
    args = ap.parse_args(argv)

    recs = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    if args.templates:
        want = set(args.templates.split(","))
        recs = [r for r in recs if r["template"] in want]
    if args.ids:
        want = set(args.ids.split(","))
        recs = [r for r in recs if r["question_id"] in want]
    if args.limit:
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for r in recs:
            if seen[r["template"]] < args.limit:
                kept.append(r)
                seen[r["template"]] += 1
        recs = kept
    if not recs:
        print("no questions selected", file=sys.stderr)
        return 2

    if federation is None:
        federation, models = build_federation()
    print(f"planner model: {getattr(models, 'planner', '?')}   translator model: {getattr(models, 'translator', '?')}"
          f"   questions: {len(recs)}", flush=True)

    rows = []
    for i, rec in enumerate(recs, 1):
        try:
            row = run_one(rec, federation)
        except Exception as ex:                                  # a crash is recorded, not hidden
            row = {"question_id": rec["question_id"], "template": rec["template"], "split": rec["split"],
                   "text": rec["text"], "plan_valid": False, "plan_equal": False, "plan_abstain": None,
                   "plan_errors": [], "plan_repairs": 0, "translation_ok": False, "translation_errors": {},
                   "translation_repairs": 0, "outcome": "abstain_unexpected", "answer_type": "abstain",
                   "guarantee": "none", "reason": f"crash: {type(ex).__name__}: {ex}", "plan": None,
                   "queries": {}, "usage": {"planner": {"prompt_tokens": 0, "reply_tokens": 0},
                                            "translator": {"prompt_tokens": 0, "reply_tokens": 0}},
                   "rows_read": 0, "native_calls": 0, "store_ms": 0.0, "total_s": 0.0, "step_errors": []}
        rows.append(row)
        print(f"[{i}/{len(recs)}] {row['question_id']:9s} {row['outcome']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        from importlib.metadata import version
        sdk = version("anthropic")
    except Exception:
        sdk = None
    meta = {"anthropic_sdk": sdk, "planner_model": getattr(models, "planner", None), "translator_model": getattr(models, "translator", None),
            "questions_file": args.questions, "cap": CAP, "n": len(rows)}
    Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1, default=str))
    print()
    print(summarise(rows))
    bad = failures(rows)
    if bad:
        print("\nnot correct:\n" + bad)
    print(f"\nrows written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
