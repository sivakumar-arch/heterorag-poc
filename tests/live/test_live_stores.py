"""
tests/live/test_live_stores.py
==============================
Batch 100 (build item 1): the federation executor and store clients against the LIVE stores, compared with
the oracle. Opt-in: these tests carry the marker ``live_stores`` and are deselected by default. Run on the
machine that has the stores and the oracle:

    python -m pytest tests/live -m live_stores -q

They read only (the clients open read-only sessions; the oracle is opened read-only). No language model is
called. A store or the oracle that cannot be reached FAILS the test rather than skipping it, so a run that
reports passes has really touched the stores.

What is tested, with gold computed from the oracle by the generator's own functions (``catalogue_d``):
  1. D04 and D07 (sql badge holders bound into a graph step) for badges with more than 1,000 holders and at
     most H = 10,000: the bound set spans several batches, the union over batches equals the oracle's answer
     and the label is EXACT; at the default batch size (1,000) and at 97.
     Because D04's answer (tags) saturates, a bound step without overlap is tested as well: the answers to the
     questions with a score of at least S (one parent question per answer, so every lost id changes the answer).
  2. The same plan with a batch size of 7 on a badge with 101 to 300 holders (many batches on a small set).
  3. D01 with the bind planner's restriction (sql reputation filter restricted to the graph askers of a tag
     with more than 1,000 askers): exact, contained (C5 counter 0), bind origin "policy"; and the independent
     plan agrees when the sql side fits the cap, or is labelled non-exact when it does not.
  4. The cap on each store: a step known to exceed 10,000 rows stops at the cap with ``cap_hit``, status
     partial and label SUBSET.
  5. Batch 105b (decision 1 of 2026-11-28): every entity of a type, read with the validator's cap for that type
     (the oracle's number of ids), is complete, not capped, and equal to the oracle's ids, on SQL and the graph
     (users) and on the document store (questions and answers, each far above Elasticsearch's 10,000-hit window:
     the client reads every match through a point in time and ``search_after``). Each case prints its time.
Environment: the store variables of ``live_clients`` (PG_DSN, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
ES_URL, with the local defaults) and HETERORAG_ORACLE (default data/oracle_stats.sqlite).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmark_v2 import union
from benchmark_v2.catalogue_d import (D01, D04, D07, _askers_by_tag, _badge_holders,
                                      _reputation, _users_at_least, d01_gold, d04_gold, d07_gold)
from benchmark_v2.questions import Ctx, step_record, validator_caps
from benchmark_v2.reference_queries import reference_queries, sql_lit
from heterorag.federation import BIND_ORIGIN_POLICY, Bind, Executor, Mode, Op, Plan, Step, Store
from heterorag.federation.bindplan import restrict
from heterorag.federation.guarantee import Guarantee, Status

pytestmark = pytest.mark.live_stores

H = 10_000                 # registered cap
BIG = (1001, H)            # a bound set above one batch of 1,000 ids and within the cap
MID = (101, 300)           # a small bound set for many small batches
SMALL_BATCH = 7


def _plan(template) -> Plan:
    return Plan.from_dict({"steps": [step_record(s, None) for s in template.plan]})


@pytest.fixture(scope="module")
def ctx():
    path = Path(os.environ.get("HETERORAG_ORACLE", "data/oracle_stats.sqlite"))
    if not path.exists():
        pytest.fail(f"oracle not found at {path} (set HETERORAG_ORACLE)")
    db = union.open_union(str(path))          # read-only, with the oracle views
    yield Ctx(db, seed=20261116)
    db.close()


@pytest.fixture(scope="module")
def clients():
    try:
        from heterorag.federation.clients import live_clients
        c = live_clients()
    except Exception as ex:                                       # missing driver or bad configuration
        pytest.fail(f"cannot build the live store clients: {type(ex).__name__}: {ex}")
    probes = {Store.SQL: ("SELECT id FROM users", "user", None),
              Store.GRAPH: ("MATCH (u:User) RETURN u.id AS id", "user", None),
              Store.DOC: (json.dumps({"query": {"bool": {"filter": [{"term": {"doc_type": "question"}}]}}}),
                          "question", Mode.FILTER)}
    for store, (q, out, mode) in probes.items():
        try:
            res = c[store].fetch(Step("p", Op.RETRIEVE, store, mode, "probe", output=out), q, {}, 1)
        except Exception as ex:
            pytest.fail(f"{store.value} store not reachable: {type(ex).__name__}: {ex}")
        if len(res.values) != 1:
            pytest.fail(f"{store.value} store returned {len(res.values)} rows for a one-row probe")
    return c


def _pick(sizes: dict, lo: int, hi: int, n: int) -> list:
    """Up to n keys whose size is in [lo, hi]: the smallest, the largest and the median (deterministic)."""
    keys = sorted((k for k, v in sizes.items() if lo <= len(v) <= hi), key=lambda k: (len(sizes[k]), k))
    if not keys:
        return []
    chosen = [keys[0], keys[-1], keys[len(keys) // 2]]
    return list(dict.fromkeys(chosen))[:n]


# ---------------------------------------------------------------- 1 and 2: D04 and D07 across batches

def _run_d04(clients, ctx, badge, batch_size, template=D04):
    holders = _badge_holders(ctx)[badge]
    r = Executor(clients, batch_size=batch_size, cap=H).run(_plan(template),
                                                            reference_queries(template.id, {"badge": badge}))
    a, b = r.records["a"], r.records["b"]
    assert a.status is Status.COMPLETE and a.guarantee is Guarantee.EXACT and len(a.value) == len(holders), \
        f"badge {badge!r}: the sql step returned {a.value and len(a.value)} holders, the oracle has {len(holders)}"
    assert b.batches == -(-len(holders) // batch_size) and b.failed_batches == 0 and not b.cap_hit, \
        f"badge {badge!r}, batch {batch_size}: batches {b.batches}, failed {b.failed_batches}, cap {b.cap_hit}"
    assert b.status is Status.COMPLETE and b.guarantee is Guarantee.EXACT
    assert b.bind_origin == "planner"
    return r


def test_d04_bound_set_above_1000_ids_matches_the_oracle(clients, ctx):
    badges = _pick(_badge_holders(ctx), *BIG, 3)
    if not badges:
        pytest.fail(f"no badge with {BIG[0]} to {BIG[1]} holders in the oracle")
    for badge in badges:
        for bs in ((1000, 97) if badge == badges[0] else (1000,)):
            r = _run_d04(clients, ctx, badge, bs)
            gold = d04_gold(ctx, {"badge": badge}).value
            assert r.answer.exact and set(r.answer.value) == set(gold), \
                f"badge {badge!r}, batch {bs}: {len(set(r.answer.value) - set(gold))} extra, " \
                f"{len(set(gold) - set(r.answer.value))} missing of {len(gold)}"


def test_bound_step_without_overlap_matches_the_oracle_across_batches(clients, ctx):
    """D04's answer (tags) saturates: holders share tags, so a lost batch need not change it. Here the bound step
    maps questions to their answers, and every answer has exactly one parent question, so every lost id changes the
    answer. The bound set is the answered questions (``answer_count > 0``) with a score of at least S, for the
    highest S that gives at least BIG[0] of them (just above one batch), so that nearly every lost id removes at
    least one answer; a question whose ``answer_count`` counts answers missing from the dump has no loaded answer,
    and a lost id there would not show, so their number is reported in the messages. The answers must fit the
    cap, otherwise the test fails, because
    above the cap a truncated answer would also be a subset of the gold and a lost id could hide behind the cap.
    (Batch 100b: the first version used the questions of a badge's holders, and on the real oracle every badge
    with 1,001 to 10,000 holders has more than 10,000 such questions.)"""
    score = dict(ctx.rows("SELECT id, score FROM v_questions WHERE answer_count > 0"))
    answers: dict = {}
    for a, q in ctx.rows("SELECT answer_id, question_id FROM v_answers_of"):
        answers.setdefault(q, set()).add(a)
    s_min, bound = None, []
    for s in sorted(set(score.values()), reverse=True):
        bound = [q for q, v in score.items() if v >= s]
        if len(bound) >= BIG[0]:
            s_min = s
            break
    if s_min is None or len(bound) > BIG[1]:
        pytest.fail(f"no score threshold gives {BIG[0]} to {BIG[1]} questions")
    gold = {a for q in bound for a in answers.get(q, ())}
    invisible = sum(1 for q in bound if not answers.get(q))       # answered per answer_count, no answer loaded
    if not 0 < len(gold) <= H:
        pytest.fail(f"the {len(bound)} questions with score at least {s_min} have {len(gold)} answers, not within the cap {H}")
    plan = Plan((Step("a", Op.RETRIEVE, Store.SQL, None, "questions with a score of at least S", output="question"),
                 Step("b", Op.RETRIEVE, Store.GRAPH, None, "answers to these questions", inputs=("a",),
                      bind=Bind("a", "question"), output="answer")))
    queries = {"a": f"SELECT id FROM posts WHERE post_type_id = 1 AND answer_count > 0 AND score >= {sql_lit(s_min)}",
               "b": "MATCH (a:Answer)-[:ANSWERS]->(q:Question) WHERE q.id IN $a RETURN DISTINCT a.id AS id"}
    for bs in (1000, 97):
        r = Executor(clients, batch_size=bs, cap=H).run(plan, queries)
        a, b = r.records["a"], r.records["b"]
        assert a.status is Status.COMPLETE and len(a.value) == len(bound), \
            f"score >= {s_min}: the sql step returned {a.value and len(a.value)} questions, the oracle has {len(bound)}"
        assert b.failed_batches == 0, (s_min, bs, b.error)
        assert b.batches == -(-len(bound) // bs) and not b.cap_hit and b.status is Status.COMPLETE
        assert r.answer.exact and set(r.answer.value) == gold, \
            f"score >= {s_min}, batch {bs}: {len(gold - set(r.answer.value))} of {len(gold)} answers missing, " \
            f"{len(set(r.answer.value) - gold)} extra ({invisible} of {len(bound)} bound questions have no loaded answer)"


def test_d07_count_over_a_bound_set_above_1000_ids_matches_the_oracle(clients, ctx):
    badges = _pick(_badge_holders(ctx), *BIG, 1)
    if not badges:
        pytest.fail(f"no badge with {BIG[0]} to {BIG[1]} holders in the oracle")
    r = _run_d04(clients, ctx, badges[0], 1000, template=D07)
    assert r.answer.exact and r.answer.value == d07_gold(ctx, {"badge": badges[0]}).value


def test_many_small_batches_on_the_live_stores_match_the_oracle(clients, ctx):
    badges = _pick(_badge_holders(ctx), *MID, 1)
    if not badges:
        pytest.fail(f"no badge with {MID[0]} to {MID[1]} holders in the oracle")
    r = _run_d04(clients, ctx, badges[0], SMALL_BATCH)
    assert r.records["b"].batches >= -(-MID[0] // SMALL_BATCH)
    assert r.answer.exact and set(r.answer.value) == set(d04_gold(ctx, {"badge": badges[0]}).value)


# ---------------------------------------------------------------- 3: D01 with the bind planner's restriction

def test_d01_policy_restriction_across_batches_matches_the_oracle(clients, ctx):
    tags = _pick(_askers_by_tag(ctx), *BIG, 2)
    if not tags:
        pytest.fail(f"no tag with {BIG[0]} to {BIG[1]} askers in the oracle")
    rep = _reputation(ctx)
    for tag in tags:
        askers = _askers_by_tag(ctx)[tag]
        reps = sorted((rep[u] for u in askers if u in rep), reverse=True)
        r_min = reps[9]                                          # the 10th best asker's reputation
        params = {"tag": tag, "min_reputation": r_min}
        gold = set(d01_gold(ctx, params).value)
        q = reference_queries("D01", params)
        restricted = restrict(_plan(D01), "b", "a")
        q_restricted = dict(q, b=f"SELECT id FROM users WHERE reputation >= {sql_lit(r_min)} AND id = ANY(%(a)s)")
        for bs in (1000, 97):
            r = Executor(clients, batch_size=bs, cap=H).run(restricted, q_restricted)
            b = r.records["b"]
            assert b.bind_origin == BIND_ORIGIN_POLICY and b.outside_bound == 0, (tag, bs, b.outside_bound)
            assert b.batches == -(-len(askers) // bs) and not b.cap_hit and b.status is Status.COMPLETE
            assert r.answer.exact and set(r.answer.value) == gold, (tag, bs)
        ind = Executor(clients, cap=H).run(_plan(D01), q)
        if _users_at_least(ctx, r_min) <= H:
            assert ind.answer.exact and set(ind.answer.value) == gold, tag
        else:                                                   # the sql side is truncated: never EXACT
            assert ind.records["b"].cap_hit and not ind.answer.exact, tag


# ---------------------------------------------------------------- 4: the cap on each store

CAP_STEPS = {
    Store.SQL: ("SELECT id FROM users", "user", None),
    Store.GRAPH: ("MATCH (u:User) RETURN u.id AS id", "user", None),
    Store.DOC: (json.dumps({"query": {"bool": {"filter": [{"term": {"doc_type": "question"}}]}}}), "question",
                Mode.FILTER),
}


@pytest.mark.parametrize("store", list(CAP_STEPS), ids=lambda s: s.value)
def test_a_step_above_the_cap_stops_at_the_cap_with_a_subset_label(clients, store):
    q, out, mode = CAP_STEPS[store]
    plan = Plan((Step("s1", Op.RETRIEVE, store, mode, "every entity of the type", output=out),))
    r = Executor(clients, cap=H).run(plan, {"s1": q})
    rec = r.records["s1"]
    assert rec.cap_hit and rec.status is Status.PARTIAL and rec.guarantee is Guarantee.SUBSET, \
        (store.value, rec.cap_hit, rec.status, rec.guarantee, rec.error)
    assert len(rec.value) == H and not r.answer.exact


# ---------------------------------------------------------------- 5: whole tables at the validator's caps

WHOLE = {
    "sql users": (Store.SQL, "SELECT id FROM users WHERE id > 0", "user", None, "v_users"),
    "graph users": (Store.GRAPH, "MATCH (u:User) WHERE u.id > 0 RETURN u.id AS id", "user", None, "v_users"),
    "doc questions": (Store.DOC, json.dumps({"query": {"bool": {"filter": [{"term": {"doc_type": "question"}}]}}}),
                      "question", Mode.FILTER, "v_questions"),
    "doc answers": (Store.DOC, json.dumps({"query": {"bool": {"filter": [{"term": {"doc_type": "answer"}}]}}}),
                    "answer", Mode.FILTER, "v_answers"),
}


@pytest.mark.parametrize("case", list(WHOLE))
def test_a_whole_table_at_the_validator_cap_is_complete_and_equals_the_oracle(clients, ctx, case):
    import time
    store, q, out, mode, view = WHOLE[case]
    caps = validator_caps(ctx.db)
    want = {int(i) for (i,) in ctx.db.execute(f"SELECT id FROM {view}")}
    assert len(want) == caps[out]
    plan = Plan((Step("s1", Op.RETRIEVE, store, mode, "every entity of the type", output=out),))
    t0 = time.perf_counter()
    r = Executor(clients, cap=max(caps.values()), step_cap=lambda st: caps[st.output]).run(plan, {"s1": q})
    secs = time.perf_counter() - t0
    rec = r.records["s1"]
    print(f"{case}: {len(rec.value or ())} ids of {caps[out]} in {secs:.1f} s, cap_hit {rec.cap_hit}")
    assert not rec.cap_hit and rec.status is Status.COMPLETE and rec.guarantee is Guarantee.EXACT, \
        (case, rec.cap_hit, rec.status, rec.error)
    got = {int(x) for x in rec.value}          # identifiers compared as integers (the document store's are JSON)
    assert got == want, f"{case}: {len(got - want)} ids not in the oracle, {len(want - got)} oracle ids missing"
