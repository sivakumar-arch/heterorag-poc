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
     Because D04's answer (tags) saturates, a bound step without overlap is tested as well: the questions asked
     by the holders of the same badge (one asker per question, so every lost id changes the answer).
  2. The same plan with a batch size of 7 on a badge with 101 to 300 holders (many batches on a small set).
  3. D01 with the bind planner's restriction (sql reputation filter restricted to the graph askers of a tag
     with more than 1,000 askers): exact, contained (C5 counter 0), bind origin "policy"; and the independent
     plan agrees when the sql side fits the cap, or is labelled non-exact when it does not.
  4. The cap on each store: a step known to exceed 10,000 rows stops at the cap with ``cap_hit``, status
     partial and label SUBSET.
Environment: the store variables of ``live_clients`` (PG_DSN, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
ES_URL, with the local defaults) and HETERORAG_ORACLE (default data/oracle_stats.sqlite).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmark_v2 import union
from benchmark_v2.catalogue_d import (D01, D04, D07, _askers_by_tag, _badge_holders, _questions_by_user,
                                      _reputation, _users_at_least, d01_gold, d04_gold, d07_gold)
from benchmark_v2.questions import Ctx, step_record
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
    """D04's answer (tags) saturates: holders share tags, so a lost batch need not change it. The questions asked
    by the holders do not overlap (one asker per question), so here every lost id changes the answer. The badge is
    the smallest qualifying one whose holders' questions fit the cap: above the cap a truncated answer would also
    be a subset of the gold, and a lost id could hide behind the cap."""
    by_user, holders_of = _questions_by_user(ctx), _badge_holders(ctx)
    asked = {b: {q for u in us for q in by_user.get(u, ())} for b, us in holders_of.items() if BIG[0] <= len(us) <= BIG[1]}
    fit = sorted((b for b, qs in asked.items() if 0 < len(qs) <= H), key=lambda b: (len(holders_of[b]), b))
    if not fit:                     # without a set that fits the cap, a lost id could hide behind the cap
        pytest.fail(f"no badge with {BIG[0]} to {BIG[1]} holders whose questions fit the cap {H}")
    badges = fit[:1]
    holders, gold = holders_of[badges[0]], asked[badges[0]]
    plan = Plan((Step("a", Op.RETRIEVE, Store.SQL, None, "users holding the badge", output="user"),
                 Step("b", Op.RETRIEVE, Store.GRAPH, None, "questions asked by these users", inputs=("a",),
                      bind=Bind("a", "user"), output="question")))
    queries = {"a": reference_queries("D04", {"badge": badges[0]})["a"],
               "b": "MATCH (u:User)-[:ASKED]->(q:Question) WHERE u.id IN $a RETURN DISTINCT q.id AS id"}
    for bs in (1000, 97):
        r = Executor(clients, batch_size=bs, cap=H).run(plan, queries)
        b = r.records["b"]
        assert b.failed_batches == 0, (badges[0], bs, b.error)
        assert b.batches == -(-len(holders) // bs) and not b.cap_hit and b.status is Status.COMPLETE
        assert r.answer.exact and set(r.answer.value) == gold, \
            f"badge {badges[0]!r}, batch {bs}: {len(gold - set(r.answer.value))} of {len(gold)} questions missing, " \
            f"{len(set(r.answer.value) - gold)} extra"


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
