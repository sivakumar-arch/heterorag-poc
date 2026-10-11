"""Executor: dependent execution, batching, caps, failures, guarantees, premise checks."""
import random
import threading
import time

import pytest

from heterorag.federation import (Bind, Executor, FetchResult, Guarantee, Kind, Mode, Op, Plan,
                                  PlanError, Step, Store)
from heterorag.federation import guarantee as G
from heterorag.federation.answer import MAX_PARTIAL_SET


class Fake:
    """In-memory store client that honours the StoreClient contract (limit, exhausted)."""

    def __init__(self, rows_fn, *, delay=0.0, fault=None):
        self.rows_fn = rows_fn          # (step, query, binds) -> full true result (list)
        self.delay = delay
        self.fault = fault              # (call_no, step, binds) -> None | "raise" | ("partial", n)
        self.calls = []
        self._lock = threading.Lock()

    def fetch(self, step, query, binds, limit):
        with self._lock:
            n = len(self.calls)
            self.calls.append((step.id, query, {k: list(v) for k, v in binds.items()}, limit))
        if self.delay:
            time.sleep(self.delay)
        mode = self.fault(n, step, binds) if self.fault else None
        if mode == "raise":
            raise ConnectionError("store unavailable")
        full = list(self.rows_fn(step, query, binds))
        values, exhausted = full[:limit], len(full) <= limit
        if isinstance(mode, tuple) and mode[0] == "partial":
            return FetchResult(values[:mode[1]], exhausted=False, error="timeout")
        return FetchResult(values, exhausted)


def retrieve(sid, store, entity="question", **kw):
    mode = Mode.FILTER if store is Store.DOC else None
    return Step(sid, Op.RETRIEVE, store, mode, "q", output=entity, **kw)


def bound(sid, store, src, src_entity, entity="question", **kw):
    return Step(sid, Op.RETRIEVE, store, None, "q", inputs=(src,), bind=Bind(src, src_entity),
                output=entity, **kw)


def p01_plan(op=Op.INTERSECT):
    return Plan((retrieve("s1", Store.GRAPH), retrieve("s2", Store.DOC),
                 Step("s3", op, inputs=("s1", "s2"), output="question")))


def run(plan, clients, queries=None, **kw):
    q = queries or {s.id: f"NATIVE {s.id}" for s in plan.steps if s.op is Op.RETRIEVE}
    return Executor(clients, **kw).run(plan, q)


def table(mapping, default=()):
    return lambda step, query, binds: mapping.get(step.id, default)


# ------------------------------------------------------------------ independent

def test_independent_intersection_is_exact():
    g, d = Fake(table({"s1": [1, 2, 3]})), Fake(table({"s2": [2, 3, 4]}))
    r = run(p01_plan(), {Store.GRAPH: g, Store.DOC: d})
    assert r.answer.type == "ids" and r.answer.value == [2, 3] and r.answer.exact
    assert r.records["s3"].guarantee is Guarantee.EXACT
    assert r.totals()["calls"] == 2 and r.totals()["rows_read"] == 6


def test_empty_exact_result_is_an_answer_not_an_abstention():
    r = run(p01_plan(), {Store.GRAPH: Fake(table({"s1": [1]})), Store.DOC: Fake(table({"s2": [2]}))})
    assert r.answer.type == "ids" and r.answer.value == [] and r.answer.exact


def test_independent_steps_run_concurrently():
    g, d = Fake(table({"s1": [1]}), delay=0.25), Fake(table({"s2": [1]}), delay=0.25)
    t0 = time.perf_counter()
    r = run(p01_plan(), {Store.GRAPH: g, Store.DOC: d})
    assert time.perf_counter() - t0 < 0.45 and r.answer.value == [1]


# ------------------------------------------------------------------ dependent

OWNS = {10: [100, 101], 11: [101, 102], 12: [], 13: [103]}


def owner_rows(step, query, binds):
    if step.id == "s1":
        return [10, 11, 12, 13]
    return [q for u in binds["s1"] for q in OWNS[u]]


def dep_plan():
    return Plan((retrieve("s1", Store.SQL, "user"), bound("s2", Store.GRAPH, "s1", "user")))


def test_dependent_execution_passes_ids_in_batches_by_code():
    sql, graph = Fake(owner_rows), Fake(owner_rows)
    r = run(dep_plan(), {Store.SQL: sql, Store.GRAPH: graph}, batch_size=3)
    assert r.answer.value == [100, 101, 102, 103] and r.answer.exact
    rec = r.records["s2"]
    assert rec.batches == 2 and rec.calls == 2 and rec.failed_batches == 0
    # identifiers travel as parameters, never inside the query text
    sent = [c for c in graph.calls]
    assert [c[2]["s1"] for c in sent] == [[10, 11, 12], [13]]
    assert all(c[1] == "NATIVE s2" for c in sent)


def test_empty_bound_set_makes_no_call_and_is_exact():
    sql = Fake(lambda *a: [])
    graph = Fake(owner_rows)
    r = run(dep_plan(), {Store.SQL: sql, Store.GRAPH: graph})
    assert graph.calls == [] and r.answer.exact and r.answer.value == []


def test_bound_step_inherits_input_guarantee():
    # the SQL read stops early (timeout after two rows): the bound step itself reads
    # everything it is given, but its result is only a subset of the true answer
    sql = Fake(owner_rows, fault=lambda n, s, b: ("partial", 2))
    graph = Fake(owner_rows)
    r = run(dep_plan(), {Store.SQL: sql, Store.GRAPH: graph})
    assert r.records["s1"].guarantee is Guarantee.SUBSET
    assert r.records["s2"].status.value == "complete"
    assert r.records["s2"].guarantee is Guarantee.SUBSET
    assert r.answer.guarantee is Guarantee.SUBSET and r.answer.value == [100, 101, 102]


def test_failed_bound_input_stops_the_chain():
    sql = Fake(owner_rows, fault=lambda n, s, b: "raise")
    graph = Fake(owner_rows)
    r = run(dep_plan(), {Store.SQL: sql, Store.GRAPH: graph})
    assert r.records["s1"].status.value == "failed" and graph.calls == []
    assert r.answer.abstained and r.answer.reason == "incomplete"


def test_failed_batch_gives_a_sound_subset():
    # three batches of one id; the second batch fails (and its retry fails)
    def fault(n, step, binds):
        return "raise" if step.id == "s2" and binds["s1"] == [11] else None
    graph = Fake(owner_rows, fault=fault)
    r = run(dep_plan(), {Store.SQL: Fake(owner_rows), Store.GRAPH: graph}, batch_size=1)
    rec = r.records["s2"]
    assert rec.failed_batches == 1 and rec.status.value == "partial"
    assert rec.guarantee is Guarantee.SUBSET
    # the retry made the failed batch cost two calls
    assert rec.calls == 5
    assert r.answer.value == [100, 101, 103] and r.answer.guarantee is Guarantee.SUBSET


def test_single_retry_recovers_a_transient_failure():
    state = {"n": 0}

    def fault(n, step, binds):
        state["n"] += 1
        return "raise" if state["n"] == 1 else None
    r = run(p01_plan(), {Store.GRAPH: Fake(table({"s1": [1, 2]}), fault=fault),
                         Store.DOC: Fake(table({"s2": [2]}))})
    assert r.answer.exact and r.answer.value == [2] and r.records["s1"].calls == 2


def test_result_cap_stops_reading_and_marks_partial():
    sql, graph = Fake(owner_rows), Fake(owner_rows)
    r = run(dep_plan(), {Store.SQL: sql, Store.GRAPH: graph}, batch_size=1, cap=2)
    rec = r.records["s2"]
    assert rec.status.value == "partial" and "cap" in (rec.error or "")


# ------------------------------------------------------------------ combine guarantees

def partial_first(n, step, binds):
    return ("partial", 1) if step.id == "s1" else None


def test_partial_intersection_branch_is_a_subset_answer():
    g = Fake(table({"s1": [1, 2, 3]}), fault=partial_first)
    r = run(p01_plan(), {Store.GRAPH: g, Store.DOC: Fake(table({"s2": [1, 2, 3, 4]}))})
    assert r.answer.guarantee is Guarantee.SUBSET and r.answer.value == [1]


def test_partial_subtrahend_gives_a_superset_answer():
    plan = p01_plan(Op.SUBTRACT)
    g = Fake(table({"s1": [1, 2, 3, 4]}))
    d = Fake(table({"s2": [1, 2, 3]}), fault=lambda n, s, b: ("partial", 1))
    r = run(plan, {Store.GRAPH: g, Store.DOC: d})
    assert r.answer.guarantee is Guarantee.SUPERSET and r.answer.value == [2, 3, 4]


def test_superset_larger_than_the_cap_abstains():
    n = MAX_PARTIAL_SET + 5
    plan = p01_plan(Op.SUBTRACT)
    d = Fake(table({"s2": [0]}), fault=lambda n_, s, b: ("partial", 1))
    r = run(plan, {Store.GRAPH: Fake(table({"s1": list(range(1, n + 1))})), Store.DOC: d})
    assert r.answer.abstained and r.answer.reason == "incomplete"


def test_mixed_subset_and_superset_abstains():
    plan = Plan((retrieve("s1", Store.GRAPH), retrieve("s2", Store.DOC), retrieve("s3", Store.SQL),
                 Step("s4", Op.SUBTRACT, inputs=("s1", "s2"), output="question"),
                 Step("s5", Op.UNION, inputs=("s4", "s3"), output="question")))
    g = Fake(table({"s1": [1, 2, 3]}), fault=partial_first)         # subset
    d = Fake(table({"s2": [2]}), fault=lambda n, s, b: ("partial", 1))   # partial subtrahend: superset
    r = run(plan, {Store.GRAPH: g, Store.DOC: d, Store.SQL: Fake(table({"s3": [9]}))})
    # s4 = subset minus partial = NONE, so the union has no guarantee
    assert r.answer.abstained


def test_count_over_partial_set_is_a_lower_bound():
    plan = Plan((retrieve("s1", Store.GRAPH),
                 Step("c", Op.COUNT, inputs=("s1",), output="number", kind=Kind.SCALAR)))
    g = Fake(table({"s1": [1, 2, 3]}), fault=lambda n, s, b: ("partial", 2))
    r = run(plan, {Store.GRAPH: g})
    assert r.answer.type == "scalar" and r.answer.value == 2 and r.answer.guarantee is Guarantee.LOWER


def test_exact_count():
    plan = Plan((retrieve("s1", Store.GRAPH),
                 Step("c", Op.COUNT, inputs=("s1",), output="number", kind=Kind.SCALAR)))
    r = run(plan, {Store.GRAPH: Fake(table({"s1": [1, 2, 3]}))})
    assert r.answer.value == 3 and r.answer.exact


def test_subset_answer_that_is_empty_abstains():
    # a partial read that found nothing relevant cannot support a useful answer
    g = Fake(table({"s1": [7, 8]}), fault=partial_first)
    d = Fake(table({"s2": [1]}))
    r = run(p01_plan(), {Store.GRAPH: g, Store.DOC: d})
    assert r.answer.abstained and r.answer.reason == "incomplete"


# ------------------------------------------------------------------ scalar, ranked, premise

def test_scalar_and_ranked_steps():
    sc = Plan((Step("s1", Op.RETRIEVE, Store.SQL, None, "q", output="number", kind=Kind.SCALAR),))
    r = run(sc, {Store.SQL: Fake(table({"s1": [4242]}))})
    assert r.answer.type == "scalar" and r.answer.value == 4242 and r.answer.exact
    bad = run(sc, {Store.SQL: Fake(table({"s1": [1, 2]}))})
    assert bad.answer.abstained
    rk = Plan((Step("s1", Op.RETRIEVE, Store.GRAPH, None, "q", output="tag", kind=Kind.RANKED, k=3),))
    r = run(rk, {Store.GRAPH: Fake(table({"s1": ["b", "a", "c"]}))})
    assert r.answer.type == "ranked" and r.answer.value == ["b", "a", "c"]
    part = run(rk, {Store.GRAPH: Fake(table({"s1": ["b", "a", "c"]}), fault=lambda *a: ("partial", 2))})
    assert part.answer.abstained


def test_scalar_over_a_bound_set_needs_an_exact_input():
    plan = Plan((retrieve("s1", Store.SQL, "user"),
                 Step("s2", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("s1",),
                      bind=Bind("s1", "user"), output="number", kind=Kind.SCALAR)))
    clients = {Store.SQL: Fake(lambda *a: [1, 2, 3]), Store.GRAPH: Fake(lambda *a: [3])}
    assert run(plan, clients).answer.value == 3
    inexact = run(plan, {Store.SQL: Fake(lambda *a: [1, 2, 3]), Store.GRAPH: clients[Store.GRAPH]}, cap=2)
    assert inexact.answer.abstained


def test_failed_premise_check_abstains_with_its_own_reason():
    plan = Plan((retrieve("s1", Store.SQL, "user", must_exist=True), bound("s2", Store.GRAPH, "s1", "user")))
    r = run(plan, {Store.SQL: Fake(lambda *a: []), Store.GRAPH: Fake(owner_rows)})
    assert r.answer.abstained and r.answer.reason == "premise_failed"
    ok = run(plan, {Store.SQL: Fake(lambda *a: [10]), Store.GRAPH: Fake(owner_rows)})
    assert ok.answer.exact and ok.answer.value == [100, 101]


def test_inexact_empty_premise_is_not_a_premise_failure():
    plan = Plan((retrieve("s1", Store.SQL, "user", must_exist=True),))
    r = run(plan, {Store.SQL: Fake(lambda *a: [], fault=lambda *a: "raise")})
    assert r.answer.reason == "incomplete"


# ------------------------------------------------------------------ robustness

def test_missing_query_or_client_fails_the_step_not_the_run():
    plan = p01_plan()
    r = Executor({Store.GRAPH: Fake(table({"s1": [1]}))}).run(plan, {"s1": "q"})
    assert r.records["s2"].status.value == "failed" and r.answer.abstained
    r = Executor({Store.GRAPH: Fake(table({"s1": [1]})), Store.DOC: Fake(table({"s2": [1]}))}) \
        .run(plan, {"s1": "q", "s2": "  "})
    assert r.answer.abstained


def test_invalid_plan_is_rejected_before_any_call():
    g = Fake(table({}))
    bad = Plan((retrieve("s1", Store.GRAPH), retrieve("s2", Store.DOC)))
    with pytest.raises(PlanError):
        run(bad, {Store.GRAPH: g})
    assert g.calls == []


def test_records_serialise():
    r = run(p01_plan(), {Store.GRAPH: Fake(table({"s1": [1, 2]})), Store.DOC: Fake(table({"s2": [2]}))})
    d = [rec.to_dict() for rec in r.records.values()]
    assert d[0]["step"] == "s1" and d[2]["guarantee"] == "exact" and r.answer.to_dict()["type"] == "ids"


# ------------------------------------------------------------------ end-to-end soundness

@pytest.mark.parametrize("seed", range(40))
def test_random_plans_always_satisfy_their_guarantee(seed):
    """Random data, random faults and caps. Whatever guarantee the answer carries must hold
    against the true answer computed directly with set algebra."""
    rng = random.Random(seed)
    users = list(range(1, 25))
    A = set(rng.sample(users, rng.randint(0, 20)))                 # sql: users
    own = {u: set(rng.sample(range(100, 160), rng.randint(0, 6))) for u in users}
    D = set(rng.sample(range(100, 160), rng.randint(0, 40)))      # doc: questions
    E = set(rng.sample(range(100, 160), rng.randint(0, 40)))      # graph: questions

    def sql(step, q, b): return sorted(A)
    def graph_bound(step, q, b): return sorted({x for u in b["s1"] for x in own[u]})
    def doc(step, q, b): return sorted(D)
    def graph_all(step, q, b): return sorted(E)

    final_op = rng.choice([Op.SUBTRACT, Op.UNION, Op.INTERSECT, "three-way"])
    final = (Step("s5", Op.INTERSECT, inputs=("s2", "s3", "s4"), output="question")
             if final_op == "three-way"
             else Step("s5", final_op, inputs=("s2", "s4"), output="question"))
    steps = [retrieve("s1", Store.SQL, "user"), bound("s2", Store.GRAPH, "s1", "user"),
             retrieve("s4", Store.GRAPH), final]
    if final_op == "three-way":                      # no dangling step: s3 only when it is used
        steps.insert(2, retrieve("s3", Store.DOC))
    plan = Plan(tuple(steps))
    truth_s2 = {x for u in A for x in own[u]}
    truth = {Op.SUBTRACT: truth_s2 - E, Op.UNION: truth_s2 | E, Op.INTERSECT: truth_s2 & E,
             "three-way": truth_s2 & D & E}[final_op]

    def fault(n, step, binds):
        r = rng.random()
        return "raise" if r < 0.12 else (("partial", rng.randint(1, 4)) if r < 0.3 else None)

    clients = {Store.SQL: Fake(sql, fault=fault), Store.DOC: Fake(doc, fault=fault),
               Store.GRAPH: Fake(lambda s, q, b: graph_bound(s, q, b) if s.id == "s2" else graph_all(s, q, b),
                                 fault=fault)}
    r = run(plan, clients, batch_size=rng.randint(1, 8), cap=rng.randint(3, 60), retries=rng.randint(0, 1))
    a = r.answer
    if not a.abstained:
        assert G.holds(a.guarantee, set(a.value), truth), (seed, a.guarantee, a.value, sorted(truth))
