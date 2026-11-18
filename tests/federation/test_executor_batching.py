"""Batch 100 (build item 1): executor behaviour that Proposition 1 relies on, with fake stores.

- the results of the batches of one bound step are accumulated as a set: duplicates across batches are removed,
  and for a pointwise additive step the result does not depend on the batch size (Formal Core, PM);
- a scalar step with a bind fails explicitly over an empty bound set and over a bound set larger than one batch,
  without calling the store;
- a bind into a ranked step is refused by plan validation before any store is called.
Existing tests already cover basic batching, the empty bound set of a set step and the inexact-input scalar guard
(test_executor.py); the same Fake client is used here.
"""
import random

import pytest

from heterorag.federation import Bind, Executor, Op, Plan, PlanError, Step, Store
from heterorag.federation import guarantee as G
from heterorag.federation.guarantee import Guarantee, Kind, Status
from tests.federation.test_executor import Fake, retrieve

# user -> tags of the questions the user asked; users in different batches share tags
OWN = {1: {"a", "b"}, 2: {"b", "c"}, 3: {"c", "d"}, 4: {"a", "d"}, 5: {"e"}, 6: {"b", "e"}, 7: {"a"}}


def users_to_tags(users):
    """s1 (sql users) -> s2 (graph tags of these users' questions), optional count s3."""
    steps = [retrieve("s1", Store.SQL, "user"),
             Step("s2", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("s1",), bind=Bind("s1", "user"), output="tag")]
    return Plan(tuple(steps)), Plan(tuple(steps + [Step("s3", Op.COUNT, inputs=("s2",), output="number",
                                                         kind=Kind.SCALAR)]))


def clients(users, own=OWN):
    sql = Fake(lambda s, q, b: sorted(users))
    graph = Fake(lambda s, q, b: sorted({t for u in b["s1"] for t in own[u]}))
    return {Store.SQL: sql, Store.GRAPH: graph}


def test_union_over_batches_removes_duplicates_and_the_count_sees_the_union():
    users = [1, 2, 3, 4, 5, 6, 7]
    truth = sorted({t for u in users for t in OWN[u]})               # a b c d e
    set_plan, count_plan = users_to_tags(users)
    c = clients(users)
    r = Executor(c, batch_size=2).run(set_plan, {"s1": "Q1", "s2": "Q2"})
    rec = r.records["s2"]
    assert rec.batches == 4 and rec.status is Status.COMPLETE and rec.guarantee is Guarantee.EXACT
    assert sorted(r.answer.value) == truth
    # the store returned 'a' and 'b' in more than one batch; the step keeps each once
    returned = [t for call in c[Store.GRAPH].calls for u in call[2]["s1"] for t in OWN[u]]
    assert len(returned) > len(truth)
    rc = Executor(clients(users), batch_size=2).run(count_plan, {"s1": "Q1", "s2": "Q2"})
    assert rc.answer.exact and rc.answer.value == len(truth)


@pytest.mark.parametrize("seed", range(25))
def test_pointwise_additive_step_does_not_depend_on_the_batch_size(seed):
    rng = random.Random(seed)
    users = sorted(rng.sample(range(1, 60), rng.randint(1, 40)))
    own = {u: {f"t{rng.randint(0, 25)}" for _ in range(rng.randint(0, 4))} for u in range(1, 60)}
    truth = sorted({t for u in users for t in own[u]})
    set_plan, _ = users_to_tags(users)
    results = set()
    for bs in sorted({1, 2, 3, 7, len(users), len(users) + 5, rng.randint(1, 50)}):
        r = Executor(clients(users, own), batch_size=bs).run(set_plan, {"s1": "Q1", "s2": "Q2"})
        assert r.answer.exact and sorted(r.answer.value) == truth, (seed, bs)
        assert r.records["s2"].batches == -(-len(users) // bs)
        results.add(tuple(sorted(r.answer.value)))
    assert len(results) == 1


def _scalar_bound(src_rows):
    plan = Plan((retrieve("s1", Store.SQL, "user"),
                 Step("s2", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("s1",), bind=Bind("s1", "user"),
                      output="number", kind=Kind.SCALAR)))
    sql, graph = Fake(lambda *a: src_rows), Fake(lambda *a: [42])
    return plan, {Store.SQL: sql, Store.GRAPH: graph}


def test_scalar_over_an_empty_bound_set_fails_explicitly_without_a_call():
    plan, c = _scalar_bound([])
    r = Executor(c).run(plan, {"s1": "Q1", "s2": "Q2"})
    s2 = r.records["s2"]
    assert s2.status is Status.FAILED and s2.guarantee is Guarantee.NONE and "empty bound set" in s2.error
    assert c[Store.GRAPH].calls == [] and r.answer.abstained


def test_scalar_over_a_bound_set_larger_than_one_batch_fails_explicitly_without_a_call():
    plan, c = _scalar_bound([1, 2, 3, 4, 5])
    r = Executor(c, batch_size=2).run(plan, {"s1": "Q1", "s2": "Q2"})
    s2 = r.records["s2"]
    assert s2.status is Status.FAILED and "larger than one batch" in s2.error
    assert c[Store.GRAPH].calls == [] and r.answer.abstained
    ok = Executor(_scalar_bound([1, 2])[1], batch_size=2).run(plan, {"s1": "Q1", "s2": "Q2"})
    assert ok.answer.exact and ok.answer.value == 42            # one batch: the scalar runs


def test_a_bind_into_a_ranked_step_is_refused_before_any_store_is_called():
    plan = Plan((retrieve("s1", Store.SQL, "user"),
                 Step("s2", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("s1",), bind=Bind("s1", "user"),
                      output="tag", kind=Kind.RANKED, k=3)))
    c = clients([1, 2])
    with pytest.raises(PlanError) as ex:
        Executor(c).run(plan, {"s1": "Q1", "s2": "Q2"})
    assert any("ranked step is not allowed" in e for e in ex.value.args[0])
    assert c[Store.SQL].calls == [] and c[Store.GRAPH].calls == []


def test_overlapping_batches_near_the_cap_stay_sound():
    """Documents a conservative behaviour: each batch may read only cap minus the ids already held, counted
    before duplicates are removed, so overlapping batches can stop the step at the cap although the union would
    fit. The label then says SUBSET, and SUBSET holds (sound); an exact answer was possible."""
    own = {1: {"a", "b"}, 2: {"c"}, 3: {"c", "d"}, 4: {"e"}}
    users = [1, 2, 3, 4]
    truth = {"a", "b", "c", "d", "e"}                               # five tags, the cap is five
    set_plan, _ = users_to_tags(users)
    r = Executor(clients(users, own), batch_size=2, cap=5).run(set_plan, {"s1": "Q1", "s2": "Q2"})
    rec = r.records["s2"]
    assert rec.cap_hit and rec.status is Status.PARTIAL and rec.guarantee is Guarantee.SUBSET
    assert G.holds(rec.guarantee, set(rec.value), truth) and set(rec.value) < truth
