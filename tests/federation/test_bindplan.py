"""Design B: probes, the bind decision (eligibility, feasibility, profitability), the restricted
plan, the pipeline integration, and Proposition 2 checked against brute force."""
import itertools
import random
import threading

import pytest

from heterorag.federation import (Bind, Estimate, Executor, FetchResult, Guarantee, Kind, Mode, Op, Plan,
                                  Step, Store)
from heterorag.federation.bindplan import BindPolicy, CostModel, Decision, apply, decide, restrict
from heterorag.federation.llm_config import Usage
from heterorag.federation.pipeline import Federation
from heterorag.federation.planner import PlanResult
from heterorag.federation.translator import TranslationResult

COST = CostModel(t_call=0.02, t_row=1e-4, batch=1000, cap=10_000)


def r(sid, store=Store.GRAPH, entity="question", **kw):
    mode = Mode.FILTER if store is Store.DOC else None
    return Step(sid, Op.RETRIEVE, store, mode, "q", output=entity, **kw)


def combine(op, *ins, sid="c", entity="question"):
    return Step(sid, op, inputs=tuple(ins), output=entity)


def plan2(op=Op.INTERSECT):
    return Plan((r("a", Store.GRAPH), r("b", Store.DOC), combine(op, "a", "b")))


def est(**sizes):
    return {k: Estimate(v if isinstance(v, int) else v[0], "count" if isinstance(v, int) else v[1])
            for k, v in sizes.items()}


def one(plan, estimates, policy=BindPolicy.COST, cost=COST):
    ds = decide(plan, estimates, policy, cost)
    assert len(ds) == 1, ds
    return ds[0]


# ------------------------------------------------------------------ level 1: eligibility

def test_union_is_never_restricted():
    d = one(plan2(Op.UNION), est(a=50, b=50_000))
    assert d.action == "independent" and d.level == "eligibility" and "union" in d.reason


def test_subtract_restricts_only_the_subtrahend():
    # a minus b: b may be restricted to a, a may not be restricted to b
    d = one(plan2(Op.SUBTRACT), est(a=100, b=50_000))
    assert (d.action, d.step, d.source) == ("bind", "b", "a")
    d = one(plan2(Op.SUBTRACT), est(a=50_000, b=100))          # the cheap direction is the wrong one
    assert d.action == "independent" and d.step == "b" and d.source == "a"


@pytest.mark.parametrize("kw,reason", [
    (dict(kind=Kind.RANKED, k=5), "ranked"),
    (dict(kind=Kind.SCALAR), "scalar"),
])
def test_only_pure_filters_are_restricted(kw, reason):
    p = Plan((r("a"), r("b", **kw), combine(Op.INTERSECT, "a", "b")))
    # the plan is not even valid (combine takes sets); eligibility is judged on the step itself
    from heterorag.federation.bindplan import _eligible
    assert reason in _eligible(p.by_id()["b"])


def test_search_must_exist_and_dependent_steps_are_not_eligible():
    from heterorag.federation.bindplan import _eligible
    assert "search" in _eligible(Step("b", Op.RETRIEVE, Store.DOC, Mode.SEARCH, "q", output="question", k=5))
    assert "must_exist" in _eligible(r("b", must_exist=True))
    bound = Step("b", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("a",), bind=Bind("a", "question"), output="question")
    assert "dependent" in _eligible(bound)
    assert _eligible(r("b")) is None


def test_a_sibling_that_is_not_a_retrieve_cannot_be_the_source():
    p = Plan((r("a"), r("b", Store.DOC), r("c", Store.SQL), combine(Op.UNION, "b", "c", sid="u"),
              combine(Op.INTERSECT, "a", "u", sid="i")))
    ds = decide(p, est(a=50_000, b=10, c=10), BindPolicy.COST, COST)
    inter = [d for d in ds if d.combine == "i"]
    assert inter and all(d.action == "independent" for d in inter)


def test_a_restriction_that_makes_the_plan_too_deep_is_not_made():
    # a chain of three retrieves (depth 3): restricting the independent sibling to its end would make four
    p = Plan((r("a", Store.SQL), Step("b", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("a",), bind=Bind("a", "question"),
                                      output="question"),
              Step("c", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("b",), bind=Bind("b", "question"), output="question"),
              r("d", Store.DOC), combine(Op.INTERSECT, "c", "d", sid="i")))
    assert p.depth() == 3 and p.validate() == []
    ds = decide(p, est(d=100_000), BindPolicy.COST, COST)
    assert all(d.action == "independent" for d in ds)
    assert any("plan depth 4 exceeds 3" in d.reason for d in ds)


def test_the_local_intersect_does_not_count_when_a_sibling_is_restricted_to_a_dependent_step():
    # CP1 D01: s2 asks the graph for the askers of the questions of s1, s3 is a large sql set, s4 intersects.
    # The restricted plan has the chain s1, s2, s3 (three retrieves) and the intersect: valid since the
    # intersect is local. (Whether it is chosen is decided by the rule tested below.)
    p = Plan((r("s1", Store.GRAPH), Step("s2", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("s1",),
                                         bind=Bind("s1", "question"), output="user"),
              r("s3", Store.SQL, entity="user"), combine(Op.INTERSECT, "s2", "s3", sid="s4", entity="user")))
    assert p.depth() == 2
    q = restrict(p, "s3", "s2")
    assert q.depth() == 3 and q.validate() == []


# ------------------------------------------------------------------ level 2: feasibility

def test_a_set_over_the_cap_must_be_bound_when_the_other_fits():
    d = one(plan2(), est(a=3_000, b=(10_001, "lower")))
    assert (d.action, d.level, d.step, d.source) == ("bind", "feasibility", "b", "a")


def test_nothing_fits_means_no_exact_execution_and_no_bind():
    d = one(plan2(), est(a=(10_001, "lower"), b=(10_001, "lower")))
    assert d.action == "independent" and d.level == "feasibility" and "cap" in d.reason


def test_exactly_at_the_cap_is_independent_feasible():
    d = one(plan2(), est(a=10_000, b=10_000))
    assert d.level == "profitability"                          # nothing is forced


def test_missing_estimates_keep_the_plan_independent():
    d = one(plan2(), est(a=100))
    assert d.action == "independent" and d.level == "missing"
    d = one(plan2(), {})
    assert d.action == "independent" and d.level == "missing"


# ------------------------------------------------------------------ level 3: profitability

def test_bind_when_the_source_is_small_and_the_other_side_is_large():
    d = one(plan2(), est(a=50, b=9_000))
    assert d.action == "bind" and d.level == "profitability" and (d.step, d.source) == ("b", "a")
    assert d.t_bound < d.t_independent


def test_independent_when_both_sides_are_small():
    d = one(plan2(), est(a=50, b=80))
    assert d.action == "independent" and d.level == "profitability"


def test_the_cheaper_direction_is_chosen_for_an_intersection():
    d = one(plan2(), est(a=9_000, b=50))
    assert (d.step, d.source) == ("a", "b")


def test_margin_makes_the_bind_harder():
    base = est(a=50, b=900)
    assert one(plan2(), base).action == "bind"
    assert one(plan2(), base, cost=CostModel(**{**COST.__dict__, "margin": 0.1})).action == "independent"


def test_an_empty_source_costs_nothing_to_restrict():
    d = one(plan2(), est(a=0, b=9_000))
    assert d.action == "bind"
    assert COST.t_bound(0, 0) == 0.0


def test_a_larger_source_needs_more_batches():
    assert COST.t_bound(2_500, 100) == pytest.approx(3 * COST.t_call + 100 * COST.t_row)


def test_never_and_always_policies():
    p = plan2()
    assert one(p, est(a=50, b=9_000), BindPolicy.NONE).action == "independent"
    d = one(p, {}, BindPolicy.ALWAYS)
    assert (d.action, d.step, d.source, d.level) == ("bind", "b", "a", "policy")
    d = one(plan2(Op.SUBTRACT), {}, BindPolicy.ALWAYS)
    assert (d.step, d.source) == ("b", "a")
    assert one(plan2(Op.UNION), {}, BindPolicy.ALWAYS).action == "independent"


def test_three_way_intersection_binds_one_pair_and_keeps_the_third_parallel():
    p = Plan((r("a"), r("b", Store.DOC), r("c", Store.SQL), combine(Op.INTERSECT, "a", "b", "c", sid="i")))
    ds = decide(p, est(a=40, b=9_000, c=500), BindPolicy.COST, COST)
    assert len(ds) == 1 and ds[0].action == "bind" and ds[0].source == "a"


def test_apply_makes_the_step_dependent_and_validates():
    p = plan2()
    d = one(p, est(a=50, b=9_000))
    q = apply(p, [d])
    b = q.by_id()["b"]
    assert b.bind == Bind("a", "question") and b.inputs == ("a",) and q.validate() == []
    assert restrict(p, "b", "a") == q
    assert apply(p, []) == p
    assert [[s.id for s in layer] for layer in q.layers()] == [["a"], ["b"], ["c"]]


def test_decision_serialises():
    d = Decision("b", "a", "c", "bind", "profitability", "why", 0.1234567, 0.01)
    assert d.to_dict()["t_independent"] == 0.1235


# ------------------------------------------------------------------ the executor's probe

class Probed:
    def __init__(self, sizes, fail=()):
        self.sizes, self.fail, self.calls = sizes, set(fail), []

    def estimate(self, step, query, limit):
        self.calls.append((step.id, limit))
        if step.id in self.fail:
            raise ConnectionError("down")
        n = self.sizes[step.id]
        return Estimate(min(n, limit + 1), "lower" if n > limit else "count")

    def fetch(self, step, query, binds, limit):
        return FetchResult([])


class NoEstimate:
    def fetch(self, step, query, binds, limit):
        return FetchResult([])


def test_probe_estimates_unbound_set_steps_and_skips_the_rest():
    bound = Step("d", Op.RETRIEVE, Store.SQL, None, "q", inputs=("a",), bind=Bind("a", "question"), output="question")
    ranked = Step("e", Op.RETRIEVE, Store.SQL, None, "q", output="question", kind=Kind.RANKED, k=3)
    p = Plan((r("a"), r("b", Store.DOC), r("f", Store.SQL), bound, ranked,
              combine(Op.INTERSECT, "a", "b", "f", sid="i")))
    g = Probed({"a": 5, "b": 20_000, "f": 1})
    ex = Executor({Store.GRAPH: g, Store.DOC: g, Store.SQL: NoEstimate()}, cap=10_000)
    out, ms = ex.probe(p, {s.id: "q" for s in p.steps if s.op is Op.RETRIEVE})
    assert set(out) == {"a", "b"} and out["a"] == Estimate(5, "count")
    assert out["b"].exceeds and ms >= 0
    assert {c[1] for c in g.calls} == {10_000}                 # probes are bounded by the executor's cap


def test_a_failing_probe_leaves_no_estimate_and_does_not_raise():
    g = Probed({"a": 5, "b": 7}, fail={"b"})
    ex = Executor({Store.GRAPH: g, Store.DOC: g})
    out, _ = ex.probe(plan2(), {"a": "q", "b": "q"})
    assert set(out) == {"a"}


# ------------------------------------------------------------------ pipeline integration

class StubPlanner:
    def __init__(self, plan):
        self._plan = plan

    @property
    def llm(self):
        class L: models_seen = []
        return L()

    def plan(self, question):
        return PlanResult(plan=self._plan)


class StubTranslator:
    """Every step translates to 'Q<id>' and a restricted step to 'Q<id>|bound'. ``refuse`` makes the
    restricted variant of those steps fail."""

    def __init__(self, refuse=()):
        self.refuse, self.seen = set(refuse), []

    @property
    def llm(self):
        class L: models_seen = []
        return L()

    def translate(self, step):
        self.seen.append((step.id, step.bind is not None))
        u = Usage(prompt_tokens=100, reply_tokens=10) if hasattr(Usage, "__dataclass_fields__") else Usage()
        if step.bind is not None and step.id in self.refuse:
            return TranslationResult(step.id, "", ["cannot restrict"], 1, u, raw=['{"query":\n {"match_all": {}}}'])
        return TranslationResult(step.id, f"Q{step.id}" + ("|bound" if step.bind else ""), [], 1, u)


class SetStore:
    """A store holding fixed sets per step id; honours limit, binds and probes."""

    def __init__(self, sets):
        self.sets, self.calls, self._lock = sets, [], threading.Lock()

    def estimate(self, step, query, limit):
        n = len(self.sets[step.id])
        return Estimate(min(n, limit + 1), "lower" if n > limit else "count")

    def fetch(self, step, query, binds, limit):
        with self._lock:
            self.calls.append((step.id, bool(binds)))
        full = sorted(self.sets[step.id])
        if binds:
            allowed = set(next(iter(binds.values())))
            full = [v for v in full if v in allowed]
        return FetchResult(full[:limit], len(full) <= limit)


def federation(plan, sets, policy, *, cap=10_000, batch=1000, refuse=(), cost=None):
    st = SetStore(sets)
    tr = StubTranslator(refuse)
    ex = Executor({Store.GRAPH: st, Store.DOC: st, Store.SQL: st}, cap=cap, batch_size=batch)
    return Federation(StubPlanner(plan), tr, ex, bind_policy=policy, cost=cost), st, tr


def test_policy_none_runs_the_plan_as_written():
    sets = {"a": set(range(50)), "b": set(range(25, 9_000))}
    fed, st, tr = federation(plan2(), sets, BindPolicy.NONE)
    res = fed.answer("q")
    assert res.answer.exact and set(res.answer.value) == set(range(25, 50))
    assert res.executed_plan == plan2() and not res.bind_decisions and res.estimates == {}
    assert all(not bound for _, bound in tr.seen)


def test_cost_policy_restricts_the_large_side_and_gets_the_same_answer():
    sets = {"a": set(range(50)), "b": set(range(25, 9_000))}
    fed, st, tr = federation(plan2(), sets, BindPolicy.COST)
    res = fed.answer("q")
    assert set(res.answer.value) == set(range(25, 50)) and res.answer.exact
    assert res.executed_plan.by_id()["b"].bind == Bind("a", "question")
    assert res.queries["b"] == "Qb|bound"
    assert ("b", True) in st.calls
    assert res.run.records["b"].rows_read == 25                # 25 rows instead of 8,975
    d = res.to_dict()
    assert d["bind_policy"] == "cost" and d["bind_decisions"][0]["action"] == "bind"
    assert d["estimates"]["b"] == {"n": 8_975, "basis": "count"} and d["probe_ms"] >= 0


def test_a_set_over_the_cap_is_answered_exactly_only_with_the_bind():
    sets = {"a": set(range(30)), "b": set(range(15, 50_000))}
    fed, *_ = federation(plan2(), sets, BindPolicy.NONE, cap=10_000)
    none = fed.answer("q")
    assert none.answer.guarantee is not Guarantee.EXACT        # truncated by the cap
    fed, *_ = federation(plan2(), sets, BindPolicy.COST, cap=10_000)
    res = fed.answer("q")
    assert res.answer.exact and set(res.answer.value) == set(range(15, 30))
    assert res.bind_decisions[0].level == "feasibility"


def test_a_failed_restricted_translation_falls_back_to_the_independent_plan():
    sets = {"a": set(range(50)), "b": set(range(25, 9_000))}
    fed, st, tr = federation(plan2(), sets, BindPolicy.COST, refuse={"b"})
    res = fed.answer("q")
    assert res.executed_plan == plan2() and res.queries["b"] == "Qb"
    assert res.bind_fallbacks and "cannot restrict" in res.bind_fallbacks[0]
    assert '| last reply: {"query": {"match_all": {}}}' in res.bind_fallbacks[0]   # whitespace collapsed
    assert set(res.answer.value) == set(range(25, 50))
    assert len(res.discarded) == 1                              # tokens of the failed attempt are counted
    assert res.translator_usage.prompt_tokens == 100 * (2 + 1)  # a, b, and the failed restricted b


def test_translator_tokens_of_a_replaced_translation_are_still_counted():
    sets = {"a": set(range(50)), "b": set(range(25, 9_000))}
    fed, st, tr = federation(plan2(), sets, BindPolicy.COST)
    res = fed.answer("q")
    assert len(res.discarded) == 1 and res.translator_usage.prompt_tokens == 300


def test_always_policy_needs_no_probe():
    sets = {"a": set(range(50)), "b": set(range(25, 100))}
    fed, st, tr = federation(plan2(), sets, BindPolicy.ALWAYS)
    res = fed.answer("q")
    assert res.executed_plan.by_id()["b"].bind is not None and res.estimates == {}
    assert set(res.answer.value) == set(range(25, 50))


def test_the_necessary_bind_of_the_planner_is_kept_untouched():
    p = Plan((r("a", Store.SQL, "user"),
              Step("b", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("a",), bind=Bind("a", "user"), output="tag")))
    sets = {"a": {1, 2, 3}, "b": {1, 2, 3, 9}}
    fed, st, tr = federation(p, sets, BindPolicy.COST)
    res = fed.answer("q")
    assert res.executed_plan == p and not any(d.action == "bind" for d in res.bind_decisions)


# ------------------------------------------------------------------ Proposition 2 against brute force

OPS = (Op.INTERSECT, Op.SUBTRACT)


@pytest.mark.parametrize("seed", range(40))
def test_restricting_never_changes_the_answer(seed):
    rng = random.Random(seed)
    op = rng.choice(OPS)
    n_a, n_b = rng.choice((0, 1, 20, 400)), rng.choice((0, 1, 30, 900, 2_500))
    universe = range(3_000)
    a = set(rng.sample(universe, n_a))
    b = set(rng.sample(universe, n_b)) | set(rng.sample(sorted(a), min(len(a), rng.choice((0, 5, 50)))))
    truth = (a & b) if op is Op.INTERSECT else (a - b)
    plan = plan2(op)
    for policy in (BindPolicy.NONE, BindPolicy.ALWAYS, BindPolicy.COST):
        fed, *_ = federation(plan, {"a": a, "b": b}, policy, cap=100_000, batch=100)
        res = fed.answer("q")
        assert res.answer.exact, (seed, policy)
        assert set(res.answer.value) == truth, (seed, op, policy)


def test_restricting_the_minuend_would_be_wrong():
    """The reason a difference only restricts its subtrahend: a minus b is not b-restricted a."""
    a, b = set(range(10)), set(range(5, 15))
    assert (a - b) != (a & b)
    p = plan2(Op.SUBTRACT)
    fed, *_ = federation(p, {"a": a, "b": b}, BindPolicy.ALWAYS)
    res = fed.answer("q")
    assert res.executed_plan.by_id()["a"].bind is None and set(res.answer.value) == a - b


def test_decisions_are_deterministic():
    p = plan2()
    e = est(a=50, b=9_000)
    assert decide(p, e, BindPolicy.COST, COST) == decide(p, e, BindPolicy.COST, COST)


@pytest.mark.parametrize("seed", range(40))
def test_every_restriction_the_planner_may_choose_is_equivalent(seed):
    """Independent of the cost model: run every restriction that ``_pairs`` allows and compare with
    brute force. This is the test of Proposition 2 itself."""
    from heterorag.federation.bindplan import _pairs
    rng = random.Random(1000 + seed)
    op = rng.choice(OPS)
    a = set(rng.sample(range(500), rng.choice((0, 3, 60, 300))))
    b = set(rng.sample(range(500), rng.choice((0, 3, 60, 300))))
    truth = (a & b) if op is Op.INTERSECT else (a - b)
    plan = plan2(op)
    pairs = [(x.id, y.id) for _, x, y in _pairs(plan) if x is not None]
    assert pairs, "no restriction allowed"
    for x, y in pairs:
        restricted = restrict(plan, x, y)
        st = SetStore({"a": a, "b": b})
        ex = Executor({Store.GRAPH: st, Store.DOC: st}, cap=100_000, batch_size=64)
        run = ex.run(restricted, {"a": "Qa", "b": "Qb"})
        assert run.answer.exact and set(run.answer.value) == truth, (seed, op, x, y)


def test_the_forbidden_restriction_is_really_different():
    """Restricting the minuend of a difference returns the intersection: a counterexample exists."""
    differs = 0
    for seed in range(40):
        rng = random.Random(1000 + seed)
        a = set(rng.sample(range(500), rng.choice((3, 60))))
        b = set(rng.sample(range(500), rng.choice((3, 60))))
        st = SetStore({"a": a, "b": b})
        wrong = restrict(plan2(Op.SUBTRACT), "a", "b")
        run = Executor({Store.GRAPH: st, Store.DOC: st}, batch_size=64).run(wrong, {"a": "Qa", "b": "Qb"})
        differs += set(run.answer.value) != (a - b)
    assert differs > 10


def test_a_store_planner_estimate_does_not_license_independent_execution():
    # regression (live pilot D01-002): the graph probe fell back to a constant planner estimate (384)
    # while the step really returned more than the cap; the sibling was counted exactly (4 rows)
    ds = decide(plan2(), est(a=(384, "planner"), b=4), BindPolicy.COST, COST)
    assert len(ds) == 1
    d = ds[0]
    assert (d.action, d.step, d.source, d.level) == ("bind", "a", "b", "feasibility")
    assert "planner's estimate" in d.reason


def test_a_known_size_still_decides_on_cost_when_both_inputs_are_counted():
    d = decide(plan2(), est(a=60, b=70), BindPolicy.COST, COST)[0]
    assert d.action == "independent" and d.level == "profitability"       # unchanged for exact counts


def test_when_no_pair_can_be_restricted_the_reasons_of_the_other_pairs_are_kept():
    p = Plan((r("a"), Step("b", Op.RETRIEVE, Store.SQL, None, "q", inputs=("a",), bind=Bind("a", "question"),
                           output="question"),
              r("c", Store.DOC), combine(Op.INTERSECT, "b", "c", sid="i")))
    d = decide(p, est(c=500), BindPolicy.COST, COST)[0]          # c fits the cap, so nothing forces a bind to b
    assert d.action == "independent" and " | also " in d.reason and "already dependent" in d.reason


def test_the_policy_is_recorded_even_when_the_plan_is_rejected():
    class Rejecting:
        def plan(self, question):
            return PlanResult(abstain="invalid_plan", errors=["plan depth 4 exceeds 3"])

    fed = Federation(Rejecting(), None, Executor({}, cap=10_000), bind_policy=BindPolicy.COST)
    res = fed.answer("q")
    assert res.bind_policy == "cost" and res.to_dict()["bind_policy"] == "cost"


# ------------------------------------------------------------------ a sibling that is itself dependent (CP1 D01)

def d01_plan():
    """s1 graph (questions), s2 graph bound to s1 (their askers), s3 sql (users above a threshold), s4 intersect."""
    return Plan((r("s1", Store.GRAPH),
                 Step("s2", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("s1",), bind=Bind("s1", "question"),
                      output="user"),
                 r("s3", Store.SQL, entity="user"),
                 combine(Op.INTERSECT, "s2", "s3", sid="s4", entity="user")))


def test_a_step_known_to_exceed_the_cap_is_bound_to_a_dependent_sibling_without_an_estimate():
    ds = decide(d01_plan(), est(s1=7, s3=(10_001, "lower")), BindPolicy.COST, COST)
    assert len(ds) == 1
    d = ds[0]
    assert (d.action, d.step, d.source, d.level) == ("bind", "s3", "s2", "feasibility")
    assert "depends on another step" in d.reason and "cannot give a weaker label" in d.reason


@pytest.mark.parametrize("estimates", [
    est(s1=7, s3=500),                              # x fits the cap: nothing forces a bind, and y has no estimate
    est(s1=7, s3=(50_000, "planner")),              # only the store planner's guess: not known to exceed
    est(s1=7),                                      # x has no estimate
    {},
])
def test_without_knowledge_that_the_step_exceeds_the_cap_a_dependent_sibling_changes_nothing(estimates):
    d = one(d01_plan(), estimates)
    assert d.action == "independent" and d.level == "missing"


def test_a_sibling_without_an_estimate_because_its_probe_failed_is_still_not_a_source():
    p = Plan((r("s1", Store.GRAPH, entity="user"), r("s3", Store.SQL, entity="user"),
              combine(Op.INTERSECT, "s1", "s3", sid="s4", entity="user")))
    d = one(p, est(s3=(10_001, "lower")))               # s1 is independent and has no estimate
    assert d.action == "independent" and d.level == "missing"


def test_the_other_policies_ignore_the_new_rule():
    e = est(s1=7, s3=(10_001, "lower"))
    assert one(d01_plan(), e, BindPolicy.NONE).action == "independent"
    d = one(d01_plan(), e, BindPolicy.ALWAYS)
    assert d.level == "policy"


def test_d01_shape_end_to_end_is_exact_with_the_bind_and_truncated_without_it():
    sets = {"s1": set(range(7)), "s2": set(range(3, 7)), "s3": set(range(2, 50_000))}
    fed, *_ = federation(d01_plan(), sets, BindPolicy.NONE, cap=10_000)
    none = fed.answer("q")
    assert none.answer.guarantee is not Guarantee.EXACT                 # s3 was cut at the cap
    fed, st, tr = federation(d01_plan(), sets, BindPolicy.COST, cap=10_000)
    res = fed.answer("q")
    assert res.answer.exact and set(res.answer.value) == {3, 4, 5, 6}
    assert res.executed_plan.by_id()["s3"].bind == Bind("s2", "user") and res.executed_plan.depth() == 3
    assert res.run.records["s3"].rows_read == 4                         # 4 rows instead of 10,000
    assert res.bind_decisions[0].level == "feasibility" and ("s3", True) in st.calls


def test_when_the_dependent_sibling_turns_out_too_large_the_bound_result_is_still_a_sound_subset():
    # s2 returns 120 askers but the cap is 100: its rows are cut; s3 restricted to them cannot claim more than SUBSET
    sets = {"s1": set(range(500)), "s2": set(range(0, 120)), "s3": set(range(60, 50_000))}
    fed, *_ = federation(d01_plan(), sets, BindPolicy.COST, cap=100)
    res = fed.answer("q")
    assert res.answer.guarantee is not Guarantee.EXACT
    assert set(res.answer.value) <= set(range(60, 120))                 # every returned id is truly in the intersection
