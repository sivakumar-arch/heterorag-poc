"""Batch 103: class F (first-contact shapes), per-step sizes and hashes, quotas and group caps, the pattern key by step.

Approved by Siva on 2026-11-26. The class F rule is the registered cap rule checked step by step, without the lower
threshold T; the pattern key compares by step id where the question record has step sizes, and keeps the store lookup
for every record without them (all dev records), so dev keys cannot change.
"""
import dataclasses

import pytest

from benchmark_v2 import catalogue_d, pattern_key as pk, questions
from benchmark_v2.questions import Bind, Gold, Step, Template
from heterorag.federation import executor
from heterorag.federation.guarantee import Guarantee, Status
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)

H = 100


# ---------------------------------------------------------------- hashes

def test_the_generator_and_the_executor_hash_id_sets_alike():
    for ids in ([3, 1, 2], {"7", "10"}, [], [42]):
        assert questions.id_hash(ids) == executor.id_hash(ids)
    assert questions.id_hash([1, 2, 10]) == questions.id_hash(["10", " 2", "1"])
    assert questions.id_hash([1, 2]) != questions.id_hash([1, 3])


def test_step_records_carry_a_hash_of_their_ids():
    r = executor.StepRecord("a", Status.COMPLETE, Guarantee.EXACT, value=frozenset({5, 3}))
    d = r.to_dict()
    assert d["value_size"] == 2 and d["value_hash"] == questions.id_hash([3, 5])
    assert executor.StepRecord("c", Status.COMPLETE, Guarantee.EXACT, value=7).to_dict()["value_hash"] is None


# ---------------------------------------------------------------- class F

def _t(plan, answer_type="ids", entity="user"):
    return Template("FX", "F", "dev", answer_type, entity, ("q", "q", "q"), tuple(plan),
                    lambda c, n: [], lambda c, p: None)


INTERSECT = [Step("a", "retrieve", "graph", output="user"), Step("b", "retrieve", "sql", output="user"),
             Step("c", "intersect", inputs=("a", "b"), output="user")]
UNION = [Step("a", "retrieve", "graph", output="question"), Step("b", "retrieve", "doc", "filter", output="question"),
         Step("c", "retrieve", "graph", output="question"), Step("d", "intersect", inputs=("a", "b"), output="question"),
         Step("e", "union", inputs=("d", "c"), output="question")]
SUBTRACT = [Step("a", "retrieve", "graph", output="user"), Step("b", "retrieve", "graph", output="user"),
            Step("c", "subtract", inputs=("a", "b"), output="user")]
BOUND = [Step("a", "retrieve", "sql", output="user"),
         Step("b", "retrieve", "graph", bind=Bind("a", "user", "a"), output="question"),
         Step("c", "retrieve", "sql", output="question"), Step("d", "intersect", inputs=("b", "c"), output="question"),
         Step("e", "count", inputs=("d",), output="number", kind="scalar")]


def chk(plan, steps, value=(1,), answer_type="ids"):
    return questions.check_gold(_t(plan, answer_type), {}, Gold(list(value) if answer_type == "ids" else value,
                                                                   steps=steps), t_ind=200, h_cap=H)


def test_class_f_has_no_lower_threshold_and_keeps_the_answer_window():
    assert chk(INTERSECT, {"a": 3, "b": 4}) is None                       # class D would call this class P
    assert chk(INTERSECT, {"a": 3, "b": 4}, value=()) == "set size 0, the answer is empty"
    assert "above 50" in chk(INTERSECT, {"a": 300, "b": 60}, value=range(51))


def test_class_f_step_rule():
    assert chk(INTERSECT, {"a": 50, "b": 5000}) is None                   # b restrictable to a
    assert "class F needs the size" in chk(INTERSECT, {"a": 5})
    assert chk(UNION, {"a": 80, "b": 900, "c": 20}) is None              # the word step is restrictable
    assert "union e" in chk(UNION, {"a": 80, "b": 50, "c": 150})         # a union input is never restricted
    assert chk(SUBTRACT, {"a": 40, "b": 800}) is None                     # the subtrahend is restrictable
    assert "not restrictable" in chk(SUBTRACT, {"a": 800, "b": 40})       # the minuend is not
    assert chk(BOUND, {"a": 30, "b": 90, "c": 5000}, value=7, answer_type="scalar") is None
    assert "bound input of step b" in chk(BOUND, {"a": 300, "b": 90, "c": 50}, value=7, answer_type="scalar")
    assert "bound step b returns more" in chk(BOUND, {"a": 30, "b": 900, "c": 50}, value=7, answer_type="scalar")
    assert "count must be at least 1" in chk(BOUND, {"a": 30, "b": 90, "c": 50}, value=0, answer_type="scalar")


def test_no_retrieve_step_within_h_is_rejected():
    assert "no retrieve step within" in chk(INTERSECT, {"a": 500, "b": 600})


def test_needs_restriction_lists_the_steps_above_h():
    t = _t(INTERSECT)
    assert questions.needs_restriction(t, Gold([1], steps={"a": 50, "b": 5000}), H) == ["b"]
    assert questions.needs_restriction(t, Gold([1], steps={"a": 50, "b": 60}), H) == []


# ---------------------------------------------------------------- quotas and group caps

def test_quota_split_by_largest_remainder():
    q = (("over-cap", 2), ("within-cap", 1))
    assert questions.quota_split(q, 30) == {"over-cap": 20, "within-cap": 10}
    assert questions.quota_split(q, 2) == {"over-cap": 1, "within-cap": 1}
    assert questions.quota_split(q, 3) == {"over-cap": 2, "within-cap": 1}


def _cands(rows):
    by = {}
    for p in rows:
        by.setdefault(p["stratum"], []).append((p, Gold([1])))
    return by


def test_group_caps_and_quota_without_top_up():
    t = dataclasses.replace(_t(INTERSECT), quota=(("over", 2), ("within", 1)),
                            group_caps=(("badge", 1), ("year", {"over": 2, "within": 3})))
    rows = ([{"stratum": "over", "badge": f"b{i}", "year": 2015 + i % 2} for i in range(10)]
            + [{"stratum": "within", "badge": "same", "year": 2010} for _ in range(5)])
    chosen = questions.select(t, _cands(rows), 6)
    over = [p for p, _ in chosen if p["stratum"] == "over"]
    within = [p for p, _ in chosen if p["stratum"] == "within"]
    assert len(over) == 4 and len({p["badge"] for p in over}) == 4       # quota 4, one per badge
    assert max(sum(p["year"] == y for p in over) for y in (2015, 2016)) == 2   # at most two per year
    assert len(within) == 1                                              # one badge only: short, not topped up


def test_round_robin_without_quota_respects_caps():
    t = dataclasses.replace(_t(INTERSECT), group_caps=(("tag", 1),))
    rows = [{"stratum": "rank 02", "tag": "x"}, {"stratum": "rank 02", "tag": "y"},
            {"stratum": "rank 05", "tag": "x"}, {"stratum": "rank 05", "tag": "z"}]
    chosen = questions.select(t, _cands(rows), 4)
    assert [p["tag"] for p, _ in chosen] == ["x", "z", "y"]


# ---------------------------------------------------------------- step sets of the existing templates

def test_repeat_records_gain_step_sizes_and_hashes_and_nothing_else_changes(world):
    _, db, _ = world
    new, _ = questions.build_records(catalogue_d.TEMPLATES, db, 5, 3, 10, 40)
    old, _ = questions.build_records([dataclasses.replace(t, step_sets=None) for t in catalogue_d.TEMPLATES],
                                     db, 5, 3, 10, 40)
    assert [{k: v for k, v in r.items() if k not in ("step_sizes", "step_hashes")} for r in new] == old
    for r in new:
        assert set(r["step_sizes"]) == {s["id"] for s in r["plan"] if s["op"] == "retrieve"}
        if r["template"] in ("D01", "D05"):
            assert r["step_sizes"] == {"a": r["intermediates"]["graph"],
                                       "b": r["intermediates"]["sql" if r["template"] == "D01" else "doc"]}
        if r["template"] in ("D04", "D07"):
            assert r["step_sizes"]["a"] == r["bound"]["a"]
        if r["template"] == "D04":
            assert r["step_hashes"]["b"] == questions.id_hash(r["gold"])


# ---------------------------------------------------------------- pattern key by step

S2_PLAN = [{"id": "a", "op": "retrieve", "store": "graph", "output": "user", "kind": "set"},
           {"id": "b", "op": "retrieve", "store": "graph", "output": "user", "kind": "set"},
           {"id": "c", "op": "subtract", "inputs": ["a", "b"], "output": "user", "kind": "set"}]


def _row(executed, records, plan_equal=True):
    return {"template": "S2", "plan_equal": plan_equal, "executed_plan": {"steps": executed},
            "step_records": records, "answer_digest": {"type": "ids", "size": 3}, "precision": 0.5, "recall": 1.0}


def _qrec(sizes, hashes=None):
    return {"plan": S2_PLAN, "intermediates": {"graph": 999}, "step_sizes": sizes,
            "step_hashes": hashes or {k: questions.id_hash(range(v)) for k, v in sizes.items()}}


EXEC = [{"id": "s1", "op": "retrieve", "store": "graph", "output": "user", "kind": "set"},
        {"id": "s2", "op": "retrieve", "store": "graph", "output": "user", "kind": "set"},
        {"id": "s3", "op": "subtract", "inputs": ["s1", "s2"], "output": "user", "kind": "set"}]


def test_alignment_follows_the_plan_structure_not_the_ids():
    assert pk.align({"steps": EXEC}, S2_PLAN) == {"s3": "c", "s1": "a", "s2": "b"}
    swapped = [EXEC[1], EXEC[0], EXEC[2]]
    assert pk.align({"steps": swapped}, S2_PLAN) == {"s3": "c", "s1": "a", "s2": "b"}
    # two inputs of one intersection with the same structure are not aligned
    inter = [dict(s, op="intersect") if s["id"] == "c" else s for s in S2_PLAN]
    ex_i = [dict(s, op="intersect") if s["id"] == "s3" else s for s in EXEC]
    assert pk.align({"steps": ex_i}, inter) == {"s3": "c"}


def test_the_second_step_on_the_same_store_is_told_apart():
    q = _qrec({"a": 40, "b": 30})
    recs = [{"step": "s1", "value_size": 40, "value_hash": questions.id_hash(range(40))},
            {"step": "s2", "value_size": 12, "value_hash": "x"}]
    assert pk.pattern_key(_row(EXEC, recs), q) == ("S2", "step2", "subset")
    # under the store lookup both steps would be compared with the one store size (999)
    legacy = {k: v for k, v in q.items() if k not in ("step_sizes", "step_hashes")}
    assert pk.pattern_key(_row(EXEC, recs), legacy) == ("S2", "step1", "subset")


def test_equal_size_with_different_ids_is_direction_other_and_equal_sets_fall_to_final():
    q = _qrec({"a": 40, "b": 30})
    same = [{"step": "s1", "value_size": 40, "value_hash": questions.id_hash(range(40))},
            {"step": "s2", "value_size": 30, "value_hash": questions.id_hash(range(30))}]
    assert pk.pattern_key(_row(EXEC, same), q) == ("S2", "final", "superset")
    other = [dict(same[0], value_hash=questions.id_hash(range(1, 41))), same[1]]
    assert pk.pattern_key(_row(EXEC, other), q) == ("S2", "step1", "other")
    no_hash = [dict(same[0], value_hash=None), same[1]]                 # an older run file: size only
    assert pk.pattern_key(_row(EXEC, no_hash), q) == ("S2", "final", "superset")


def test_a_step_bound_by_the_bind_planner_is_skipped_as_before():
    q = _qrec({"a": 40, "b": 30})
    bound = [EXEC[0], dict(EXEC[1], bind={"input": "s1", "entity": "user", "param": "s1"}), EXEC[2]]
    recs = [{"step": "s1", "value_size": 40, "value_hash": questions.id_hash(range(40))},
            {"step": "s2", "value_size": 5, "value_hash": "x"}]
    assert pk.pattern_key(_row(bound, recs), q) == ("S2", "final", "superset")


def test_records_without_step_sizes_keep_the_registered_store_lookup():
    q = {"intermediates": {"graph": 40, "sql": 9}}
    plan = [{"id": "s1", "op": "retrieve", "store": "graph", "output": "user"},
            {"id": "s2", "op": "retrieve", "store": "sql", "output": "user"},
            {"id": "s3", "op": "intersect", "inputs": ["s1", "s2"], "output": "user"}]
    row = _row(plan, [{"step": "s1", "value_size": 40}, {"step": "s2", "value_size": 0}])
    assert pk.pattern_key(row, q) == ("S2", "step2", "empty")


def test_pool_counts_report_step_sizes_and_the_restriction_share(world):
    from benchmark_v2 import pool_counts
    _, db, _ = world

    def sample(c, n):
        return [{"stratum": "s", "k": k} for k in range(4)]

    def gold(c, p):
        return Gold([1], steps={"a": 10 + p["k"], "b": 150 * (p["k"] + 1)})

    t = dataclasses.replace(_t(INTERSECT), sample=sample, gold=gold)
    r = pool_counts.count(t, questions.Ctx(db, 1, 10, H), set(), 10, H)
    assert r["pool"] == 4 and r["step_sizes"]["a"] == {"min": 10, "median": 11, "max": 13}
    assert r["needs_restriction"] == 4                     # b is 150 to 600, above H = 100, in every instance
