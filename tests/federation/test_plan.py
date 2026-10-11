"""Plan model: validation, layering, (de)serialisation."""
import pytest

from heterorag.federation.guarantee import Kind
from heterorag.federation.plan import (MAX_DEPTH, MAX_STEPS, Bind, Mode, Op, Plan, PlanError,
                                       Step, Store, STEP_FIELDS)


def p01() -> Plan:
    return Plan((
        Step("s1", Op.RETRIEVE, Store.GRAPH, question="questions tagged T", output="question"),
        Step("s2", Op.RETRIEVE, Store.DOC, Mode.FILTER, "questions with the word W in the title",
             output="question"),
        Step("s3", Op.INTERSECT, inputs=("s1", "s2"), output="question"),
    ))


def dependent() -> Plan:
    return Plan((
        Step("s1", Op.RETRIEVE, Store.SQL, question="users with reputation at least R", output="user"),
        Step("s2", Op.RETRIEVE, Store.GRAPH, question="questions asked by these users",
             inputs=("s1",), bind=Bind("s1", "user"), output="question"),
    ))


def test_valid_plans():
    assert p01().validate() == []
    assert dependent().validate() == []


def test_layers_and_sink():
    layers = [[s.id for s in layer] for layer in p01().layers()]
    assert layers == [["s1", "s2"], ["s3"]]
    assert p01().sink().id == "s3"
    assert [[s.id for s in l] for l in dependent().layers()] == [["s1"], ["s2"]]
    assert dependent().depth() == 2


def replace(plan: Plan, sid: str, **kw) -> Plan:
    import dataclasses
    return Plan(tuple(dataclasses.replace(s, **kw) if s.id == sid else s for s in plan.steps))


@pytest.mark.parametrize("mutate,fragment", [
    (lambda p: replace(p, "s1", store=None), "needs a store"),
    (lambda p: replace(p, "s1", question=""), "sub-question"),
    (lambda p: replace(p, "s2", mode=None), "mode is required"),
    (lambda p: replace(p, "s1", mode=Mode.FILTER), "mode is required"),
    (lambda p: replace(p, "s2", k=5), "filter mode takes no k"),
    (lambda p: replace(p, "s3", inputs=("s1",)), "at least two inputs"),
    (lambda p: replace(p, "s3", inputs=("s1", "zz")), "unknown input"),
    (lambda p: replace(p, "s3", output="user"), "one entity type"),
    (lambda p: replace(p, "s3", store=Store.SQL), "takes no store"),
    (lambda p: replace(p, "s1", output=""), "output entity type missing"),
])
def test_invalid_independent_plan(mutate, fragment):
    errs = mutate(p01()).validate()
    assert any(fragment in e for e in errs), errs


@pytest.mark.parametrize("mutate,fragment", [
    (lambda p: replace(p, "s2", bind=None), "only through a bind"),
    (lambda p: replace(p, "s2", bind=Bind("s1", "question")), "differs from output entity"),
    (lambda p: replace(p, "s2", bind=Bind("sX", "user"), inputs=("s1",)), "single input"),
    (lambda p: replace(p, "s2", kind=Kind.RANKED, k=5), "ranked step is not allowed"),
    (lambda p: replace(p, "s1", kind=Kind.SCALAR), "can only bind a set"),
    (lambda p: replace(p, "s2", must_exist=True, kind=Kind.SCALAR), "set steps only"),
])
def test_invalid_dependent_plan(mutate, fragment):
    errs = mutate(dependent()).validate()
    assert any(fragment in e for e in errs), errs


def test_cycle_and_duplicate_ids_and_multiple_sinks():
    a = Step("a", Op.RETRIEVE, Store.SQL, question="q", output="user")
    cyc = Plan((
        Step("a", Op.INTERSECT, inputs=("b", "b"), output="user"),
        Step("b", Op.INTERSECT, inputs=("a", "a"), output="user"),
    ))
    assert any("cycle" in e or "final step" in e for e in cyc.validate())
    assert any("not unique" in e for e in Plan((a, a)).validate())
    two = Plan((a, Step("b", Op.RETRIEVE, Store.SQL, question="q", output="user")))
    assert any("exactly one final step" in e for e in two.validate())
    assert Plan(()).validate() == ["plan has no steps"]


def test_size_and_depth_limits():
    steps = [Step(f"s{i}", Op.RETRIEVE, Store.SQL, question="q", output="user") for i in range(MAX_STEPS)]
    steps.append(Step("u", Op.UNION, inputs=tuple(s.id for s in steps), output="user"))
    assert any("limit is" in e for e in Plan(tuple(steps)).validate())

    chain = [Step("s0", Op.RETRIEVE, Store.SQL, question="q", output="user")]
    for i in range(1, MAX_DEPTH + 1):
        chain.append(Step(f"s{i}", Op.RETRIEVE, Store.GRAPH, question="q", inputs=(f"s{i-1}",),
                          bind=Bind(f"s{i-1}", "user"), output="user"))
    assert any("exceeds" in e for e in Plan(tuple(chain)).validate())


def test_count_and_subtract_rules():
    base = p01().steps[:2]
    ok_count = Plan((base[0], Step("c", Op.COUNT, inputs=("s1",), output="number", kind=Kind.SCALAR)))
    assert ok_count.validate() == []
    bad_count = Plan((base[0], Step("c", Op.COUNT, inputs=("s1",), output="question")))
    assert any("produces a scalar" in e for e in bad_count.validate())
    sub3 = Plan((*base, Step("s3", Op.RETRIEVE, Store.SQL, question="q", output="question"),
                 Step("d", Op.SUBTRACT, inputs=("s1", "s2", "s3"), output="question")))
    assert any("exactly two inputs" in e for e in sub3.validate())


def test_round_trip_and_parse_errors():
    for plan in (p01(), dependent()):
        assert Plan.from_dict(plan.to_dict()) == plan
    with pytest.raises(PlanError) as ei:
        Plan.from_dict({"steps": [{"id": "a", "op": "join", "bogus": 1}, "x"]})
    msgs = ei.value.args[0]
    assert any("unknown fields" in m for m in msgs) and any("not an object" in m for m in msgs)
    with pytest.raises(PlanError):
        Plan.from_dict({"steps": []})


def test_step_fields_are_the_shared_schema():
    assert STEP_FIELDS == ("id", "op", "store", "mode", "question", "inputs", "bind", "output",
                           "kind", "k", "must_exist")
