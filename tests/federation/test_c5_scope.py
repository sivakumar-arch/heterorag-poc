"""Batch 99: scope of C5 (containment) and T2 (bound step larger than its input), decided 2026-11-16.

The four cases required by the decision:
  1. a planner-written same-entity bind on the graph store is out of scope (S1: duplicates of the bound questions);
  2. a bind made by the bind planner (policy) on the graph store is in scope;
  3. a same-entity bind on SQL is in scope;
  4. T2 has exactly the same scope as C5.
Then the plumbing: who wrote a bind is set by the code, recorded per step, never read from a model's reply, and
derived for result files written before this batch; and the re-score script stops when a row changes.
"""
import json

import pytest

from benchmark_v2 import dev_pass, rescore_c5_scope
from heterorag.federation import (BIND_ORIGIN_PLANNER, BIND_ORIGIN_POLICY, Bind, Executor, Guarantee, Mode, Op,
                                  Plan, PlanError, Step, Store, containment_applies)
from heterorag.federation.bindplan import restrict
from heterorag.federation.guarantee import Kind
from heterorag.federation.semantic_checks import Signatures
from tests.federation.test_executor import Fake, retrieve, table

TAGGED = [1, 2, 3]          # s1: questions tagged with the tag
DUPS = [2, 7, 8]            # s2: questions marked as duplicates of them; 7 and 8 are not tagged themselves


def chain(store: Store, origin: str, out: str = "question", src_entity: str = "question") -> Plan:
    """s1 (graph) -> s2 bound to s1 on ``store``, bind written by ``origin``."""
    s1 = retrieve("s1", Store.GRAPH, src_entity)
    s2 = Step("s2", Op.RETRIEVE, store, Mode.FILTER if store is Store.DOC else None, "q", inputs=("s1",),
              bind=Bind("s1", src_entity, origin=origin), output=out, kind=Kind.SET)
    return Plan((s1, s2))


def run(plan: Plan, s2_rows=DUPS, contain=True, s1_rows=TAGGED):
    clients = {st: Fake(table({"s1": s1_rows, "s2": s2_rows})) for st in (Store.GRAPH, Store.SQL, Store.DOC)}
    return Executor(clients, containment=contain).run(plan, {"s1": "N1", "s2": "N2"})


def row_of(planner_plan: Plan, executed: Plan, result, outcome="correct", with_origin=True) -> dict:
    """A result row as ``run_slice`` writes it, reduced to the fields T2 and C5 read."""
    recs = [r.to_dict() for r in result.records.values()]
    if not with_origin:                                  # a file written before Batch 99
        for r in recs:
            r.pop("bind_origin", None)
    return {"question_id": "Q-1", "text": "q", "outcome": outcome, "plan": planner_plan.to_dict(),
            "executed_plan": executed.to_dict(), "queries": {}, "step_records": recs, "answer_digest": None}


# ---------------------------------------------------------------- the four required cases

def test_1_planner_written_same_entity_graph_bind_is_out_of_scope():
    plan = chain(Store.GRAPH, BIND_ORIGIN_PLANNER)
    assert containment_applies(plan.by_id()["s2"]) is False
    r = run(plan, contain=True)
    s2 = r.records["s2"]
    assert s2.guarantee is Guarantee.EXACT and s2.error is None and r.answer.exact
    assert sorted(r.answer.value) == DUPS
    assert s2.outside_bound == 2 and s2.bind_origin == BIND_ORIGIN_PLANNER      # still counted, not acted on


def test_2_policy_made_graph_bind_is_in_scope():
    plan = chain(Store.GRAPH, BIND_ORIGIN_POLICY)
    assert containment_applies(plan.by_id()["s2"]) is True
    r = run(plan, contain=True)
    s2 = r.records["s2"]
    assert s2.guarantee is Guarantee.NONE and "containment violated: 2 of 3" in s2.error
    assert r.answer.type == "abstain" and s2.bind_origin == BIND_ORIGIN_POLICY
    # the bind planner itself writes policy binds, so its restrictions stay in scope on the graph store
    p = Plan((retrieve("a", Store.SQL), retrieve("b", Store.GRAPH), Step("c", Op.INTERSECT, inputs=("a", "b"), output="question")))
    rb = restrict(p, "b", "a").by_id()["b"]
    assert rb.bind.origin == BIND_ORIGIN_POLICY and containment_applies(rb) is True


@pytest.mark.parametrize("origin", [BIND_ORIGIN_PLANNER, BIND_ORIGIN_POLICY])
def test_3_same_entity_sql_bind_is_in_scope(origin):
    plan = chain(Store.SQL, origin)
    assert containment_applies(plan.by_id()["s2"]) is True
    r = run(plan, contain=True)
    assert r.records["s2"].guarantee is Guarantee.NONE and r.answer.type == "abstain"


CASES = [  # (store of s2, origin, output entity, bind entity)
    (Store.GRAPH, BIND_ORIGIN_PLANNER, "question", "question"),
    (Store.GRAPH, BIND_ORIGIN_POLICY, "question", "question"),
    (Store.SQL, BIND_ORIGIN_PLANNER, "question", "question"),
    (Store.SQL, BIND_ORIGIN_POLICY, "question", "question"),
    (Store.DOC, BIND_ORIGIN_PLANNER, "question", "question"),
    (Store.DOC, BIND_ORIGIN_POLICY, "question", "question"),
    (Store.GRAPH, BIND_ORIGIN_PLANNER, "user", "question"),      # different entities: never in scope
    (Store.SQL, BIND_ORIGIN_PLANNER, "user", "question"),
]


@pytest.mark.parametrize("store,origin,out,ent", CASES)
def test_4_t2_has_exactly_the_scope_of_c5(store, origin, out, ent):
    plan = chain(store, origin, out=out, src_entity=ent)
    step = plan.by_id()["s2"]
    # same predicate on the step ...
    assert dev_pass.in_scope(step, "narrow") is containment_applies(step)
    # ... and the same verdicts on a row: s2 returns more ids than its input and some outside it, so T2 and C5 fire
    # exactly when the step is in scope, and C5 in dev_pass fires exactly when the executor failed the step
    # (T2 and C5 are scored on OFF files, so the row comes from a run with the checks off; the executor's own
    # verdict comes from a run of the same plan with the checks on)
    rows_s2 = [2, 7, 8, 9]
    off = run(plan, s2_rows=rows_s2, contain=False)
    res, _ = dev_pass.fired(row_of(plan, plan, off), Signatures.poc(), "narrow")
    scoped = containment_applies(step)
    assert res["T2"] is scoped and res["C5"] is scoped
    on = run(plan, s2_rows=rows_s2, contain=True)
    assert (on.records["s2"].guarantee is Guarantee.NONE) is scoped


# ---------------------------------------------------------------- origin plumbing

def test_a_model_cannot_declare_its_own_bind_a_policy_restriction():
    raw = {"steps": [{"id": "s1", "op": "retrieve", "store": "graph", "question": "q", "output": "question"},
                     {"id": "s2", "op": "retrieve", "store": "graph", "question": "q", "inputs": ["s1"],
                      "bind": {"input": "s1", "entity": "question", "origin": "policy"}, "output": "question"}]}
    assert Plan.from_dict(raw).by_id()["s2"].bind.origin == BIND_ORIGIN_PLANNER
    assert Plan.from_dict(raw, trust_origin=True).by_id()["s2"].bind.origin == BIND_ORIGIN_POLICY


def test_unknown_origin_in_a_trusted_plan_is_an_error():
    raw = {"steps": [{"id": "s1", "op": "retrieve", "store": "graph", "question": "q", "output": "question"},
                     {"id": "s2", "op": "retrieve", "store": "graph", "question": "q", "inputs": ["s1"],
                      "bind": {"input": "s1", "entity": "question", "origin": "model"}, "output": "question"}]}
    with pytest.raises(PlanError):
        Plan.from_dict(raw, trust_origin=True)


def test_planner_binds_serialise_as_before_and_policy_binds_round_trip():
    planner = chain(Store.GRAPH, BIND_ORIGIN_PLANNER).to_dict()["steps"][1]["bind"]
    assert planner == {"input": "s1", "entity": "question", "param": ""}
    pol = chain(Store.GRAPH, BIND_ORIGIN_POLICY)
    assert pol.to_dict()["steps"][1]["bind"]["origin"] == BIND_ORIGIN_POLICY
    assert Plan.from_dict(pol.to_dict(), trust_origin=True) == pol


def test_step_record_stores_who_wrote_the_bind():
    r = run(chain(Store.SQL, BIND_ORIGIN_POLICY), s2_rows=[2, 3])
    d = {x["step"]: x for x in (rec.to_dict() for rec in r.records.values())}
    assert d["s2"]["bind_origin"] == BIND_ORIGIN_POLICY and d["s1"]["bind_origin"] is None


def test_origin_is_derived_for_files_written_before_batch_99():
    planner_plan = Plan((retrieve("a", Store.SQL), retrieve("b", Store.GRAPH),
                         Step("c", Op.INTERSECT, inputs=("a", "b"), output="question")))
    executed = restrict(planner_plan, "b", "a")                       # the bind planner bound b to a
    r = run(chain(Store.GRAPH, BIND_ORIGIN_PLANNER), s2_rows=[2])     # any run, only the plan fields matter here
    row = row_of(planner_plan, executed, r, with_origin=False)
    row["executed_plan"]["steps"][1]["bind"].pop("origin", None)      # old files carry no origin anywhere
    assert dev_pass.bind_origins(row) == {"b": BIND_ORIGIN_POLICY}
    p2 = chain(Store.GRAPH, BIND_ORIGIN_PLANNER)                      # the planner wrote the bind itself
    assert dev_pass.bind_origins(row_of(p2, p2, r, with_origin=False)) == {"s2": BIND_ORIGIN_PLANNER}
    no_exec = row_of(p2, p2, r, with_origin=False)
    no_exec["executed_plan"] = None                                   # files with only the planner's plan
    assert dev_pass.bind_origins(no_exec) == {"s2": BIND_ORIGIN_PLANNER}


def test_wide_scope_reproduces_the_rule_before_batch_99():
    plan = chain(Store.GRAPH, BIND_ORIGIN_PLANNER)
    r = run(plan, s2_rows=[2, 7, 8, 9], contain=False)
    row = row_of(plan, plan, r)
    wide, _ = dev_pass.fired(row, Signatures.poc(), "wide")
    narrow, _ = dev_pass.fired(row, Signatures.poc(), "narrow")
    assert wide["T2"] and wide["C5"] and not narrow["T2"] and not narrow["C5"]


# ---------------------------------------------------------------- re-score script

def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(json.dumps({"rows": rows}))
    return str(p)


def test_rescore_passes_when_no_row_changes(tmp_path, capsys):
    plan = chain(Store.SQL, BIND_ORIGIN_PLANNER)                       # in scope under both rules
    r = run(plan, s2_rows=[2, 7, 8, 9], contain=False)
    f = _write(tmp_path, "a.json", [row_of(plan, plan, r, outcome="wrong")])
    assert rescore_c5_scope.main([f]) == 0
    assert "no row changes" in capsys.readouterr().out


def test_rescore_stops_when_a_row_changes(tmp_path, capsys):
    plan = chain(Store.GRAPH, BIND_ORIGIN_PLANNER)                     # in scope before, out of scope now
    r = run(plan, s2_rows=[2, 7, 8, 9], contain=False)
    f = _write(tmp_path, "b.json", [row_of(plan, plan, r)])
    assert rescore_c5_scope.main([f]) == 1
    out = capsys.readouterr().out
    assert "Q-1 (correct): T2, C5" in out and "Report and stop" in out and "graph/planner/OUT 1" in out


def test_rescore_reports_an_unreadable_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    assert rescore_c5_scope.main([str(p)]) == 2
