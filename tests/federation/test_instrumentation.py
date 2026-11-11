"""Batch 91: bytes moved, cap hits and the time split are recorded per step and per question."""
import json

from heterorag.federation import Executor, FetchResult, Kind, Mode, Op, Plan, Step, Store
from heterorag.federation.executor import logical_bytes
from tests.federation.test_executor import Fake, bound, dep_plan, owner_rows, p01_plan, retrieve, run, table


def test_logical_bytes_is_the_compact_json_length_of_the_values():
    assert logical_bytes([]) == 2 and logical_bytes([1, 2, 3]) == len("[1,2,3]") == 7
    assert logical_bytes(["ab", "c"]) == len('["ab","c"]')
    assert logical_bytes([object()]) > 0          # unserialisable values are measured as text, never raise


def test_independent_steps_report_the_bytes_they_returned_and_none_sent():
    r = run(p01_plan(), {Store.GRAPH: Fake(table({"s1": [1, 22, 333]})), Store.DOC: Fake(table({"s2": [2]}))})
    assert r.records["s1"].bytes == len("[1,22,333]") and r.records["s2"].bytes == len("[2]")
    assert r.records["s1"].bytes_out == 0 and not r.records["s1"].cap_hit
    t = r.totals()
    assert t["bytes"] == len("[1,22,333]") + len("[2]") and t["bytes_out"] == 0 and t["cap_hits"] == 0


def test_a_bound_step_reports_the_identifiers_it_was_sent_in_every_batch():
    sql, graph = Fake(owner_rows), Fake(owner_rows)
    r = run(dep_plan(), {Store.SQL: sql, Store.GRAPH: graph}, batch_size=3)
    sent = sum(len(json.dumps(c[2][next(iter(c[2]))], separators=(",", ":"))) for c in graph.calls)
    rec = r.records["s2"]
    assert rec.batches == len(graph.calls) > 1 and rec.bytes_out == sent > 0 and rec.bytes > 0


def test_reaching_the_cap_is_flagged_on_a_single_and_on_a_batched_step():
    single = run(p01_plan(), {Store.GRAPH: Fake(table({"s1": list(range(10))})), Store.DOC: Fake(table({"s2": [1]}))},
                 cap=4)
    assert single.records["s1"].cap_hit and not single.records["s2"].cap_hit
    assert single.records["s1"].to_dict()["cap_hit"] is True and single.totals()["cap_hits"] == 1
    batched = run(dep_plan(), {Store.SQL: Fake(owner_rows), Store.GRAPH: Fake(owner_rows)}, batch_size=1, cap=2)
    assert batched.records["s2"].cap_hit


def test_an_error_after_some_rows_is_not_a_cap_hit():
    def fault(n, step, binds):
        return ("partial", 2)
    r = run(p01_plan(), {Store.GRAPH: Fake(table({"s1": [1, 2, 3, 4]}), fault=fault), Store.DOC: Fake(table({"s2": [2]}))})
    assert r.records["s1"].error == "timeout" and not r.records["s1"].cap_hit


def test_a_scalar_is_never_a_cap_hit():
    plan = Plan((Step("s1", Op.RETRIEVE, Store.SQL, None, "q", output="question", kind=Kind.SCALAR),))
    r = run(plan, {Store.SQL: Fake(table({"s1": [42]}))}, cap=1)
    assert not r.records["s1"].cap_hit and r.records["s1"].bytes == len("[42]")
