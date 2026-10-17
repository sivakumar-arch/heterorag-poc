"""probe_check classifies probes against real row counts (scripted clients, no live store)."""
import json

from benchmark_v2 import probe_check
from heterorag.federation import Estimate, FetchResult, Store
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)
from tests.federation.test_reference_and_validation import records_for


class Client:
    def __init__(self, n, probe):
        self.n, self.probe = n, probe

    def estimate(self, step, query, limit):
        return self.probe(self.n, limit)

    def fetch(self, step, query, binds, limit):
        return FetchResult(list(range(min(self.n, limit))), self.n <= limit)


def run(world, n, probe, template="S05", cap=100):
    rec = records_for(world, [template], per=1)[0]
    c = Client(n, probe)
    return probe_check.check_record(rec, {Store.GRAPH: c, Store.SQL: c, Store.DOC: c}, cap)


def test_exact_count_and_lower_bound_are_ok(world):
    assert run(world, 7, lambda n, lim: Estimate(n, "count"))[0]["status"] == "ok"
    assert run(world, 500, lambda n, lim: Estimate(lim + 1, "lower"))[0]["status"] == "ok"


def test_wrong_probes_are_flagged(world):
    assert run(world, 7, lambda n, lim: Estimate(8, "count"))[0]["status"] == "WRONG"
    assert run(world, 7, lambda n, lim: Estimate(lim + 1, "lower"))[0]["status"] == "WRONG"    # actually fits
    assert run(world, 500, lambda n, lim: Estimate(40, "count"))[0]["status"] == "WRONG"        # truncated step


def test_fallback_and_missing_estimates_are_reported_separately(world):
    assert run(world, 7, lambda n, lim: Estimate(900, "planner"))[0]["status"] == "planner estimate"
    assert run(world, 7, lambda n, lim: None)[0]["status"] == "no estimate"


def test_only_unbound_set_steps_are_checked(world):
    rows = run(world, 3, lambda n, lim: Estimate(n, "count"), template="D04")
    assert [r["step"] for r in rows] == ["a"]                 # step b is dependent
    assert run(world, 3, lambda n, lim: Estimate(n, "count"), template="S01") == []   # a scalar


def test_cli_prints_table_and_exit_code(world, tmp_path, capsys):
    recs = records_for(world, ["S05", "S09"], per=2)
    q = tmp_path / "q.jsonl"
    q.write_text("\n".join(json.dumps(r) for r in recs))
    c = Client(5, lambda n, lim: Estimate(n, "count"))
    clients = {Store.GRAPH: c, Store.SQL: c, Store.DOC: c}
    assert probe_check.main(["--questions", str(q), "--out", str(tmp_path / "o.json")], clients=clients) == 0
    assert "graph" in capsys.readouterr().out and (tmp_path / "o.json").exists()
    bad = Client(5, lambda n, lim: Estimate(6, "count"))
    assert probe_check.main(["--questions", str(q)], clients={Store.GRAPH: bad, Store.SQL: bad, Store.DOC: bad}) == 1


def test_run_mode_checks_the_translated_queries_of_the_unbound_steps(world, tmp_path, capsys):
    rec = records_for(world, ["D05"], per=1)[0]
    row = {"question_id": "X-1", "template": "D05", "plan": rec["plan"], "queries": {"a": "MATCH 1", "b": "{}"},
           "executed_plan": [dict(s) for s in rec["plan"]]}
    row["executed_plan"] = {"steps": row["executed_plan"]}
    row["plan"] = {"steps": rec["plan"]}
    seen = []

    class Seeing(Client):
        def estimate(self, step, query, limit):
            seen.append((step.id, query))
            return Estimate(self.n, "count", 0.0, "subquery count: boom")

    c = Seeing(4, None)
    rows = probe_check.check_run_row(row, {Store.GRAPH: c, Store.SQL: c, Store.DOC: c}, 100)
    assert seen == [("a", "MATCH 1"), ("b", "{}")] and {r["status"] for r in rows} == {"ok"}
    assert rows[0]["why"] == "subquery count: boom"
    assert probe_check.check_run_row({"question_id": "X", "template": "D05", "plan": None, "queries": {}},
                                     {}, 100) == []

    f = tmp_path / "run.json"
    f.write_text(json.dumps({"rows": [row]}))
    assert probe_check.main(["--run", str(f)], clients={Store.GRAPH: c, Store.SQL: c, Store.DOC: c}) == 0
    assert "doc" in capsys.readouterr().out


def test_exactly_one_source_of_queries_is_required(tmp_path):
    assert probe_check.main([]) == 2
    assert probe_check.main(["--questions", "a", "--run", "b"]) == 2
