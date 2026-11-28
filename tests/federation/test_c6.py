"""Batch 107: C6, the empty-bind probe; its positive control; the translator wiring; the run metadata; the replay rule."""
import json

import pytest

from heterorag.federation import Bind, FetchResult, Kind, Mode, Op, Step, Store
from heterorag.federation import c6
from heterorag.federation.translator import Translator
from tests.federation.fake_descriptors import DESCRIPTORS, catalogue
from tests.federation.test_clients import es, neo, pg
from tests.federation.test_planner_translator import metered


def bound(store=Store.SQL, output="user", kind=Kind.SET, mode=None, bind_entity="user"):
    if store is Store.DOC and mode is None:
        mode = Mode.FILTER
    return Step("s2", Op.RETRIEVE, store, mode, "q", inputs=("s1",), bind=Bind("s1", bind_entity), output=output,
                kind=kind)


class BindAware:
    """A store that returns the ids of its table that pass the query: a neutralised bind (``OR TRUE``, ``OR true``,
    ``match_all``) returns every id, a restricted query only the bound ones. Records the calls."""
    def __init__(self, ids=(1, 2, 3), error=None, raises=None):
        self.ids, self.error, self.raises, self.calls = list(ids), error, raises, []

    def fetch(self, step, query, binds, limit):
        self.calls.append((step.id, query, dict(binds), limit))
        if self.raises:
            raise self.raises
        vacuous = any(t in query for t in ("OR TRUE", "OR true", "match_all"))
        bound_ids = set(next(iter(binds.values()))) if binds else set()
        got = [i for i in self.ids if vacuous or i in bound_ids]
        return FetchResult(got[:limit], len(got) <= limit, error=self.error)


# ---------------------------------------------------------------- scope and the probe

def test_scope_is_bound_set_steps_that_are_not_ranked():
    assert c6.in_scope(bound())
    assert not c6.in_scope(Step("s1", Op.RETRIEVE, Store.SQL, None, "q", output="user"))           # no bind
    assert not c6.in_scope(bound(kind=Kind.SCALAR, output="number"))
    assert not c6.in_scope(bound(store=Store.DOC, mode=Mode.SEARCH, kind=Kind.RANKED))
    client = BindAware()
    assert c6.probe(client, Step("s1", Op.RETRIEVE, Store.SQL, None, "q", output="user"), "x").status == c6.OUT_OF_SCOPE
    assert client.calls == []


def test_the_probe_empties_the_bind_and_caps_the_rows():
    client = BindAware()
    r = c6.probe(client, bound(), "SELECT id FROM users WHERE id = ANY(%(s1)s)")
    assert r.status == c6.PASSED
    assert client.calls == [("s2", "SELECT id FROM users WHERE id = ANY(%(s1)s)", {"s1": []}, c6.C6_ROWS)]


def test_a_neutralised_bind_fires_and_a_restricted_one_passes():
    client = BindAware()
    assert c6.probe(client, bound(), "SELECT id FROM users WHERE id = ANY(%(s1)s) OR TRUE").status == c6.FIRED
    assert c6.probe(client, bound(), "SELECT id FROM users WHERE id = ANY(%(s1)s)").status == c6.PASSED
    r = c6.probe(client, bound(), "SELECT id FROM users WHERE id = ANY(%(s1)s) OR TRUE")
    assert r.reason == c6.C6_MESSAGE and r.identifiers == 1


def test_a_store_that_cannot_be_asked_is_a_skip_never_a_firing():
    r = c6.probe(BindAware(raises=ConnectionError("down")), bound(), "q")
    assert r.status == c6.SKIPPED and "ConnectionError" in r.reason
    r = c6.probe(BindAware(ids=(), error="timeout"), bound(), "q")
    assert r.status == c6.SKIPPED and "timeout" in r.reason
    # identifiers that came back before an error still show a query that ignores its bind
    assert c6.probe(BindAware(error="timeout"), bound(), "x OR TRUE").status == c6.FIRED


# ---------------------------------------------------------------- empty means no identifier (decision 1)

def test_a_null_row_on_sql_does_not_fire():
    client, conn = pg([None])                     # an outer join returns one row with a null identifier
    q = "SELECT u.id FROM (SELECT 1) d LEFT JOIN users u ON u.id = ANY(%(s1)s)"
    assert c6.probe(client, bound(), q).status == c6.PASSED
    client, _ = pg([7])
    assert c6.probe(client, bound(), q).status == c6.FIRED


def test_a_null_row_on_the_graph_does_not_fire():
    client, _ = neo([None])                       # OPTIONAL MATCH returns one row with a null identifier
    q = "OPTIONAL MATCH (u:User) WHERE u.id IN $s1 RETURN u.id"
    assert c6.probe(client, bound(Store.GRAPH), q).status == c6.PASSED
    client, _ = neo([7])
    assert c6.probe(client, bound(Store.GRAPH), q).status == c6.FIRED


def test_a_null_identifier_on_the_document_store_does_not_fire():
    client, _ = es([None])                        # a hit whose identifier field is null
    q = json.dumps({"query": {"terms": {"post_id": "$bind:s1"}}})
    st = bound(Store.DOC, output="question", bind_entity="question")
    assert c6.probe(client, st, q).status == c6.PASSED
    client, _ = es([7])
    assert c6.probe(client, st, q).status == c6.FIRED


# ---------------------------------------------------------------- positive control (decision 2)

def test_the_control_uses_each_stores_placeholder_and_the_probe():
    clients = {s: BindAware() for s in (Store.SQL, Store.GRAPH, Store.DOC)}
    out = c6.control(clients)
    assert out["ok"] and all(v["ok"] for v in out["stores"].values())
    sql_q = [q for _, q, _, _ in clients[Store.SQL].calls]
    assert all("= ANY(%(c6)s)" in q for q in sql_q)
    assert all("IN $c6" in q for _, q, _, _ in clients[Store.GRAPH].calls)
    assert all('"$bind:c6"' in q for _, q, _, _ in clients[Store.DOC].calls)
    assert all(b == {"c6": []} and lim == c6.C6_ROWS for s in clients.values() for _, _, b, lim in s.calls)


def test_the_control_fails_when_a_store_ignores_the_vacuous_query_or_cannot_be_asked():
    class Deaf(BindAware):
        def fetch(self, step, query, binds, limit):
            return FetchResult([], True)
    class Loose(BindAware):                       # returns rows whatever the bind: the restricted query fires too
        def fetch(self, step, query, binds, limit):
            return FetchResult([1][:limit], True)
    out = c6.control({Store.SQL: BindAware(), Store.GRAPH: Loose(), Store.DOC: BindAware()})
    assert not out["ok"] and "restricted query: fired" in out["stores"]["graph"]["why"]
    out = c6.control({Store.SQL: Deaf(), Store.GRAPH: BindAware(), Store.DOC: BindAware()})
    assert not out["ok"] and not out["stores"]["sql"]["ok"] and "vacuous query: passed" in out["stores"]["sql"]["why"]
    out = c6.control({Store.SQL: BindAware(raises=ConnectionError("x")), Store.GRAPH: BindAware(),
                      Store.DOC: BindAware()})
    assert not out["ok"] and "skipped" in out["stores"]["sql"]["why"]
    out = c6.control({Store.SQL: BindAware(), Store.GRAPH: BindAware()})
    assert not out["ok"] and out["stores"]["doc"]["why"] == "no client for the store"


# ---------------------------------------------------------------- translator: every attempt, one repair

OK_Q = "SELECT id FROM users WHERE reputation >= 5 AND id = ANY(%(s1)s)"
LOOSE_Q = "SELECT id FROM users WHERE reputation >= 5 AND (id = ANY(%(s1)s) OR id > 0)"


class ByQuery(BindAware):
    def fetch(self, step, query, binds, limit):
        self.calls.append((step.id, query, dict(binds), limit))
        return FetchResult([9][:limit] if "id > 0" in query else [], True)


def test_the_translator_probes_every_attempt_and_repairs_once_on_a_firing():
    m, inner = metered([LOOSE_Q, OK_Q])
    client = ByQuery()
    t = Translator(m, DESCRIPTORS, catalogue(), semantic_checks=True, c6_clients={Store.SQL: client})
    r = t.translate(bound())
    assert r.ok and r.attempts == 2 and r.query == OK_Q
    assert (r.c6_probes, r.c6_fired, r.c6_skipped, r.store) == (2, 1, 0, "sql")
    assert c6.C6_MESSAGE in inner.prompts[1]
    assert [q for _, q, _, _ in client.calls] == [LOOSE_Q, OK_Q]


def test_the_translator_does_not_probe_with_the_checks_off_or_out_of_scope():
    m, _ = metered([LOOSE_Q])
    client = ByQuery()
    r = Translator(m, DESCRIPTORS, catalogue(), semantic_checks=False, c6_clients={Store.SQL: client}).translate(bound())
    assert r.ok and r.c6_probes == 0 and client.calls == []
    m, _ = metered(["SELECT id FROM users WHERE reputation >= 5"])
    r = Translator(m, DESCRIPTORS, catalogue(), semantic_checks=True,
                   c6_clients={Store.SQL: client}).translate(Step("s1", Op.RETRIEVE, Store.SQL, None, "q", output="user"))
    assert r.ok and r.c6_probes == 0 and client.calls == []


def test_a_skipped_probe_passes_with_a_note_and_is_counted():
    m, _ = metered([OK_Q])
    t = Translator(m, DESCRIPTORS, catalogue(), semantic_checks=True,
                   c6_clients={Store.SQL: BindAware(raises=ConnectionError("down"))})
    r = t.translate(bound())
    assert r.ok and (r.c6_probes, r.c6_fired, r.c6_skipped) == (1, 0, 1)
    assert any(n.startswith("C6 probe skipped") for n in r.notes) and "ConnectionError" in r.c6_skip_reasons[0]


# ---------------------------------------------------------------- run metadata (decisions 2 and 3)

def test_run_slice_stops_before_any_question_when_the_control_fails(tmp_path, monkeypatch, capsys):
    from benchmark_v2 import run_slice

    class Fed:
        asked = 0

        def store_clients(self):
            class Deaf(BindAware):
                def fetch(self, step, query, binds, limit):
                    return FetchResult([], True)
            return {Store.SQL: Deaf(), Store.GRAPH: BindAware(), Store.DOC: BindAware()}

        def answer(self, *a, **k):
            Fed.asked += 1
            raise AssertionError("no question may run")
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"question_id": "D01-001", "template": "D01", "split": "dev", "text": "t", "plan": []}) + "\n")
    out = tmp_path / "out.json"
    monkeypatch.setattr(run_slice, "semantic_checks_enabled", lambda: True)
    code = run_slice.main(["--questions", str(q), "--out", str(out)], federation=Fed(), models=None)
    assert code == run_slice.EXIT_C6_CONTROL and Fed.asked == 0
    meta = json.loads(out.read_text())["meta"]
    assert meta["semantic_checks"] is True and meta["store_check"] is True
    assert meta["c6_control"]["ok"] is False and meta["counts_for_F"] is False
    # an existing output (a run being resumed) is never overwritten: the record goes beside it
    out.write_text("keep me")
    assert run_slice.main(["--questions", str(q), "--out", str(out)], federation=Fed(), models=None) == 3
    assert out.read_text() == "keep me"
    assert json.loads((tmp_path / "out.json.c6_control_failed.json").read_text())["meta"]["c6_control"]["ok"] is False


def test_c6_counts_per_store_sum_every_translation():
    from benchmark_v2.run_slice import c6_counts, c6_totals
    from heterorag.federation.translator import TranslationResult
    a = TranslationResult("s2", store="sql", c6_probes=2, c6_fired=1)
    b = TranslationResult("s3", store="doc", c6_probes=1, c6_skipped=1, c6_skip_reasons=["down"])
    row = {"c6": c6_counts([a, b, TranslationResult("s1", store="graph")])}
    assert row["c6"] == {"sql": {"probes": 2, "fired": 1, "skipped": 0, "skip_reasons": []},
                         "doc": {"probes": 1, "fired": 0, "skipped": 1, "skip_reasons": ["down"]}}
    assert c6_totals([row, row]) == {"sql": {"probes": 4, "fired": 2, "skipped": 0},
                                     "doc": {"probes": 2, "fired": 0, "skipped": 2}}


# ---------------------------------------------------------------- the replay and the rule

def _row(qid, outcome, query, store="sql"):
    plan = {"steps": [{"id": "s1", "op": "retrieve", "store": "graph", "question": "q", "output": "user"},
                      {"id": "s2", "op": "retrieve", "store": store, "question": "q", "inputs": ["s1"],
                       "bind": {"input": "s1", "entity": "user"}, "output": "user",
                       **({"mode": "filter"} if store == "doc" else {})},
                      {"id": "s3", "op": "intersect", "inputs": ["s1", "s2"], "output": "user"}]}
    return {"question_id": qid, "outcome": outcome, "executed_plan": plan,
            "queries": {"s1": "MATCH (u:User) RETURN u.id", "s2": query}}


def test_the_rule_keeps_at_two_correct_firings_and_drops_at_three():
    from benchmark_v2.c6_replay import rule
    from collections import Counter
    base = {"rows": Counter(correct=232), "fired_rows": Counter(correct=2), "probes": Counter(sql=100),
            "skipped": Counter(sql=5)}
    assert rule([base])["verdict"] == "keep"
    assert rule([{**base, "fired_rows": Counter(correct=3)}])["verdict"] == "drop"
    assert rule([{**base, "skipped": Counter(sql=6)}])["verdict"] == "drop"
    assert rule([])["verdict"] == "drop"


def test_the_replay_probes_recorded_final_queries_and_applies_the_rule_to_the_ladder_only(tmp_path, capsys):
    from benchmark_v2 import c6_replay
    clients = {s: BindAware() for s in (Store.SQL, Store.GRAPH, Store.DOC)}
    ladder = {"meta": {"translator_model": "ollama:qwen3:30b-instruct"},
              "rows": [_row("D01-001", "correct", "SELECT id FROM users WHERE id = ANY(%(s1)s)"),
                       _row("D01-002", "wrong", "SELECT id FROM users WHERE id = ANY(%(s1)s) OR TRUE"),
                       _row("D01-003", "correct", "SELECT id FROM users WHERE id = ANY(%(s1)s) OR TRUE"),
                       {"question_id": "D01-004", "outcome": "abstain_unexpected", "plan": None}]}
    gemma = {"meta": {"translator_model": "ollama:gemma4:31b"},
             "rows": [_row(f"D01-0{i}", "correct", "x OR TRUE") for i in range(10, 15)]}
    f1, f2 = tmp_path / "t0_qwen.json", tmp_path / "t0_gemma4.json"
    f1.write_text(json.dumps(ladder)); f2.write_text(json.dumps(gemma))
    out = tmp_path / "rep.json"
    assert c6_replay.main([str(f1), str(f2), "--out", str(out)], clients=clients) == 0
    rep = json.loads(out.read_text())
    q = rep["files"][str(f1)]
    assert q["rows"] == {"correct": 2, "other": 1, "silent": 1} and q["unscored"] == {"other": 1}
    assert q["fired_rows"] == {"correct": 1, "silent": 1} and q["probes"] == {"sql": 3}
    assert rep["files"][str(f2)]["group"] == "gemma" and rep["files"][str(f2)]["fired_rows"] == {"correct": 5}
    assert rep["rule"]["correct_rows"] == 2 and rep["rule"]["correct_rows_fired"] == 1   # Gemma is not in the rule
    assert rep["rule"]["verdict"] == "keep"
    # only the recorded final query is probed, with the bind emptied
    assert [(qq, b) for _, qq, b, _ in clients[Store.SQL].calls if "c6" not in b][0] == \
        ("SELECT id FROM users WHERE id = ANY(%(s1)s)", {"s1": []})


def test_the_replay_stops_when_the_control_fails(tmp_path):
    from benchmark_v2 import c6_replay
    f = tmp_path / "t0.json"
    f.write_text(json.dumps({"meta": {}, "rows": []}))
    out = tmp_path / "rep.json"
    clients = {Store.SQL: BindAware(), Store.GRAPH: BindAware()}
    assert c6_replay.main([str(f), "--out", str(out)], clients=clients) == c6_replay.EXIT_CONTROL
    assert json.loads(out.read_text())["control"]["ok"] is False
