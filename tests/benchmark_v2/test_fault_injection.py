"""Fault injection: operators, verdicts, report, and the command line (no live stores, no model)."""
import dataclasses
import json
import random

import pytest

from benchmark_v2 import questions
from benchmark_v2.fault_injection import (FAULT_BY_ID, FAULTS, SEMANTIC, STRUCTURAL, baseline_problems, final_verdict,
                                          inject_record, main, render)
from benchmark_v2.reference_queries import reference_queries
from heterorag.federation import Executor, FetchResult, Plan, Store
from heterorag.federation.contract import GraphCatalogue
from heterorag.federation.step_validator import StepValidator
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)
from tests.federation.fake_descriptors import catalogue
from tests.federation.test_reference_and_validation import records_for

ALL = [t.id for t in questions.all_templates()]


@pytest.fixture
def cat():
    return catalogue()


@pytest.fixture
def validator(cat):
    return StepValidator(cat, case_insensitive_text=False)


def mutants(world, cat, validator, ids=ALL):
    """(fault, record, step id, mutated query, rejected?) for every applicable mutant of the synthetic questions."""
    out = []
    for rec in records_for(world, ids, per=2):
        plan = Plan.from_dict({"steps": rec["plan"]})
        for sid, q in sorted(reference_queries(rec["template"], rec["params"]).items()):
            step = plan.by_id()[sid]
            for f in FAULTS:
                m = f.apply(q, step, cat)
                if m is not None:
                    out.append((f, rec, sid, m, bool(validator.check(step, m))))
    return out


# ------------------------------------------------------------------ operators

def test_fault_ids_are_unique_and_grouped():
    assert len({f.id for f in FAULTS}) == len(FAULTS) == len(FAULT_BY_ID)
    assert {f.group for f in FAULTS} == {STRUCTURAL, SEMANTIC}
    assert [f.id for f in FAULTS if not f.run] == ["sql_write_appended", "cy_write_appended"]


def test_the_reference_queries_pass_the_validator_with_the_case_rule_off(world, validator):
    for rec in records_for(world, ALL, per=2):
        assert baseline_problems(rec, validator) == []


def test_every_fault_applies_to_some_reference_query_and_changes_it(world, cat, validator):
    got = {f.id for f, *_ in mutants(world, cat, validator)}
    assert got == {f.id for f in FAULTS}, sorted({f.id for f in FAULTS} - got)


def test_a_fault_never_applies_to_a_step_of_another_store(world, cat):
    for rec in records_for(world, ALL, per=1):
        plan = Plan.from_dict({"steps": rec["plan"]})
        for sid, q in reference_queries(rec["template"], rec["params"]).items():
            for f in FAULTS:
                if plan.by_id()[sid].store not in f.stores:
                    assert f.apply(q, plan.by_id()[sid], cat) is None


def test_structural_faults_built_to_be_caught_are_always_rejected(world, cat, validator):
    must = {"sql_unknown_table", "sql_limit_changed", "sql_write_appended", "cy_unknown_label",
            "cy_unknown_relationship", "cy_write_appended", "es_unknown_field", "es_drop_doc_type",
            "es_extra_size", "bind_renamed"}
    for f, rec, sid, m, rejected in mutants(world, cat, validator):
        if f.id in must:
            assert rejected, (f.id, rec["template"], sid, m)


def test_semantic_faults_are_valid_against_the_catalogue(world, cat, validator):
    """The group label is a claim: a semantic fault must pass the validator, or the report would count a
    structural fault as semantic. The one exception is documented: removing a filter also removes a bind."""
    allowed = {"cy_drop_filter"}
    for f, rec, sid, m, rejected in mutants(world, cat, validator):
        if f.group == SEMANTIC and f.id not in allowed:
            assert not rejected, (f.id, rec["template"], sid, m, validator.check(
                Plan.from_dict({"steps": rec["plan"]}).by_id()[sid], m))


def test_the_wrong_entity_fault_reproduces_the_s05_005_class(world, cat, validator):
    rec = next(r for r in records_for(world, ["D01"], per=1))
    plan = Plan.from_dict({"steps": rec["plan"]})
    step = plan.by_id()["a"]
    m = FAULT_BY_ID["cy_wrong_entity"].apply(reference_queries("D01", rec["params"])["a"], step, cat)
    assert "zz:Question" in m and "RETURN DISTINCT zz.id" in m
    assert any("Question node" in e for e in validator.check(step, m))


def test_semantic_operators_on_known_queries(cat):
    sql_step = Plan.from_dict({"steps": [{"id": "a", "op": "retrieve", "store": "sql", "question": "q", "output": "user"}]}).by_id()["a"]
    q = "SELECT id FROM users WHERE reputation >= 5000"
    assert FAULT_BY_ID["sql_flip_comparison"].apply(q, sql_step, cat) == "SELECT id FROM users WHERE reputation < 5000"
    assert FAULT_BY_ID["sql_scale_number"].apply(q, sql_step, cat) == "SELECT id FROM users WHERE reputation >= 10001"
    assert FAULT_BY_ID["sql_drop_predicate"].apply(q, sql_step, cat) == "SELECT id FROM users"
    two = "SELECT id FROM posts WHERE post_type_id = 1 AND score >= 3"
    assert FAULT_BY_ID["sql_drop_predicate"].apply(two, sql_step, cat) == "SELECT id FROM posts WHERE post_type_id = 1"
    assert FAULT_BY_ID["sql_string_suffix"].apply("SELECT 1 WHERE name = 'a'", sql_step, cat) == "SELECT 1 WHERE name = 'ax'"
    g_step = Plan.from_dict({"steps": [{"id": "a", "op": "retrieve", "store": "graph", "question": "q", "output": "tag"}]}).by_id()["a"]
    cy = 'MATCH (q:Question)-[:TAGGED_WITH]->(:Tag {name: "x"}) RETURN DISTINCT q.id AS id'
    assert FAULT_BY_ID["cy_reverse_direction"].apply(cy, g_step, cat) == 'MATCH (q:Question)<-[:TAGGED_WITH]-(:Tag {name: "x"}) RETURN DISTINCT q.id AS id'
    assert FAULT_BY_ID["cy_drop_filter"].apply(cy, g_step, cat) == "MATCH (q:Question)-[:TAGGED_WITH]->(:Tag) RETURN DISTINCT q.id AS id"
    assert FAULT_BY_ID["cy_scale_number"].apply("MATCH (a {id: 7}) RETURN a.id LIMIT 5", g_step, cat) == "MATCH (a {id: 15}) RETURN a.id LIMIT 5"
    assert FAULT_BY_ID["cy_swap_relationship"].apply(cy, g_step, cat) != cy


# ------------------------------------------------------------------ verdicts

@pytest.mark.parametrize("rejected,dyn,want", [
    (True, "silent", "rejected"), (True, "no_effect", "rejected"), (False, "no_effect", "harmless"),
    (False, "silent", "silent"), (False, "flagged_holds", "flagged"), (False, "flagged_violated", "violated"),
    (False, "store_error", "stopped"), (False, "premise_guard", "stopped"), (False, "incomplete", "stopped"),
    (False, "not_run", "unrun_pass"), (True, "not_run", "rejected"),
])
def test_final_verdict(rejected, dyn, want):
    assert final_verdict(rejected, dyn) == want


class Scripted:
    """A store that answers the original query with ``right`` and every other query with ``other``."""
    def __init__(self, original, right, other=(), exhausted=True, boom=False):
        self.original, self.right, self.other, self.exhausted, self.boom, self.calls = original, list(right), list(other), exhausted, boom, []

    def fetch(self, step, query, binds, limit):
        self.calls.append(query)
        if self.boom and query != self.original:
            raise RuntimeError("syntax error at or near")
        values = self.right if query == self.original else self.other
        return FetchResult(list(values), exhausted=self.exhausted if query != self.original else True)


def s05(world):
    rec = next(r for r in records_for(world, ["S05"], per=3) if r["gold"])
    return rec, reference_queries("S05", rec["params"])["a"]


def run_faults(world, cat, validator, client, ids=("cy_unknown_label", "cy_reverse_direction", "cy_write_appended", "cy_scale_number")):
    rec, q = s05(world)
    ex = Executor({Store.GRAPH: client(q, rec)}, retries=0)
    rows = inject_record(rec, ex, validator, cat, tuple(FAULT_BY_ID[i] for i in ids))
    return {r["fault"]: r for r in rows}, rec


def test_a_wrong_but_valid_query_answered_with_exact_is_silent(world, cat, validator):
    rows, _ = run_faults(world, cat, validator, lambda q, rec: Scripted(q, rec["gold"], other=[10**9]))
    assert rows["cy_unknown_label"]["final"] == "rejected" and rows["cy_unknown_label"]["static_rejected"]
    assert rows["cy_reverse_direction"]["final"] == "silent" and rows["cy_reverse_direction"]["dyn"] == "silent"
    assert rows["cy_write_appended"]["final"] == "rejected" and rows["cy_write_appended"]["dyn"] == "not_run"
    assert rows["cy_unknown_label"]["dyn"] == "silent"             # it would have been silent without the validator


def test_a_mutant_that_does_not_change_the_answer_is_harmless(world, cat, validator):
    rows, _ = run_faults(world, cat, validator, lambda q, rec: Scripted(q, rec["gold"], other=rec["gold"]))
    assert rows["cy_reverse_direction"]["final"] == "harmless"
    assert rows["cy_unknown_label"]["final"] == "rejected"          # rejected although it would have been harmless


def test_a_store_error_is_a_stop_not_a_silent_error(world, cat, validator):
    rows, _ = run_faults(world, cat, validator, lambda q, rec: Scripted(q, rec["gold"], boom=True))
    assert rows["cy_reverse_direction"]["final"] == "stopped" and rows["cy_reverse_direction"]["dyn"] == "store_error"


def test_a_truncated_answer_is_flagged_and_its_label_is_checked_against_gold(world, cat, validator):
    ok, rec = run_faults(world, cat, validator, lambda q, rec: Scripted(q, rec["gold"], other=rec["gold"][:1], exhausted=False))
    assert ok["cy_reverse_direction"]["final"] == "flagged"                       # SUBSET, and a subset of gold
    bad, _ = run_faults(world, cat, validator, lambda q, rec: Scripted(q, rec["gold"], other=[10**9], exhausted=False))
    assert bad["cy_reverse_direction"]["final"] == "violated"                     # SUBSET, but 10**9 is not in gold


def test_one_question_with_no_applicable_fault_yields_no_rows(world, cat, validator):
    rec, q = s05(world)
    ex = Executor({Store.GRAPH: Scripted(q, rec["gold"])})
    assert inject_record(rec, ex, validator, cat, (FAULT_BY_ID["sql_flip_comparison"],)) == []


def test_a_broken_record_is_reported_as_a_setup_error(world, cat, validator):
    rec, q = s05(world)
    rows = inject_record(dict(rec, template="NOPE"), Executor({}), validator, cat)
    assert rows[0]["final"] == "setup_error"


def test_the_cap_per_fault_bounds_the_work(world, cat, validator):
    rec, q = s05(world)
    ex = Executor({Store.GRAPH: Scripted(q, rec["gold"], other=[1])})
    taken = {}
    rows = inject_record(rec, ex, validator, cat, (FAULT_BY_ID["cy_unknown_label"],), taken, 1, random.Random(1))
    assert len(rows) == 1 and taken == {"cy_unknown_label": 1}
    assert inject_record(rec, ex, validator, cat, (FAULT_BY_ID["cy_unknown_label"],), taken, 1, random.Random(1)) == []


# ------------------------------------------------------------------ report and command line

def row(fault, group, final, dyn, store="graph", tpl="S05"):
    return {"question_id": "q", "template": tpl, "step": "a", "store": store, "fault": fault, "group": group,
            "final": final, "dyn": dyn, "query": "x"}


def test_the_report_counts_and_uses_exact_intervals():
    rows = ([row("f", SEMANTIC, "silent", "silent")] * 3 + [row("f", SEMANTIC, "harmless", "no_effect")] * 5
            + [row("f", SEMANTIC, "flagged", "flagged_holds")] * 2 + [row("g", STRUCTURAL, "rejected", "silent")] * 4
            + [row("g", STRUCTURAL, "rejected", "no_effect")] * 2 + [row("x", STRUCTURAL, "infra", "infra")])
    text = render(rows)
    assert "16 mutants over 1 questions (1 rows excluded" in text
    sem = next(line for line in text.splitlines() if line.startswith("| semantic"))
    assert "| 10 | 5 | 0 | 0 | 2 | 0 | 3 | 60.0% [14.7, 94.7] of 5 |" in sem          # 3 unsound of 5 effective; exact interval for 3 of 5
    stru = next(line for line in text.splitlines() if line.startswith("| structural"))
    assert stru.endswith("| 6 | 0 | 6 | 0 | 0 | 0 | 0 | 0.0% [0.0, 60.2] of 4 | 4 |")                      # 2 harmless rejections leave 4 effective, 4 silent without the validator
    assert "Mutants that were not caught" in text and "WARNING" not in text
    assert "WARNING" in render([row("sql_write_appended", STRUCTURAL, "unrun_pass", "not_run")])


def test_the_command_line_end_to_end_and_the_baseline_guard(world, cat, tmp_path, capsys):
    rec, q = s05(world)
    qfile = tmp_path / "q.jsonl"
    qfile.write_text(json.dumps(rec) + "\n")
    out, rep = tmp_path / "f.json", tmp_path / "f.md"
    clients = {Store.GRAPH: Scripted(q, rec["gold"], other=[10**9])}
    assert main(["--questions", str(qfile), "--out", str(out), "--report", str(rep), "--faults", "cy_reverse_direction,cy_unknown_label"],
                clients=clients, catalogue=cat) == 0
    saved = json.loads(out.read_text())
    assert {r["fault"] for r in saved["rows"]} == {"cy_reverse_direction", "cy_unknown_label"} and rep.read_text().startswith("# Fault injection")
    assert main(["--questions", str(qfile), "--faults", "nope"], clients=clients, catalogue=cat) == 2
    assert main(["--questions", str(qfile), "--only", "D01"], clients=clients, catalogue=cat) == 2
    broken = dataclasses.replace(cat, graph=GraphCatalogue(frozenset(), frozenset(), frozenset()))
    assert main(["--questions", str(qfile)], clients=clients, catalogue=broken) == 3
    assert "unmodified reference queries" in capsys.readouterr().err
