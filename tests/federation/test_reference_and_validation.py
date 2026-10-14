"""Reference queries and the instance validator (without live stores)."""
import json
import sqlite3

import pytest

from benchmark_v2 import catalogue_sp, questions, union
from benchmark_v2.reference_queries import QUERIES, cy_lit, es_title_has_word, reference_queries, sql_lit
from benchmark_v2.validate_instances import main as validate_main, validate_record
from heterorag.federation import Executor, FetchResult, Store
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)


# The synthetic world is far smaller than the real dump, so class D uses scaled thresholds there.
D_T, D_H = 10, 40


def records_for(world, ids, per=3, seed=5):
    _, db, _ = world
    chosen = [t for t in questions.all_templates() if t.id in ids]
    recs, _ = questions.build_records([t for t in chosen if t.cls != "D"], db, seed, per)
    d, _ = questions.build_records([t for t in chosen if t.cls == "D"], db, seed, per, D_T, D_H)
    return recs + d


def test_every_template_has_reference_queries_for_exactly_its_retrieve_steps(world):
    for t in questions.all_templates():
        assert t.id in QUERIES
    for r in records_for(world, [t.id for t in questions.all_templates()], per=1):
        q = reference_queries(r["template"], r["params"])
        assert set(q) == {s["id"] for s in r["plan"] if s["op"] == "retrieve"}, r["template"]
    with pytest.raises(KeyError):
        reference_queries("D99", {})


def test_literals_are_escaped():
    assert sql_lit("o'brien") == "'o''brien'" and sql_lit(7) == "7"
    assert cy_lit('a"b\\c') == json.dumps('a"b\\c') and cy_lit(12) == "12"
    q = json.loads(es_title_has_word('quo"te'))
    assert q["query"]["bool"]["filter"][1]["match"]["title"]["query"] == 'quo"te'


def test_document_queries_are_single_query_objects(world):
    for r in records_for(world, ["S09", "P01", "P03", "P04", "D05"], per=1):
        for sid, text in reference_queries(r["template"], r["params"]).items():
            step = next(s for s in r["plan"] if s["id"] == sid)
            if step.get("store") == "doc":
                assert set(json.loads(text)) == {"query"}


def run_sql_on_oracle(db, text):
    """The reference SQL targets PostgreSQL; the only dialect difference used is EXTRACT(YEAR)."""
    text = text.replace("EXTRACT(YEAR FROM creation_date)", "CAST(substr(creation_date, 1, 4) AS INTEGER)")
    return [r[0] for r in db.execute(text).fetchall()]


def test_sql_reference_queries_reproduce_gold_on_the_oracle(world):
    _, db, _ = world
    for r in records_for(world, ["S01", "S02", "S03", "P02", "P03"], per=4):
        q = reference_queries(r["template"], r["params"])
        if r["template"] in ("S01", "S02"):
            assert run_sql_on_oracle(db, q["a"]) == [r["gold"]]
        elif r["template"] == "S03":
            assert run_sql_on_oracle(db, q["a"]) == r["gold"]                 # order included
        else:
            assert len(run_sql_on_oracle(db, q["a"])) == r["intermediates"]["sql"]


# ------------------------------------------------------------------ validator logic

class Scripted:
    def __init__(self, fn): self.fn, self.calls = fn, 0
    def fetch(self, step, query, binds, limit):
        self.calls += 1
        return FetchResult(list(self.fn(step, query, binds)))


def only_graph(rec_ids):
    return {Store.GRAPH: Scripted(lambda s, q, b: rec_ids), Store.SQL: Scripted(lambda *a: []),
            Store.DOC: Scripted(lambda *a: [])}


def test_validator_classifies_match_mismatch_abstain_error(world):
    s05 = next(r for r in records_for(world, ["S05"], per=3))
    gold = list(s05["gold"])
    assert validate_record(s05, Executor(only_graph(gold))).status == "match"
    extra = validate_record(s05, Executor(only_graph(gold + [10**9])))
    assert (extra.status, extra.extra, extra.missing) == ("mismatch", 1, 0)
    missing = validate_record(s05, Executor(only_graph(gold[1:])))
    assert (missing.status, missing.missing, missing.extra) == ("mismatch", 1, 0)
    assert missing.examples["missing"] == [gold[0]]

    class Boom:
        def fetch(self, *a): raise ConnectionError("down")
    ab = validate_record(s05, Executor({Store.GRAPH: Boom()}, retries=0))
    assert ab.status == "abstain" and "down" in ab.reason

    broken = dict(s05, plan=[{"id": "a", "op": "join"}])
    assert validate_record(broken, Executor(only_graph(gold))).status == "error"
    assert validate_record(dict(s05, template="NOPE"), Executor(only_graph(gold))).status == "error"


def test_validator_scalar_and_ranked(world):
    s01 = records_for(world, ["S01"], per=1)[0]
    sql = {Store.SQL: Scripted(lambda *a: [s01["gold"]])}
    assert validate_record(s01, Executor(sql)).status == "match"
    sql = {Store.SQL: Scripted(lambda *a: [s01["gold"] + 1])}
    o = validate_record(s01, Executor(sql))
    assert o.status == "mismatch" and o.examples["gold"] == s01["gold"]
    s03 = records_for(world, ["S03"], per=1)
    if s03:
        r = s03[0]
        rev = {Store.SQL: Scripted(lambda *a: list(reversed(r["gold"])))}
        o = validate_record(r, Executor(rev))
        assert o.status == "mismatch" and o.reason == "order differs"


def test_cli_writes_report_and_cleaned_file_and_hides_test_gold(world, tmp_path, capsys):
    recs = records_for(world, ["S05", "S09"], per=2)
    gold_by = {r["question_id"]: r for r in recs}
    qfile = tmp_path / "q.jsonl"
    qfile.write_text("\n".join(json.dumps(r) for r in recs))

    # a graph store that is right for the first S05 record only; documents always right
    first = next(r for r in recs if r["template"] == "S05")
    ans = {"cur": None}

    class Graph:
        def fetch(self, step, query, binds, limit):
            return FetchResult(list(first["gold"]))

    class Doc:
        def fetch(self, step, query, binds, limit):
            word = json.loads(query)["query"]["bool"]["filter"][1]["match"]["title"]["query"]
            rec = next(r for r in recs if r["template"] == "S09" and r["params"]["word"] == word)
            return FetchResult(list(rec["gold"]))
    out, clean = tmp_path / "rep.json", tmp_path / "clean.jsonl"
    code = validate_main(["--questions", str(qfile), "--out", str(out), "--clean", str(clean)],
                         clients={Store.GRAPH: Graph(), Store.DOC: Doc()})
    report = json.loads(out.read_text())
    kept = [json.loads(l)["question_id"] for l in clean.read_text().splitlines()]
    assert len(report) == len(recs)
    assert first["question_id"] in kept and all(gold_by[k]["template"] in ("S05", "S09") for k in kept)
    printed = capsys.readouterr().out
    assert "instances reproduced by the stores" in printed
    n_bad = len(recs) - len(kept)
    assert code == (1 if n_bad else 0)
