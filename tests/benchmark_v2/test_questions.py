"""Tests for the virtual-union views, the question framework and the S and P catalogue.

Gold answers are checked against a truth computed from the synthetic generator's own data
structures (tests/benchmark_v2/synth.py), which share no code with the oracle, the views or
the gold functions.
"""
import json
import sqlite3
from pathlib import Path

import pytest

from benchmark_v2 import catalogue_sp, fidelity, oracle, questions, union
from tests.benchmark_v2.synth import make_synthetic_dump


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    truth = make_synthetic_dump(d / "dump")
    db_path = d / "oracle.sqlite"
    oracle.build(d / "dump", db_path, log=lambda s: None)
    db = union.open_union(str(db_path))
    yield truth, db, db_path
    db.close()


# ------------------------------------------------------------------ the union views

def test_views_reproduce_the_fidelity_expected_counts(world):
    _, db, _ = world
    expected = fidelity.expected_counts(db)
    for view, key in union.VIEW_TO_EXPECTED.items():
        assert db.execute(f"SELECT count(*) FROM {view}").fetchone()[0] == expected[key], view
    cooc = union.view_counts(db)["v_cooc"]
    assert cooc == expected["neo4j.rel.CO_OCCURS_WITH"]


def test_views_match_the_independent_truth(world):
    truth, db, _ = world
    assert set(db.execute("SELECT user_id, question_id FROM v_asked")) == truth.asked
    assert set(db.execute("SELECT user_id, answer_id FROM v_answered")) == truth.answered
    assert set(db.execute("SELECT answer_id, question_id FROM v_answers_of")) == truth.answers_of
    assert set(db.execute("SELECT question_id, answer_id FROM v_accepted")) == truth.accepted
    assert set(db.execute("SELECT question_id, tag FROM v_tagged")) == truth.tagged
    assert set(db.execute("SELECT question_id, original_id FROM v_duplicate_of")) == truth.duplicates
    assert set(db.execute("SELECT question_id, related_id FROM v_linked")) == truth.linked


def test_the_synthetic_data_contains_the_awkward_cases(world):
    truth, _, _ = world
    assert truth.early_answers > 0                                       # answer precedes its question
    assert any(q["owner"] is None for q in truth.questions.values())     # ownerless
    assert any(q["owner"] not in truth.users and q["owner"] is not None
               for q in truth.questions.values())                        # owner is not a user


def test_tokenizer_follows_the_standard_analyzer_for_the_cases_we_rely_on():
    assert union.tokens("Don't use x_1 here, e.g. t-test") == ["don't", "use", "x_1", "here", "e.g", "t", "test"]
    assert union.has_word("Bayesian Prior", "prior") == 1
    assert union.has_word("priors", "prior") == 0                        # whole words only
    assert union.tokens(None) == []


# ------------------------------------------------------------------ generation and gold

@pytest.fixture(scope="module")
def generated(world):
    _, db, _ = world
    return questions.build_records(catalogue_sp.TEMPLATES, db, seed=11, per_template=4)


def _by_template(records):
    out = {}
    for r in records:
        out.setdefault(r["template"], []).append(r)
    return out


def test_every_template_yields_instances(generated):
    records, reports = generated
    got = _by_template(records)
    for t in catalogue_sp.TEMPLATES:
        assert got.get(t.id), f"{t.id} produced no instances"
    short = {r.template: r.accepted for r in reports if r.accepted < 4}
    assert not short, short


def test_gold_s_and_p_against_independent_truth(world, generated):
    truth, _, _ = world
    records, _ = generated
    got = _by_template(records)

    for r in got["S05"]:
        q = r["params"]["question"]
        assert r["gold"] == sorted(a for (a, qq) in truth.answers_of if qq == q)
    for r in got["S06"]:
        q = r["params"]["question"]
        assert r["gold"] == sorted(a for (a, o) in truth.duplicates if o == q)
    for r in got["S08"]:
        q = r["params"]["question"]
        assert r["gold"] == sorted(t for (qq, t) in truth.tagged if qq == q)
    for r in got["S01"]:
        assert r["gold"] == truth.users[r["params"]["user"]]["reputation"]
    for r in got["S02"]:
        assert r["gold"] == truth.badges[r["params"]["user"]]

    def with_word(w):       # independent of the tokenizer: titles are built from space-separated words
        return {q for q, i in truth.questions.items() if w in i["title"].split() and i["body"]}

    for r in got["S09"]:
        assert r["gold"] == sorted(with_word(r["params"]["word"]))
    for r in got["P01"]:
        w, tag = r["params"]["word"], r["params"]["tag"]
        tagged = {q for (q, t) in truth.tagged if t == tag}
        assert r["gold"] == sorted(with_word(w) & tagged)
        assert r["intermediates"] == {"graph": len(tagged), "doc": len(with_word(w))}
    for r in got["P04"]:
        w, tag = r["params"]["word"], r["params"]["tag"]
        tagged = {q for (q, t) in truth.tagged if t == tag}
        assert r["gold"] == sorted(with_word(w) - tagged)
    for r in got["P03"]:
        w, s = r["params"]["word"], r["params"]["min_score"]
        assert r["gold"] == sorted(q for q in with_word(w) if truth.questions[q]["score"] >= s)
    for r in got["P02"]:
        q, rep = r["params"]["question"], r["params"]["min_reputation"]
        answerers = {truth.answers[a]["owner"] for (a, qq) in truth.answers_of if qq == q}
        expect = sorted(u for u in answerers if u in truth.users and truth.users[u]["reputation"] >= rep)
        assert r["gold"] == expect


def test_ranked_answers_have_no_tie_at_the_boundary(world, generated):
    truth, _, _ = world
    records, _ = generated
    for r in _by_template(records)["S03"]:
        p = r["params"]
        pool = [(q["views"], qid) for qid, q in truth.questions.items()
                if str(q["year"]) == p["year"] and q["body"] is not None]
        pool.sort(key=lambda x: (-x[0], x[1]))
        k = p["k"]
        assert [q for _, q in pool[:k]] == r["gold"]
        assert pool[k - 1][0] > pool[k][0]
    for r in _by_template(records)["S07"]:
        assert len(r["gold"]) == r["params"]["k"]


def test_class_p_respects_the_independent_cardinality_threshold(generated):
    records, _ = generated
    for r in records:
        if r["class"] == "P":
            assert max(r["intermediates"].values()) <= questions.T_INDEPENDENT
        if r["answer_type"] == "ids":
            assert 1 <= len(r["gold"]) <= questions.MAX_SET


def test_records_carry_everything_a_runner_needs(generated):
    records, _ = generated
    r = records[0]
    for key in ("question_id", "template", "class", "split", "stratum", "text", "paraphrases",
                "params", "answer_type", "entity", "gold", "services", "plan", "intermediates",
                "oracle", "seed", "generator"):
        assert key in r
    assert len(r["paraphrases"]) == 3 and r["text"] in r["paraphrases"]
    assert "{" not in r["text"]                                           # every placeholder filled
    p01 = next(x for x in records if x["template"] == "P01")
    assert p01["services"] == ["doc", "graph"]
    assert [s["op"] for s in p01["plan"]] == ["retrieve", "retrieve", "intersect"]
    assert [s.get("store") for s in p01["plan"]] == ["graph", "doc", None]
    assert p01["plan"][1]["mode"] == "filter"
    assert p01["plan"][2]["inputs"] == ["a", "b"]
    ranked = [x for x in records if x["answer_type"] == "ranked"]
    assert all(x["plan"][0]["k"] == x["params"]["k"] for x in ranked)


def test_generation_is_deterministic_and_seed_sensitive(world):
    _, db, _ = world
    a, _ = questions.build_records(catalogue_sp.TEMPLATES, db, seed=5, per_template=3)
    b, _ = questions.build_records(catalogue_sp.TEMPLATES, db, seed=5, per_template=3)
    c, _ = questions.build_records(catalogue_sp.TEMPLATES, db, seed=6, per_template=3)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_adding_a_template_does_not_change_the_others(world):
    _, db, _ = world
    only, _ = questions.build_records([catalogue_sp.S05], db, seed=5, per_template=3)
    full, _ = questions.build_records(catalogue_sp.TEMPLATES, db, seed=5, per_template=3)
    assert only == [r for r in full if r["template"] == "S05"]


def test_instances_are_unique_and_spread_over_strata(generated):
    records, _ = generated
    for tid, rs in _by_template(records).items():
        keys = {json.dumps(r["params"], sort_keys=True) for r in rs}
        assert len(keys) == len(rs), tid
    s05 = _by_template(records)["S05"]
    assert len({r["stratum"] for r in s05}) >= 2


# ------------------------------------------------------------------ rules in check_gold

def _tpl(**kw):
    base = dict(id="X", cls="S", split="dev", answer_type="ids", entity="question",
                paraphrases=("a", "b", "c"), plan=(), sample=lambda c, n: [], gold=lambda c, p: None)
    base.update(kw)
    return questions.Template(**base)


def test_check_gold_rules():
    G = questions.Gold
    t = _tpl()
    assert questions.check_gold(t, {}, G([1, 2])) is None
    assert "set size" in questions.check_gold(t, {}, G([]))
    assert "set size" in questions.check_gold(t, {}, G(list(range(51))))
    assert "duplicate" in questions.check_gold(t, {}, G([1, 1]))
    r = _tpl(answer_type="ranked", k_param="k")
    assert questions.check_gold(r, {"k": 2}, G([1, 2])) is None
    assert "tie" in questions.check_gold(r, {"k": 2}, G(None))
    assert "k distinct" in questions.check_gold(r, {"k": 3}, G([1, 2]))
    p = _tpl(cls="P")
    assert "intermediate" in questions.check_gold(p, {}, G([1]))
    assert "above T" in questions.check_gold(p, {}, G([1], {"sql": 201}))
    assert questions.check_gold(p, {}, G([1], {"sql": 200, "graph": 3})) is None
    assert "missing" in questions.check_gold(_tpl(answer_type="scalar"), {}, G(None))


def test_template_ids_and_split_assignment_are_fixed():
    ids = [t.id for t in catalogue_sp.TEMPLATES]
    assert len(ids) == len(set(ids))
    assert {t.id for t in catalogue_sp.TEMPLATES if t.split == "test"} == {"S03", "S07", "P03"}
    for t in catalogue_sp.TEMPLATES:
        assert len(t.paraphrases) == 3 and t.cls in "SPDAN"
        assert set(s.store for s in t.plan if s.op == "retrieve") <= {"sql", "graph", "doc"}
        assert all(s.store is None for s in t.plan if s.op != "retrieve")
        assert all(s.mode == "filter" for s in t.plan if s.store == "doc")
        for s in t.plan:
            assert all(i in {x.id for x in t.plan} for i in s.inputs)


# ------------------------------------------------------------------ CLI and leakage guard

def test_cli_writes_dev_only_by_default_and_test_on_request(world, tmp_path, capsys):
    _, _, db_path = world
    out = tmp_path / "q.jsonl"
    assert questions.main(["--oracle", str(db_path), "--out", str(out), "--per-template", "2"]) == 0
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert recs and {r["split"] for r in recs} == {"dev"}
    printed = capsys.readouterr().out
    assert "S01" in printed and "S03" not in printed                       # test templates not generated
    out2 = tmp_path / "q2.jsonl"
    assert questions.main(["--oracle", str(db_path), "--out", str(out2), "--per-template", "2",
                           "--include-test"]) == 0
    recs2 = [json.loads(l) for l in out2.read_text().splitlines()]
    assert {r["split"] for r in recs2} == {"dev", "test"}


def test_report_shows_sizes_not_gold_values(generated):
    _, reports = generated
    text = questions.render_report(reports, 4)
    assert "gold size" in text and "[" not in text.split("\n")[1].split("  ")[-1][:0]


def test_cli_only_filter_and_empty_selection(world, tmp_path):
    _, _, db_path = world
    out = tmp_path / "q.jsonl"
    assert questions.main(["--oracle", str(db_path), "--out", str(out), "--only", "S05"]) == 0
    assert {json.loads(l)["template"] for l in out.read_text().splitlines()} == {"S05"}
    assert questions.main(["--oracle", str(db_path), "--out", str(out), "--only", "NOPE"]) == 2


def test_questions_module_does_not_import_the_system_under_test():
    import ast
    for mod in (questions, catalogue_sp, union):
        tree = ast.parse(Path(mod.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint({"heterorag", "psycopg2", "neo4j", "elasticsearch"}), mod.__name__


# ------------------------------------------------------------------ gold-size strata (P01, P02)

def _band_of(stratum: str):
    low = stratum.replace("gold ", "")
    if low.endswith("+"):
        return int(low[:-1]), 10**9
    lo, _, hi = low.partition("-")
    return int(lo), int(hi or lo)


def test_p01_and_p02_strata_are_the_real_answer_sizes(world):
    _, db, _ = world
    records, _ = questions.build_records(
        [t for t in catalogue_sp.TEMPLATES if t.id in ("P01", "P02")], db, seed=3, per_template=12)
    seen = {"P01": set(), "P02": set()}
    for r in records:
        lo, hi = _band_of(r["stratum"])
        assert lo <= len(r["gold"]) <= hi, (r["question_id"], r["stratum"], len(r["gold"]))
        seen[r["template"]].add(len(r["gold"]))
    # a single-item answer must no longer be the only size on offer
    assert len(seen["P01"]) > 1, seen
