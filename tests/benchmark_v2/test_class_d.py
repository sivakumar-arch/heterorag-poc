"""Class D: templates, gold answers and the rules that separate D from P.

The synthetic world is small, so these tests use scaled thresholds (T = 10, H = 40). Gold is
checked against the synthetic generator's own data structures, which share no code with the
oracle, the views or the gold functions.
"""
import json

import pytest

from benchmark_v2 import catalogue_d, catalogue_sp, questions
from benchmark_v2.reference_queries import reference_queries
from benchmark_v2.validate_instances import validate_record
from heterorag.federation import Executor, FetchResult, Plan, Store
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)
from tests.federation.fake_descriptors import catalogue

T, H = 10, 40


@pytest.fixture(scope="module")
def generated_d(world):
    _, db, _ = world
    return questions.build_records(catalogue_d.TEMPLATES, db, seed=11, per_template=4, t_independent=T, h_cap=H)


def _by(records):
    out = {}
    for r in records:
        out.setdefault(r["template"], []).append(r)
    return out


def test_templates_ids_classes_and_split():
    assert [t.id for t in catalogue_d.TEMPLATES] == ["D01", "D04", "D05", "D07"]
    assert {t.id: t.cls for t in catalogue_d.TEMPLATES} == {"D01": "D", "D04": "D", "D05": "D", "D07": "A"}
    assert all(t.split == "dev" for t in catalogue_d.TEMPLATES)
    all_ids = [t.id for t in questions.all_templates()]
    assert len(all_ids) == len(set(all_ids))
    for t in catalogue_d.TEMPLATES:
        assert len(t.paraphrases) == 3
        assert Plan.from_dict({"steps": [questions.step_record(s, None) for s in t.plan]}).validate() == []


def test_every_template_yields_instances_in_more_than_one_stratum(generated_d):
    records, reports = generated_d
    got = _by(records)
    for t in catalogue_d.TEMPLATES:
        assert len(got[t.id]) == 4, t.id
    assert not [r.template for r in reports if r.accepted < 4]
    assert len({r["stratum"] for r in got["D01"]}) >= 2
    assert len({r["stratum"] for r in got["D05"]}) == 2


def test_gold_against_independent_truth(world, generated_d):
    truth, _, _ = world
    got = _by(generated_d[0])

    def with_word(w):
        return {q for q, i in truth.questions.items() if w in i["title"].split() and i["body"]}

    for r in got["D05"]:
        tag, w = r["params"]["tag"], r["params"]["word"]
        tagged = {q for (q, t) in truth.tagged if t == tag}
        assert r["gold"] == sorted(tagged & with_word(w))
        assert r["intermediates"] == {"graph": len(tagged), "doc": len(with_word(w))}
    for r in got["D01"]:
        tag, rep = r["params"]["tag"], r["params"]["min_reputation"]
        askers = {truth.questions[q]["owner"] for (q, t) in truth.tagged if t == tag}
        askers = {u for u in askers if u in truth.users}
        assert r["gold"] == sorted(u for u in askers if truth.users[u]["reputation"] >= rep)
        assert r["intermediates"] == {"graph": len(askers),
                                      "sql": sum(1 for u in truth.users.values() if u["reputation"] >= rep)}
    for tid in ("D04", "D07"):
        for r in got[tid]:
            holders = truth.rare_badges[r["params"]["badge"]]
            qs = {q for q, i in truth.questions.items() if i["owner"] in holders}
            tags = sorted({t for (q, t) in truth.tagged if q in qs})
            assert r["gold"] == (tags if tid == "D04" else len(tags))
            assert r["intermediates"] == {"sql": len(holders)} and r["bound"] == {"a": len(holders)}


def test_class_d_records_obey_the_class_rules(generated_d):
    for r in generated_d[0]:
        assert r["class"] == ("A" if r["template"] == "D07" else "D") and r["thresholds"] == {"T": T, "H": H}
        if r["answer_type"] == "ids":
            assert 1 <= len(r["gold"]) <= questions.MAX_SET
        else:
            assert isinstance(r["gold"], int) and r["gold"] >= 1
        sizes = r["intermediates"].values()
        if r["independent"]:
            assert max(sizes) > T                    # an independent formulation needs more than T
            assert min(sizes) <= H                   # and one set is small enough to bind
            assert r["bound"] == {}
        else:
            assert r["template"] in ("D04", "D07")
            assert max(r["bound"].values()) <= H


def test_strata_state_the_real_size_band(generated_d):
    got = _by(generated_d[0])
    for r in got["D05"]:
        df = r["intermediates"]["doc"]
        assert (r["stratum"] == f"df {T}-{H}") == (df <= H), r["stratum"]
    for r in got["D01"]:
        side, _, band = r["stratum"].partition(" large ")
        g, s = r["intermediates"]["graph"], r["intermediates"]["sql"]
        assert (side == "sql") == (s >= g)
        assert (band == f"{T}-{H}") == (max(g, s) <= H)


def test_d07_strata_are_the_bound_set_size(generated_d):
    for r in _by(generated_d[0])["D07"]:
        lo = int(r["stratum"].split("-")[0].split("+")[0])
        assert lo <= r["bound"]["a"] <= H and r["bound"]["a"] > 10
        assert r["plan"][2]["op"] == "count" and r["answer_type"] == "scalar"


def test_d04_cannot_be_asked_of_one_store(world):
    """The graph and document stores do not know badges, which is why the bind is necessary."""
    cat = catalogue()
    assert "Badge" not in cat.graph.labels and not any("badge" in p.lower() for p in cat.graph.properties)
    assert not any("badge" in f.lower() for f in cat.doc.fields)
    assert "badges" in cat.sql.tables
    plan = catalogue_d.D04.plan
    assert plan[1].bind is not None and plan[1].bind.input == "a" and plan[1].inputs == ("a",)


def test_the_gold_plan_of_d04_runs_through_the_executor_with_a_bind(world, generated_d):
    truth, _, _ = world

    class Sql:
        def fetch(self, step, query, binds, limit):
            badge = query.split("name = ")[1].strip("'")
            return FetchResult(sorted(truth.rare_badges[badge]))

    class Graph:
        seen = []

        def fetch(self, step, query, binds, limit):
            ids = set(binds["a"])
            self.seen.append(sorted(ids))
            qs = {q for q, i in truth.questions.items() if i["owner"] in ids}
            return FetchResult(sorted({t for (q, t) in truth.tagged if q in qs}))

    graph = Graph()
    for r in _by(generated_d[0])["D04"] + _by(generated_d[0])["D07"]:
        out = validate_record(r, Executor({Store.SQL: Sql(), Store.GRAPH: graph}))
        assert out.status == "match", (r["template"], out)
    assert graph.seen and all(len(ids) <= H for ids in graph.seen)
    q = reference_queries("D04", {"badge": "o'x"})
    assert "name = 'o''x'" in q["a"] and "$a" in q["b"]


def test_check_gold_class_d_rules():
    G, d = questions.Gold, catalogue_d.D01
    chk = lambda g, **kw: questions.check_gold(d, {}, g, t_ind=200, h_cap=10_000, **kw)
    assert chk(G([1], {"graph": 300, "sql": 40})) is None
    assert "class P" in chk(G([1], {"graph": 200, "sql": 40}))
    assert "no set can be bound" in chk(G([1], {"graph": 20_000, "sql": 30_000}))
    assert chk(G([1], {"graph": 20_000, "sql": 3_000})) is None            # bind the small side
    assert "intermediate" in chk(G([1]))
    dep = dict(independent=False)
    assert "bound sets" in chk(G([1], {"sql": 5}, **dep))
    assert "above H" in chk(G([1], {"sql": 5}, bound={"a": 10_001}, **dep))
    assert chk(G([1], {"sql": 5}, bound={"a": 5}, **dep)) is None
    a = catalogue_d.D07                                              # class A follows the class D rules
    assert "above H" in questions.check_gold(a, {}, G(7, {"sql": 5}, bound={"a": 10_001}, **dep), 200, 10_000)
    assert questions.check_gold(a, {}, G(7, {"sql": 5000}, bound={"a": 5000}, **dep), 200, 10_000) is None
    assert "missing" in questions.check_gold(a, {}, G(None, {"sql": 5}, bound={"a": 5}, **dep), 200, 10_000)
    assert "empty" in questions.check_gold(d, {}, G([], {"graph": 300, "sql": 3}), 200, 10_000)


def test_class_p_threshold_is_the_one_in_force(world):
    """Class P instances are rejected against the configured T, not a constant."""
    P = catalogue_sp.P01
    assert questions.check_gold(P, {}, questions.Gold([1], {"graph": 50}), t_ind=200) is None
    assert "above T=10" in questions.check_gold(P, {}, questions.Gold([1], {"graph": 50}), t_ind=10)


def test_records_are_deterministic(world):
    _, db, _ = world
    a, _ = questions.build_records(catalogue_d.TEMPLATES, db, 5, 3, T, H)
    b, _ = questions.build_records(catalogue_d.TEMPLATES, db, 5, 3, T, H)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_catalogue_d_does_not_import_the_system_under_test():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(catalogue_d.__file__).read_text())
    mods = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods.isdisjoint({"heterorag", "psycopg2", "neo4j", "elasticsearch"})


def test_a_paraphrase_that_names_a_tag_says_that_it_is_a_tag():
    # CP1 diagnosis: 'askers of "regression" questions' can be read as a text search; the gold means the tag
    import re
    for t in catalogue_d.TEMPLATES:
        for text in t.paraphrases:
            if "{tag}" in text:
                assert re.search(r"\b(tag|tags|tagged)\b", text.split("{tag}")[0], re.I), (t.id, text)
    assert not any('askers of "{tag}" questions' in text for t in catalogue_d.TEMPLATES for text in t.paraphrases)
