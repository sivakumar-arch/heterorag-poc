"""Batch 102: N1 correct-empty controls (templates D01E and D05E), the empty-gold rules of ``check_gold``, the
scoring of an empty answer against an empty gold, and the N1 condition of the live validation.

The synthetic world is small, so the thresholds are scaled as in the class D tests (T = 10, H = 40). The synthetic
bodies carry no word of five letters, so a copy of the oracle gets bodies with a known vocabulary (one word for
the intermediate cell of D05E, and words inside code, pre and TeX spans that must never be chosen). The cell
conditions are checked with direct queries on the oracle views, not with the gold functions.
"""
import json
import re
import shutil
import sqlite3
from types import SimpleNamespace

import pytest

from benchmark_v2 import agent_baseline, catalogue_d, catalogue_n1, pool_counts, questions, union
from benchmark_v2.reference_queries import reference_queries
from benchmark_v2.run_slice import answer_scores, classify
from benchmark_v2.validate_instances import n1_condition, validate_record
from heterorag.federation import Executor, FetchResult, Plan, Store
from heterorag.federation.answer import Answer
from heterorag.federation.guarantee import Guarantee
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)

T, H = 10, 40
BODY_WORD = "zygomorphic"
HIDDEN = ("preonlyword", "codeonlyword", "texonlyword", "vdotsword", "parenword", "bracketword", "envword")
NAME = "Zebedee"           # a name: in every body, always capitalised, in no title (Batch 102b)
MOSTLY_LOWER = "quokkas"   # lower case in 19 of 20 occurrences: eligible at the 90% rule


@pytest.fixture(scope="module")
def n1_path(world, tmp_path_factory):
    _, _, db_path = world
    path = tmp_path_factory.mktemp("n1") / "oracle_n1.sqlite"
    shutil.copy(db_path, path)
    con = sqlite3.connect(path)
    con.execute("UPDATE posts SET body = '<p>body ' || id || ' " + BODY_WORD + " " + NAME + "</p><pre>preonlyword</pre>"
                "<code>codeonlyword</code> $\\texonlyword$ &nbsp; is $5 so \\vdotsword "
                "\\(parenword\\) \\[bracketword\\] \\begin{align} envword \\end{align}' "
                "WHERE post_type_id = 1 AND body <> ''")
    ids = [i for (i,) in con.execute("SELECT id FROM posts WHERE post_type_id = 1 AND body <> '' ORDER BY id LIMIT 20")]
    for k, i in enumerate(ids):
        word = MOSTLY_LOWER.capitalize() if k == 0 else MOSTLY_LOWER
        con.execute("UPDATE posts SET body = body || ' <p>' || ? || '</p>' WHERE id = ?", (word, i))
    con.commit()
    con.close()
    return str(path)


@pytest.fixture(scope="module")
def n1_db(n1_path):
    db = union.open_union(n1_path)
    yield db
    db.close()


@pytest.fixture(scope="module")
def generated(n1_db):
    return questions.build_records(catalogue_n1.TEMPLATES, n1_db, seed=11, per_template=4, t_independent=T,
                                   h_cap=H)


def _by(records):
    out = {}
    for r in records:
        out.setdefault((r["template"], r["stratum"]), []).append(r)
    return out


# ---------------------------------------------------------------- templates and records

def test_n1_templates_reuse_the_d01_and_d05_shapes():
    d01e, d05e = catalogue_n1.TEMPLATES
    assert (d01e.id, d05e.id) == ("D01E", "D05E")
    assert d01e.paraphrases == catalogue_d.D01.paraphrases and d01e.plan == catalogue_d.D01.plan
    assert d05e.paraphrases == catalogue_d.D05.paraphrases and d05e.plan == catalogue_d.D05.plan
    assert d01e.empty_step == d05e.empty_step == "b"
    assert d01e.cls == d05e.cls == "D" and d01e.split == d05e.split == "dev"
    assert not any(s.must_exist for t in catalogue_n1.TEMPLATES for s in t.plan)
    assert all(t.empty_step is None for t in catalogue_d.TEMPLATES)
    assert {"D01E", "D05E"} <= {t.id for t in questions.all_templates()}


def test_every_cell_is_generated_with_an_empty_gold_and_the_n1_fields(generated):
    records, reports = generated
    cells = _by(records)
    for t in ("D01E", "D05E"):
        for stratum in (questions.N1_FINAL, questions.N1_INTERMEDIATE):
            assert len(cells.get((t, stratum), [])) == 2, (t, stratum, {r.template: r.strata for r in reports})
    for r in records:
        assert r["gold"] == [] and r["control"] == "N1" and r["generator"] == "2.2"
        assert r["empty_at"] == ("final" if r["stratum"] == questions.N1_FINAL else "b")
        assert not any(s.get("must_exist") for s in r["plan"])
        assert r["text"] == r["paraphrases"][(int(r["question_id"].split("-")[1]) - 1) % 3]


def _askers(db, tag):
    return {u for (u,) in db.execute("SELECT DISTINCT a.user_id FROM v_asked a JOIN v_tagged t "
                                     "ON t.question_id = a.question_id WHERE t.tag = ?", (tag,))}


def _users_at_least(db, r):
    return {u for (u,) in db.execute("SELECT id FROM v_users WHERE reputation >= ?", (r,))}


def _title_has(db, word):
    return {q for q, title in db.execute("SELECT post_id, title FROM v_question_docs")
            if word in union.tokens(title)}


def _tagged(db, tag):
    return {q for (q,) in db.execute("SELECT question_id FROM v_tagged WHERE tag = ?", (tag,))}


def test_d01e_cells_hold_on_the_oracle_views(n1_db, generated):
    cells = _by(generated[0])
    top = n1_db.execute("SELECT max(reputation) FROM v_users").fetchone()[0]
    for r in cells[("D01E", questions.N1_FINAL)]:
        a, b = _askers(n1_db, r["params"]["tag"]), _users_at_least(n1_db, r["params"]["min_reputation"])
        assert a and b and not (a & b)
        best = max(n1_db.execute(f"SELECT reputation FROM v_users WHERE id IN ({','.join(map(str, a))})"))[0]
        below = n1_db.execute("SELECT count(*) FROM v_users WHERE reputation > ? AND reputation < ?",
                              (best, r["params"]["min_reputation"])).fetchone()[0]
        assert below == 0                         # R is the lowest reputation above the tag's best asker
    for r in cells[("D01E", questions.N1_INTERMEDIATE)]:
        assert r["params"]["min_reputation"] == (top // 1000 + 1) * 1000
        assert not _users_at_least(n1_db, r["params"]["min_reputation"])
        assert T < len(_askers(n1_db, r["params"]["tag"])) <= H


def test_d05e_cells_hold_on_the_oracle_views(n1_db, generated):
    cells = _by(generated[0])
    for r in cells[("D05E", questions.N1_FINAL)]:
        tag, word = r["params"]["tag"], r["params"]["word"]
        docs = _title_has(n1_db, word)
        assert len(docs) > T and _tagged(n1_db, tag) and not (docs & _tagged(n1_db, tag))
    for r in cells[("D05E", questions.N1_INTERMEDIATE)]:
        assert r["params"]["word"] in (BODY_WORD, MOSTLY_LOWER) and not _title_has(n1_db, r["params"]["word"])
        assert T < len(_tagged(n1_db, r["params"]["tag"])) <= H


def test_words_inside_code_pre_and_tex_spans_are_never_chosen(n1_db):
    ctx = questions.Ctx(n1_db, 1, T, H)
    words = catalogue_n1._body_words(ctx)
    assert BODY_WORD in words and not set(HIDDEN) & set(words)
    assert NAME.lower() not in words                          # always capitalised: a name, never chosen
    assert MOSTLY_LOWER in words                              # 19 of 20 in lower case: at least 90%
    assert catalogue_n1.clean_body("<p>a <b>bold</b></p><pre>x\ny</pre> $$z$$ &amp;").split() == ["a", "bold"]
    # Batch 102c: TeX outside a recognised dollar span (an odd dollar sign shifts the pairing) never survives
    assert catalogue_n1.clean_body("a $5 b \\vdots c").split() == ["a", "$5", "b", "c"]
    assert catalogue_n1.clean_body("a \\vdots b \\(x\\) c \\[y\\] d \\begin{matrix} e \\end{matrix} f").split() == \
        ["a", "b", "c", "d", "f"]


def test_stemming_guard_rejects_inflected_forms_both_ways():
    g = catalogue_n1.TitleGuard
    assert g(["regressions"]).clashes("regression")
    assert g(["regression"]).clashes("regressions")
    assert g(["model"]).clashes("modelling")
    assert g(["test"]).clashes("tests")
    assert g(["analysis"]).clashes("analyses")
    assert not g(["bayesian", "prior"]).clashes("regression")
    assert not g(["an", "of"]).clashes("regression")


def test_adding_n1_templates_leaves_the_repeat_questions_unchanged(world):
    _, db, _ = world
    a, _ = questions.build_records(catalogue_d.TEMPLATES, db, 5, 3, T, H)
    b, _ = questions.build_records(list(catalogue_d.TEMPLATES) + list(catalogue_n1.TEMPLATES), db, 5, 3, T, H)
    assert [r for r in b if r["template"] in {"D01", "D04", "D05", "D07"}] == a
    assert all("control" not in r and "empty_at" not in r for r in a)


# ---------------------------------------------------------------- check_gold

def test_check_gold_accepts_an_empty_gold_only_for_n1_and_checks_the_cell():
    G, e = questions.Gold, catalogue_n1.D01E
    fin, mid = {"stratum": questions.N1_FINAL}, {"stratum": questions.N1_INTERMEDIATE}
    chk = lambda p, g: questions.check_gold(e, p, g, t_ind=200, h_cap=10_000)
    assert chk(fin, G([], {"graph": 300, "sql": 40})) is None
    assert "not empty" in chk(fin, G([7], {"graph": 300, "sql": 40}))
    assert "a sub-result is empty" in chk(fin, G([], {"graph": 300, "sql": 0}))
    assert "class P" in chk(fin, G([], {"graph": 150, "sql": 40}))
    assert chk(mid, G([], {"graph": 300, "sql": 0})) is None
    assert "sql sub-result is not empty" in chk(mid, G([], {"graph": 300, "sql": 3}))
    assert "not within" in chk(mid, G([], {"graph": 150, "sql": 0}))
    assert "not within" in chk(mid, G([], {"graph": 20_000, "sql": 0}))
    assert "unknown stratum" in chk({"stratum": "x"}, G([], {"graph": 300, "sql": 40}))
    d05e = catalogue_n1.D05E
    assert questions.check_gold(d05e, mid, G([], {"graph": 300, "doc": 0}), 200, 10_000) is None
    assert "doc sub-result" in questions.check_gold(d05e, mid, G([], {"graph": 300, "doc": 2}), 200, 10_000)
    # every other template still rejects an empty gold
    assert questions.check_gold(catalogue_d.D01, {}, G([], {"graph": 300, "sql": 40}), 200, 10_000) == \
        "set size 0, the answer is empty"


# ---------------------------------------------------------------- scoring

def _agent(values, kind="set"):
    return agent_baseline.AgentResult(kind=kind, values=values, stop="answered")


@pytest.mark.parametrize("got,gold", [([], []), ([], [1, 2]), ([3], []), ([1, 3], [1, 2]), ([1, 2], [1, 2])])
def test_pipeline_and_agent_score_the_same_and_empty_against_empty_is_one(got, gold):
    rec = {"answer_type": "ids", "gold": gold}
    pipe = answer_scores(Answer("ids", got, Guarantee.EXACT), rec)
    agent = agent_baseline.score(rec, _agent(got))
    assert {k: pipe[k] for k in ("precision", "recall", "f1")} == {k: agent[k] for k in ("precision", "recall", "f1")}
    assert (classify(Answer("ids", got, Guarantee.EXACT), rec) == "correct") == (agent["outcome"] == "correct")
    if not got and not gold:
        assert pipe == {"precision": 1.0, "recall": 1.0, "f1": 1.0} and agent["outcome"] == "correct"


def test_scores_of_non_empty_gold_rows_are_unchanged():
    rec = {"answer_type": "ids", "gold": [1, 2]}
    assert answer_scores(Answer("ids", [], Guarantee.SUBSET), rec) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert answer_scores(Answer("ids", [1, 3], Guarantee.EXACT), rec)["f1"] == 0.5


# ---------------------------------------------------------------- live validation (fake stores)

class _OracleStores:
    """Store clients that answer the reference queries of D01 and D05 from the oracle views."""

    def __init__(self, path, empty_doc=False):      # own connection per call: the executor uses threads
        self.path, self.empty_doc = path, empty_doc

    def client(self, store):
        outer = self

        class C:
            def fetch(self, step, query, binds, limit):
                db = union.open_union(outer.path)
                try:
                    return FetchResult(sorted(self._ids(db, query)))
                finally:
                    db.close()

            def _ids(self, db, query):
                if store is Store.GRAPH:
                    tag = json.loads(re.search(r'name: ("(?:[^"\\]|\\.)*")', query).group(1))
                    return _askers(db, tag) if "User" in query else _tagged(db, tag)
                if store is Store.SQL:
                    return _users_at_least(db, int(re.search(r">= (\d+)", query).group(1)))
                word = json.loads(query)["query"]["bool"]["filter"][1]["match"]["title"]["query"]
                return set() if outer.empty_doc else _title_has(db, word)
        return C()

    def executor(self):
        return Executor({s: self.client(s) for s in (Store.GRAPH, Store.SQL, Store.DOC)})


def test_n1_instances_validate_on_stores_that_agree_with_the_oracle(n1_path, generated):
    ex = _OracleStores(n1_path).executor()
    for r in generated[0]:
        assert reference_queries(r["template"], r["params"]) == reference_queries(r["template"][:3], r["params"])
        out = validate_record(r, ex)
        assert out.status == "match", (r["question_id"], r["stratum"], out)


def test_validation_flags_a_final_cell_whose_step_is_empty_on_the_stores(n1_path, generated):
    # a store whose title search finds nothing: the answer is still empty, but the final cell no longer holds
    ex = _OracleStores(n1_path, empty_doc=True).executor()
    r = _by(generated[0])[("D05E", questions.N1_FINAL)][0]
    out = validate_record(r, ex)
    assert out.status == "mismatch" and "step b should be non-empty" in out.reason


def test_n1_condition_rules():
    plan = Plan.from_dict({"steps": [questions.step_record(s, None) for s in catalogue_n1.D01E.plan]})
    rec = lambda where: {"control": "N1", "empty_at": where}
    full = lambda a, b: {"a": SimpleNamespace(value=frozenset(a)), "b": SimpleNamespace(value=frozenset(b)),
                         "c": SimpleNamespace(value=frozenset())}
    assert n1_condition(rec("final"), plan, full({1}, {2})) == ""
    assert "step b should be non-empty" in n1_condition(rec("final"), plan, full({1}, set()))
    assert n1_condition(rec("b"), plan, full({1}, set())) == ""
    assert "step b should be empty" in n1_condition(rec("b"), plan, full({1}, {2}))
    assert "step a should be non-empty" in n1_condition(rec("b"), plan, full(set(), set()))
    assert "unknown empty_at" in n1_condition(rec("z"), plan, full({1}, {2}))
    assert n1_condition({}, plan, full(set(), set())) == ""          # not an N1 record


# ---------------------------------------------------------------- pool counts

def test_pool_counts_count_the_n1_supply_per_cell(n1_db):
    for t in catalogue_n1.TEMPLATES:
        c = pool_counts.count(t, questions.Ctx(n1_db, 5, T, H), set(), T, H)
        assert c["pool"] == c["passing_check_gold"] > 0
        assert set(c["pool_by_stratum"]) == {questions.N1_FINAL, questions.N1_INTERMEDIATE}
    assert pool_counts.key("D05E", {"tag": "r", "word": "a"}, "N1 final") == \
        pool_counts.key("D05E", {"tag": "r", "word": "b", "stratum": "N1 final"})


# ---------------------------------------------------------------- Batch 102b: one cell per tag

def test_no_tag_is_used_by_both_cells_of_a_template(n1_db, generated):
    cells = _by(generated[0])
    for t in ("D01E", "D05E"):
        fin = {r["params"]["tag"] for r in cells[(t, questions.N1_FINAL)]}
        mid = {r["params"]["tag"] for r in cells[(t, questions.N1_INTERMEDIATE)]}
        assert not fin & mid, t
    for t in catalogue_n1.TEMPLATES:                          # the whole candidate pool, not only the chosen
        cand = t.sample(questions.Ctx(n1_db, 3, T, H), 10_000)
        by_tag = {}
        for p in cand:
            by_tag.setdefault(p["tag"], set()).add(p["stratum"])
        assert all(len(s) == 1 for s in by_tag.values()), t.id
        assert {questions.N1_FINAL, questions.N1_INTERMEDIATE} <= {p["stratum"] for p in cand}


def test_the_word_list_lists_exactly_the_eligible_words(n1_path, tmp_path, capsys):
    out = tmp_path / "words.tsv"
    assert catalogue_n1.main(["--oracle", n1_path, "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0] == "word\tbodies\tlower_share"
    words = {ln.split("\t")[0]: (int(ln.split("\t")[1]), float(ln.split("\t")[2])) for ln in lines[1:]}
    assert BODY_WORD in words and words[BODY_WORD][1] == 1.0 and words[MOSTLY_LOWER] == (20, 0.95)
    assert not (set(HIDDEN) | {NAME.lower()}) & set(words)
    assert f"{len(words)} eligible words" in capsys.readouterr().out
