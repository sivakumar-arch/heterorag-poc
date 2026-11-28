"""Batch 105b: decisions of 2026-11-28 on the first-contact templates.

0. the pool report gives, per H3 rank, the share of thresholds at exactly 2 and exactly 102 and how many ranks each
   word keeps;
1. the validator's cap is computed from the oracle, per step, as the size of the table of the step's output entity;
   a store step that reaches it is a data mismatch, named by step; the report records the caps and the oracle;
2. H1: at most one question per holder set; single-holder badges capped at half of the questions when they are
   more than half of the accepted badges; the pool reported by holder band;
3. a threshold condition must change the answer (S1, H3, H4);
4. the H3 exclusion list;
and the two reporting fixes (the not-validatable reason without the size; duplicates removed by the full
parameters in the pool count).
"""
import json
from collections import Counter

import pytest

from benchmark_v2 import catalogue_fc as fc, pool_counts, questions
from benchmark_v2.questions import Ctx, Gold
from benchmark_v2.validate_instances import main as validate_main, validate_record
from heterorag.federation import Executor, FetchResult, Mode, Op, Plan, Step, Store
from tests.benchmark_v2.test_fc_105 import HH, T, H, db, gen, path  # noqa: F401  (fixtures)
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)


class Listing:
    """A store that returns ``ids`` under the clients' contract: at most ``limit`` values, ``exhausted`` False
    exactly when more existed."""

    def __init__(self, ids):
        self.ids = list(ids)

    def fetch(self, step, query, binds, limit):
        return FetchResult(self.ids[:limit], exhausted=len(self.ids) <= limit)


# ---------------------------------------------------------------- 1: caps per entity from the oracle

def test_validator_caps_are_the_oracle_table_sizes(db):
    caps = questions.validator_caps(db)
    for entity, view in (("user", "v_users"), ("question", "v_questions"), ("answer", "v_answers")):
        assert caps[entity] == db.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
    assert caps["tag"] == db.execute("SELECT count(*) FROM tags").fetchone()[0]
    assert Ctx(db, 1).validator_caps() == caps and Ctx(None, 1).validator_caps() is None


def test_the_executor_caps_each_step_by_its_own_cap():
    stores = {Store.SQL: Listing(range(1, 51)), Store.GRAPH: Listing(range(100, 300))}
    caps = {"user": 50, "question": 150}
    ex = Executor(stores, cap=10, step_cap=lambda s: caps[s.output])
    u = Plan((Step("u", Op.RETRIEVE, Store.SQL, None, "users", output="user"),))
    q = Plan((Step("q", Op.RETRIEVE, Store.GRAPH, None, "questions", output="question"),))
    ru, rq = ex.run(u, {"u": "q1"}).records["u"], ex.run(q, {"q": "q2"}).records["q"]
    assert not ru.cap_hit and len(ru.value) == 50          # exactly the cap: complete
    assert rq.cap_hit and len(rq.value) == 150             # more than the cap: capped
    plain = Executor(stores, cap=10).run(u, {"u": "q1"}).records["u"]
    assert plain.cap_hit and len(plain.value) == 10         # without step caps: the one cap, as before
    assert ex.cap_for(u.steps[0]) == 50 and Executor(stores, cap=10).cap_for(u.steps[0]) == 10


def test_a_step_equal_to_the_whole_table_matches_and_one_above_it_is_a_mismatch():
    plan = [{"id": "a", "op": "retrieve", "store": "sql", "output": "user", "kind": "set", "question": "users"}]
    rec = {"question_id": "X-001", "template": "S01", "split": "dev", "plan": plan, "params": {},
           "answer_type": "ids", "gold": list(range(1, 21))}
    q = {"a": "SELECT id FROM users"}

    def run(ids, cap):
        ex = Executor({Store.SQL: Listing(ids)}, cap=cap, step_cap=lambda s: cap)
        import benchmark_v2.validate_instances as vi
        orig = vi.reference_queries
        vi.reference_queries = lambda t, p: q
        try:
            return validate_record(rec, ex)
        finally:
            vi.reference_queries = orig

    assert run(range(1, 21), 20).status == "match"
    out = run(range(1, 22), 20)
    assert out.status == "mismatch" and out.cap_steps == ["a"] and "a (user, cap 20)" in out.reason


def test_the_report_records_the_caps_and_counts_cap_mismatches(world, tmp_path, capsys):
    from tests.federation.test_reference_and_validation import records_for
    _, db_, db_path = world
    recs = records_for(world, ["S05"], per=1)
    qfile, out = tmp_path / "q.jsonl", tmp_path / "rep.json"
    qfile.write_text("\n".join(json.dumps(r) for r in recs))

    class Huge:
        def fetch(self, step, query, binds, limit):
            return FetchResult(list(range(limit)), exhausted=False)

    assert validate_main(["--questions", str(qfile), "--oracle", str(db_path), "--out", str(out)],
                         clients={Store.GRAPH: Huge(), Store.DOC: Huge(), Store.SQL: Huge()}) == 1
    rep = json.loads(out.read_text())
    assert rep["validator_caps"] == questions.validator_caps(db_)
    assert all(o["status"] == "mismatch" and o["cap_steps"] for o in rep["outcomes"])
    printed = capsys.readouterr().out
    assert "validator caps per entity" in printed
    row = next(l for l in printed.splitlines() if l.startswith("S05"))
    assert row.split()[-1] == str(len(recs))                  # the cap column counts them


def test_the_not_validatable_rule_uses_the_contexts_caps(db):
    c = Ctx(db, 3, T, HH)
    c._cache["validator_caps"] = {"user": 1, "question": 1, "answer": 1, "tag": 1}
    _, rep = questions.generate_template(fc.H3, c, 2)
    assert rep.accepted == 0 and any(r.startswith("not validatable: step ") and "ids" not in r
                                     for r in rep.rejected)


# ---------------------------------------------------------------- 2: H1 holder sets and bands

def _h1(badge, holders, band, stratum="over-cap", year=2015):
    return ({"stratum": stratum, "badge": badge, "year": year, "holders": holders, "holder_band": band}, Gold(3))


def test_h1_takes_one_question_per_holder_set():
    rows = {"over-cap": [_h1("a", "h1", "2 to 10", year=2015), _h1("b", "h1", "2 to 10", year=2016),
                         _h1("c", "h2", "2 to 10", year=2017)]}
    chosen = questions.select(fc.H1, rows, 3)
    assert sorted(p["badge"] for p, _ in chosen) == ["a", "c"]


def test_h1_single_holder_badges_are_capped_at_half_when_they_are_more_than_half():
    years = iter(range(1990, 2100))
    single = [_h1(f"s{i}", f"hs{i}", "1", year=next(years)) for i in range(8)]
    multi = [_h1(f"m{i}", f"hm{i}", "2 to 10", year=next(years)) for i in range(2)]
    notes = Counter()
    chosen = questions.select(fc.H1, {"over-cap": single + multi}, 6, notes)
    bands = [p["holder_band"] for p, _ in chosen]
    # quota over-cap 4 of 6 (all rows here are over-cap): 6 // 2 = 3 single-holder badges, then a multi-holder one
    assert bands.count("1") == 3 and bands.count("2 to 10") == 1
    assert any(k.startswith("H1 share cap holder_band=1: 8 of 10 badges, at most 3 of 6") for k in notes)
    # at half or below, no cap
    half = [_h1(f"s{i}", f"hs{i}", "1", year=next(years)) for i in range(3)] + \
        [_h1(f"m{i}", f"hm{i}", "more than 10", year=next(years)) for i in range(3)]
    chosen = questions.select(fc.H1, {"over-cap": half}, 4)
    assert [p["holder_band"] for p, _ in chosen].count("1") == 3


def test_h1_records_carry_the_holder_set_and_band(db):
    recs, _ = gen(db, [fc.H1], 6)
    assert recs
    for r in recs:
        holders = {u for (u,) in db.execute("SELECT user_id FROM badges WHERE name = ?", (r["params"]["badge"],))}
        assert r["params"]["holders"] == questions.id_hash(holders) == r["step_hashes"]["a"]
        assert r["params"]["holder_band"] == fc.holder_band(len(holders))
    assert len({r["params"]["holders"] for r in recs}) == len(recs)
    assert [fc.holder_band(n) for n in (1, 2, 10, 11)] == ["1", "2 to 10", "2 to 10", "more than 10"]


def test_pool_counts_keep_both_h1_strata_and_report_the_bands(db):
    r = pool_counts.count(fc.H1, Ctx(db, 3, T, H), set(), T, H)
    c = Ctx(db, 3, T, H)
    cands = fc.H1.sample(c, pool_counts.UNLIMITED)
    assert r["distinct"] == len({questions._canonical(p) for p in cands})          # by full parameters
    assert r["pool_keys"] == len({p["badge"] for p in cands}) <= r["pool"]
    assert sum(v["candidates"] for v in r["pool_by_band"].values()) == r["pool"]
    assert sum(v["keys"] for v in r["pool_by_band"].values()) == r["pool_keys"]


def test_pool_counts_remove_duplicates_by_the_full_parameters_and_exclude_by_the_key(db):
    import dataclasses
    from tests.benchmark_v2.test_batch103 import INTERSECT, _t
    two = [{"stratum": "over-cap", "badge": "x", "year": 1}, {"stratum": "within-cap", "badge": "x", "year": 2},
           {"stratum": "within-cap", "badge": "x", "year": 2}]
    t = dataclasses.replace(_t(INTERSECT), id="H1", sample=lambda c, n: [dict(p) for p in two],
                            gold=lambda c, p: Gold([1], steps={"a": 10, "b": 20}))
    c = Ctx(db, 1, T, H)
    c._cache["validator_caps"] = {}
    r = pool_counts.count(t, c, set(), T, H)
    assert (r["candidates"], r["distinct"], r["pool"], r["pool_keys"]) == (3, 2, 2, 1)
    assert r["pool_by_stratum"] == {"over-cap": 1, "within-cap": 1}
    c2 = Ctx(db, 1, T, H)
    c2._cache["validator_caps"] = {}
    used = {pool_counts.key("H1", {"badge": "x"})}
    assert pool_counts.count(t, c2, used, T, H)["pool"] == 0          # a used badge excludes both strata


# ---------------------------------------------------------------- 3: a threshold must change the answer

@pytest.mark.parametrize("t,step", [(fc.S1, "a"), (fc.H3, "b"), (fc.H4, "b"), (fc.H4P, "b")])
def test_a_threshold_that_removes_nothing_is_rejected(t, step):
    steps = {"a": 5, "b": 5, "c": 5}
    assert t.condition_step == step
    assert questions.check_gold(t, {}, Gold([1, 2, 3, 4, 5], steps=steps), T, H) == \
        "the threshold removes nothing from the answer"
    assert questions.check_gold(t, {}, Gold([1, 2, 3, 4], steps=steps), T, H) != \
        "the threshold removes nothing from the answer"


def test_generated_s1_h3_h4_answers_are_strictly_smaller_than_without_the_threshold(db, monkeypatch):
    monkeypatch.setattr(fc, "H4_YEARS", (2015, 2016, 2017, 2018))
    monkeypatch.setattr(fc, "H4_RANKS", (2, 4, 6))
    recs, reps = gen(db, [fc.S1, fc.H3, fc.H4], 6, HH)
    for r in recs:
        step = {"S1": "a", "H3": "b", "H4": "b"}[r["template"]]
        assert len(r["gold"]) < r["step_sizes"][step], r["question_id"]


# ---------------------------------------------------------------- 0 and 4: the H3 report and exclusions

def test_h3_pool_reports_threshold_marks_and_ranks_per_word(db):
    r = pool_counts.count(fc.H3, Ctx(db, 3, T, HH), set(), T, HH)
    for st, d in r["per_stratum"].items():
        assert set(d["marks"]) == {2, 102} and d["n"] == r["pool_by_stratum"][st]
    assert sum(r["ranks_per_key"].values()) == r["pool_keys"]
    assert sum(k * v for k, v in r["ranks_per_key"].items()) == r["pool"]


def test_h3_exclusions_are_the_two_misspellings():
    assert fc.H3_EXCLUDED == {"coeffcients": "misspelling", "seperately": "misspelling"}
