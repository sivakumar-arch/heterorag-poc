"""Batch 104: first-contact templates S1, S2 and H2 (class F).

Checks: wording as decided (H2 verbatim from the hand-in v1.2; S1 and S2 as decided on 2026-11-26); the gold of every
generated question re-derived with direct queries on the oracle views, not with the gold functions; the H2 selection
rule and exclusive-reading gold; draw caps; rank thresholds with ties; the reference queries on fake stores that
answer from the oracle; the supply report. The synthetic world is small, so H = 40 as in the class D tests.
"""
import json
import re
import shutil

import pytest

from benchmark_v2 import catalogue_fc as fc, pool_counts, questions, union
from benchmark_v2.questions import Ctx, Gold
from benchmark_v2.reference_queries import reference_queries
from benchmark_v2.validate_instances import validate_record
from heterorag.federation import Executor, FetchResult, Store
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)

T, H = 10, 40


@pytest.fixture(scope="module")
def generated(world):
    _, db, _ = world
    return questions.build_records(fc.TEMPLATES, db, seed=7, per_template=8, t_independent=T, h_cap=H)


def _by(records, t):
    return [r for r in records if r["template"] == t]


# ---------------------------------------------------------------- templates

def test_templates_wording_and_class():
    assert [t.id for t in fc.TEMPLATES] == ["S1", "S2", "H2"]
    assert all(t.cls == "F" for t in fc.TEMPLATES)
    assert fc.S1.paraphrases[0] == ('Which questions that are marked as duplicates of a question tagged "{tag}" '
                                    'have a score of at least {number}?')
    assert fc.S1.paraphrases[1] == ('Among the questions marked as duplicates of a question tagged "{tag}", '
                                    'list those with a score of {number} or more.')
    assert fc.S2.paraphrases[0] == ('Which users who asked a question tagged "{tag_a}" have not asked a question '
                                    'tagged "{tag_b}"?')
    assert fc.S2.paraphrases[2].endswith('who have never asked a question tagged "{tag_b}".')
    assert fc.H2.paraphrases == (                       # hand-in v1.2, verbatim
        'Which questions tagged with "{tag}" have either the word "{word}" in their title or an accepted answer?',
        'List questions carrying the tag "{tag}" that either feature the word "{word}" in their title or possess an accepted answer.',
        'Find any question with the "{tag}" tag that has an accepted answer or includes the word "{word}" in its title.')
    assert [s.op for s in fc.H2.plan] == ["retrieve", "retrieve", "retrieve", "intersect", "union"]
    assert fc.S1.group_caps == (("tag", 1),) and fc.H2.group_caps == (("tag", 1), ("word", 1))
    assert fc.S2.group_caps == (("tag_a", 1), ("tag_b", 2))


def test_records_carry_step_sizes_restriction_flag_and_answer_size(generated):
    records, reports = generated
    for t in ("S1", "S2", "H2"):
        assert _by(records, t), {r.template: (r.rejected, r.notes) for r in reports}
    for r in records:
        assert 1 <= r["answer_size"] == len(r["gold"]) <= questions.MAX_SET
        assert set(r["step_sizes"]) == {s["id"] for s in r["plan"] if s["op"] == "retrieve"}
        assert r["needs_restriction"] == sorted(s for s, n in r["step_sizes"].items() if n > H)
        assert "forms_agree" not in r and r["class"] == "F"


# ---------------------------------------------------------------- gold re-derived from the views

def _tagged(db, tag):
    return {q for (q,) in db.execute("SELECT question_id FROM v_tagged WHERE tag = ?", (tag,))}


def _askers(db, tag):
    return {u for (u,) in db.execute("SELECT DISTINCT a.user_id FROM v_asked a JOIN v_tagged t "
                                     "ON t.question_id = a.question_id WHERE t.tag = ?", (tag,))}


def test_s1_gold_and_threshold(world, generated):
    _, db, _ = world
    for r in _by(generated[0], "S1"):
        tag, n, k = r["params"]["tag"], r["params"]["number"], r["params"]["rank"]
        dups = {q for (q,) in db.execute("SELECT DISTINCT d.question_id FROM v_duplicate_of d JOIN v_tagged t "
                                         "ON t.question_id = d.original_id WHERE t.tag = ?", (tag,))}
        score = {q: s or 0 for q, s in db.execute("SELECT id, score FROM v_questions")}
        assert r["gold"] == sorted(q for q in dups if score[q] >= n)
        assert n >= 0 and sorted((score[q] for q in dups), reverse=True)[k - 1] == n
        assert r["stratum"] == f"rank {k:02d}"
        assert r["step_sizes"]["b"] == sum(1 for s in score.values() if s >= n)


def test_s2_gold(world, generated):
    _, db, _ = world
    for r in _by(generated[0], "S2"):
        a, b = _askers(db, r["params"]["tag_a"]), _askers(db, r["params"]["tag_b"])
        assert r["gold"] == sorted(a - b) and a & b and len(a) <= H and len(b) <= H


def test_h2_gold_selection_rule_and_exclusive_reading(world, generated):
    _, db, _ = world
    acc = {q for (q,) in db.execute("SELECT question_id FROM v_accepted")}
    for r in _by(generated[0], "H2"):
        qs, w = _tagged(db, r["params"]["tag"]), r["params"]["word"]
        word = {q for q, title in db.execute("SELECT post_id, title FROM v_question_docs") if w in union.tokens(title)}
        assert r["gold"] == sorted(qs & (word | acc))
        assert r["gold_exclusive"] == sorted((qs & word) ^ (qs & acc))
        assert len((qs & word) - acc) >= 3 and len((qs & acc) - word) >= 3


def test_draw_caps_hold(generated):
    recs = generated[0]
    for t, key in (("S1", "tag"), ("H2", "tag"), ("H2", "word"), ("S2", "tag_a")):
        vals = [r["params"][key] for r in _by(recs, t)]
        assert len(vals) == len(set(vals)), (t, key)
    tb = [r["params"]["tag_b"] for r in _by(recs, "S2")]
    assert max(tb.count(x) for x in tb) <= 2


def test_forms_that_disagree_are_rejected():
    g = Gold([1, 2], steps={"a": 5, "b": 5, "c": 3}, extra={"forms_agree": False})
    assert "two forms" in questions.check_gold(fc.H2, {}, g, T, H)


# ---------------------------------------------------------------- rank thresholds and ties

def _ctx(members, dups, score):
    c = Ctx(None, 3, T, H)
    c._cache.update({"tag_members": members, "fc_duplicates_of": dups, "fc_question_score": score})
    return c


def test_s1_ranks_ties_and_negative_thresholds():
    dups = {1: {10, 11, 12, 13, 14, 15}, 2: {20, 21, 22}, 3: {30, 31}}
    score = {10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 2, 20: 4, 21: -1, 22: -2, 30: -3, 31: -4}
    c = _ctx({"x": [1], "y": [2], "z": [3]}, dups, score)
    out = fc.s1_sample(c, 10)
    by_tag = {}
    for p in out:
        by_tag.setdefault(p["tag"], []).append((p["rank"], p["number"]))
    assert by_tag["x"] == [(2, 5)]                     # rank 5 gives 5 again: one candidate, the smaller rank
    assert "y" not in by_tag                           # the 2nd best score is -1: not used
    assert "z" not in by_tag                           # every threshold negative
    assert c.notes["S1 rank collision (same NUMBER as a smaller rank)"] == 1
    assert c.notes["S1 tags with a collision"] == 1 and c.notes["S1 tags with two or more ranks"] == 1
    assert c.notes["S1 threshold below 1 not used"] == 2


def test_s1_split_is_8_8_7_7_by_rank_with_quota():
    rows = {f"rank {k:02d}": [({"stratum": f"rank {k:02d}", "tag": f"t{k}-{i}"}, Gold([1])) for i in range(20)]
            for k in fc.S1_RANKS}
    chosen = questions.select(fc.S1, rows, 30)
    counts = {s: sum(p["stratum"] == s for p, _ in chosen) for s in rows}
    assert counts == {"rank 02": 8, "rank 05": 8, "rank 10": 7, "rank 20": 7}


# ---------------------------------------------------------------- reference queries on fake stores

class _OracleStores:
    def __init__(self, path):
        self.path = path

    def client(self, store):
        outer = self

        class C:
            def fetch(self, step, query, binds, limit):
                db = union.open_union(outer.path)
                try:
                    ids = sorted(self._ids(db, query))
                    return FetchResult(ids[:limit], exhausted=len(ids) <= limit)
                finally:
                    db.close()

            def _ids(self, db, q):
                if store is Store.GRAPH:
                    tag = json.loads(re.search(r'name: ("(?:[^"\\]|\\.)*")', q).group(1))
                    if "DUPLICATE_OF" in q:
                        return {d for (d,) in db.execute(
                            "SELECT DISTINCT d.question_id FROM v_duplicate_of d JOIN v_tagged t "
                            "ON t.question_id = d.original_id WHERE t.tag = ?", (tag,))}
                    if "ACCEPTED" in q:
                        acc = {x for (x,) in db.execute("SELECT question_id FROM v_accepted")}
                        return _tagged(db, tag) & acc
                    return _askers(db, tag) if "User" in q else _tagged(db, tag)
                if store is Store.SQL:
                    n = int(re.search(r">= (-?\d+)", q).group(1))
                    return {i for i, s in db.execute("SELECT id, score FROM v_questions") if (s or 0) >= n}
                w = json.loads(q)["query"]["bool"]["filter"][1]["match"]["title"]["query"]
                return {i for i, t in db.execute("SELECT post_id, title FROM v_question_docs") if w in union.tokens(t)}
        return C()

    def executor(self):
        return Executor({s: self.client(s) for s in (Store.GRAPH, Store.SQL, Store.DOC)})


def test_reference_queries_reproduce_the_gold_on_stores_that_agree_with_the_oracle(world, generated, tmp_path):
    _, _, db_path = world
    path = tmp_path / "o.sqlite"
    shutil.copy(db_path, path)
    ex = _OracleStores(str(path)).executor()
    for r in generated[0]:
        assert set(reference_queries(r["template"], r["params"])) == {s["id"] for s in r["plan"] if s["op"] == "retrieve"}
        out = validate_record(r, ex)
        assert out.status == "match", (r["question_id"], out)


# ---------------------------------------------------------------- supply report and word list

def test_pool_counts_report_per_stratum_numbers_notes_and_artificial_thresholds(world):
    _, db, _ = world
    r = pool_counts.count(fc.S1, Ctx(db, 5, T, H), set(), T, H)
    assert r["pool"] > 0 and set(r["per_stratum"]) <= {f"rank {k:02d}" for k in fc.S1_RANKS}
    assert all(d["number"] for d in r["per_stratum"].values())
    assert "artificial_threshold" in r and r["notes"]["S1 tags with a candidate"] > 0
    r2 = pool_counts.count(fc.H2, Ctx(db, 5, T, H), set(), T, H)
    assert r2["per_stratum"]["all"]["number"] is None and r2["step_sizes"]["c"]["max"] <= H


def test_h2_word_list_and_seeded_sample(world, tmp_path, monkeypatch, capsys):
    _, _, db_path = world
    out = tmp_path / "h2.tsv"
    assert fc.main(["--oracle", str(db_path), "--words", "H2", "--out", str(out)]) == 0
    rows = out.read_text().splitlines()
    assert rows[0] == "word\ttags" and len(rows) > 1
    monkeypatch.setattr(fc, "WORD_SAMPLE", 1)
    fc.main(["--oracle", str(db_path), "--words", "H2", "--out", str(out)])
    assert len((tmp_path / "h2.sample1.tsv").read_text().splitlines()) == 2
    assert "seeded sample" in capsys.readouterr().out


def test_s2_requires_overlapping_asker_sets():
    c = Ctx(None, 3, T, H)
    c._cache["askers_by_tag"] = {"a": {1, 2, 3, 4}, "disjoint": {7, 8, 9}, "overlap": {4, 9, 10}}
    out = fc.s2_sample(c, 10)
    pairs = {(p["tag_a"], p["tag_b"]) for p in out}
    assert ("a", "overlap") in pairs and ("a", "disjoint") not in pairs


# ---------------------------------------------------------------- Batch 104b

def test_s1_threshold_zero_is_not_used():
    c = _ctx({"x": [1]}, {1: {10, 11, 12, 13, 14}}, {10: 4, 11: 0, 12: 0, 13: 0, 14: 0})
    out = fc.s1_sample(c, 10)
    assert [(p["rank"], p["number"]) for p in out] == []          # rank 2 is 0, rank 5 is 0
    assert c.notes["S1 threshold below 1 not used"] == 2


def test_s1_rank_20_shortfall_is_filled_from_rank_10_and_recorded():
    from collections import Counter
    rows = {f"rank {k:02d}": [({"stratum": f"rank {k:02d}", "tag": f"t{k}-{i}"}, Gold([1]))
                              for i in range(3 if k == 20 else 20)] for k in fc.S1_RANKS}
    notes = Counter()
    chosen = questions.select(fc.S1, rows, 30, notes)
    counts = {s: sum(p["stratum"] == s for p, _ in chosen) for s in rows}
    assert counts == {"rank 02": 8, "rank 05": 8, "rank 10": 11, "rank 20": 3}
    assert notes["S1 shortfall of rank 20"] == 4 and notes["S1 shortfall of rank 20 filled from rank 10"] == 4
    rows2 = {k: list(v) for k, v in rows.items()}
    rows2["rank 05"] = rows2["rank 05"][:2]                       # no fallback for rank 05: not topped up
    assert sum(p["stratum"] == "rank 05" for p, _ in questions.select(fc.S1, rows2, 30)) == 2


def test_h2_word_may_not_be_a_component_of_the_tag_or_an_inflection_of_one():
    c = Ctx(None, 3, T, H)
    qs = list(range(1, 21))
    acc = frozenset(range(1, 7))                                 # 6 accepted
    words = {q: frozenset({"regressions", "puzzle", "bayes", "multicolinearity"}) for q in range(4, 13)}
    c._cache.update({"tag_members": {"logistic-regression": qs, "puzzle": qs}, "fc_accepted": acc,
                     "title_words": words})
    assert fc.h2_words_for_tag(c, "logistic-regression") == ["bayes", "puzzle"]
    assert fc.h2_words_for_tag(c, "puzzle") == ["bayes", "regressions"]
    assert fc.H2_EXCLUDED == {"multicolinearity": "misspelling"}


def test_questions_whose_unrestricted_gold_plan_exceeds_the_validator_cap_are_rejected():
    big = questions.VALIDATOR_CAP + 1
    assert "not validatable" in questions.check_gold(fc.S1, {}, Gold([1, 2], steps={"a": 5, "b": big}), T, H)
    from benchmark_v2 import catalogue_d
    assert "not validatable" in questions.check_gold(catalogue_d.D01, {}, Gold([1], {"graph": 300, "sql": big}),
                                                     200, 10_000)
    assert questions.check_gold(fc.S1, {}, Gold([1, 2], steps={"a": 5, "b": 1000}), T, H) is None


def test_a_step_that_hits_the_validator_cap_is_a_failure_named_by_step(world, generated, tmp_path):
    _, _, db_path = world
    path = tmp_path / "o.sqlite"
    shutil.copy(db_path, path)
    stores = _OracleStores(str(path))
    ex = Executor({s: stores.client(s) for s in (Store.GRAPH, Store.SQL, Store.DOC)}, cap=3)
    r = _by(generated[0], "S1")[0]
    out = validate_record(r, ex)
    assert out.status == "cap_failure" and "b hit the validator cap of 3" in out.reason and out.reason.startswith("step(s) ")
