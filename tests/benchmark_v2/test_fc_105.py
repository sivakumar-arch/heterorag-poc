"""Batch 105: first-contact templates H1, H3, H4 and the off-grid probe template H4P (class F).

The synthetic world has years 2015 to 2020, short answer bodies and equal up-vote counts, so a copy of its oracle gets
answer bodies with a known vocabulary and varied up-vote counts, and the H4 years and ranks are set to the synthetic
range by monkeypatching the module constants. Every gold is re-derived with direct queries on the oracle views.
"""
import json
import re
import shutil
import sqlite3

import pytest

from benchmark_v2 import catalogue_fc as fc, catalogue_n1, pattern_key as pk, pool_counts, questions, union
from benchmark_v2.questions import Ctx
from benchmark_v2.validate_instances import validate_record
from heterorag.federation import Executor, FetchResult, Store
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)

T, H = 10, 40
HH = 1000                          # H3 and H4: the synthetic years and words are larger than 40
WORD = "heteroskedastic"           # in 25 cleaned answer bodies, lower case
CODE_WORD = "codeonlyterm"         # only inside <code>: never eligible
NAME = "Fisherman"                 # capitalised in every body: never eligible


@pytest.fixture(scope="module")
def path(world, tmp_path_factory):
    _, _, db_path = world
    p = tmp_path_factory.mktemp("fc105") / "oracle.sqlite"
    shutil.copy(db_path, p)
    con = sqlite3.connect(p)
    answers = [i for (i,) in con.execute("SELECT id FROM posts WHERE post_type_id = 2 AND body <> '' ORDER BY id")]
    for k, a in enumerate(answers[:60]):
        extra = f" {NAME} <code>{CODE_WORD} {WORD}</code>"
        if k < 25:
            extra += f" <p>{WORD}</p>"
        extra += " " + " ".join(catalogue_n1.EXCLUDED_WORDS)
        con.execute("UPDATE posts SET body = body || ? WHERE id = ?", (extra, a))
    con.execute("UPDATE users SET up_votes = id * 7 + 1 WHERE id > 0")      # distinct: no ties
    con.commit()
    con.close()
    return str(p)


@pytest.fixture(scope="module")
def db(path):
    d = union.open_union(path)
    yield d
    d.close()


@pytest.fixture
def synthetic_years(monkeypatch):
    monkeypatch.setattr(fc, "H4_YEARS", (2015, 2016, 2017, 2018))
    monkeypatch.setattr(fc, "H4_RANKS", (2, 4, 6))
    monkeypatch.setattr(fc, "H4P_YEARS", (2016, 2019))
    monkeypatch.setattr(fc, "H4P_RANKS", (3, 5))


def gen(db, templates, per, h=H):
    return questions.build_records(templates, db, seed=11, per_template=per, t_independent=T, h_cap=h)


# ---------------------------------------------------------------- wording (hand-in v1.2, verbatim)

def test_wording_is_the_hand_in_text():
    assert fc.H1.paraphrases[0] == 'How many questions were asked in {year} by users who hold the badge "{badge}"?'
    assert fc.H3.paraphrases[2] == 'Find authors of answers containing "{word}" who have a reputation below {number}.'
    assert fc.H4.paraphrases[1] == ('List the authors of questions created during {year} who have cast {number} '
                                    'or more up-votes.')
    assert fc.H4P.split == "probe" and fc.H4P not in fc.TEMPLATES and fc.H4P.plan == fc.H4.plan


# ---------------------------------------------------------------- H1

def test_h1_quota_caps_gold_and_restriction(db):
    recs, reps = gen(db, [fc.H1], 6)
    year_n = {y: n for y, n in db.execute(
        "SELECT CAST(substr(creation_date, 1, 4) AS INTEGER), count(*) FROM v_questions GROUP BY 1")}
    assert recs, reps[0].notes
    for r in recs:
        b, y = r["params"]["badge"], r["params"]["year"]
        holders = {u for (u,) in db.execute("SELECT user_id FROM badges WHERE name = ?", (b,))}
        asked = {q for q, u in db.execute("SELECT question_id, user_id FROM v_asked") if u in holders}
        in_year = {q for q, d in db.execute("SELECT id, creation_date FROM v_questions") if int(d[:4]) == y}
        assert r["gold"] == len(asked & in_year) >= 1
        assert r["stratum"] == ("over-cap" if year_n[y] > H else "within-cap")
        assert r["needs_restriction"] == (["c"] if year_n[y] > H else [])
    badges = [r["params"]["badge"] for r in recs]
    assert len(badges) == len(set(badges))
    for stratum, cap in (("over-cap", 2), ("within-cap", 3)):
        ys = [r["params"]["year"] for r in recs if r["stratum"] == stratum]
        assert all(ys.count(y) <= cap for y in ys)


# ---------------------------------------------------------------- H3

def test_h3_vocabulary_rules_and_raw_postings(db):
    c = Ctx(db, 1, T, H)
    vocab = fc.h3_vocab(c)
    assert WORD in vocab
    assert CODE_WORD not in vocab and NAME.lower() not in vocab
    assert not set(catalogue_n1.EXCLUDED_WORDS) & set(vocab)
    # the gold uses the raw body as the store indexes it: the code-only occurrences count too
    raw = {a for a, body in db.execute("SELECT id, body FROM posts WHERE post_type_id = 2 AND body <> ''")
           if WORD in union.tokens(body)}
    assert vocab[WORD] == raw and len(raw) == 60


def test_h3_gold_threshold_and_caps(db):
    recs, reps = gen(db, [fc.H3], 6, HH)
    assert recs, (reps[0].notes, reps[0].rejected)
    rep = dict(db.execute("SELECT id, reputation FROM v_users"))
    author = dict(db.execute("SELECT answer_id, user_id FROM v_answered"))
    for r in recs:
        w, n, k = r["params"]["word"], r["params"]["number"], r["params"]["rank"]
        answers = {a for a, body in db.execute("SELECT id, body FROM posts WHERE post_type_id = 2 AND body <> ''")
                   if w in union.tokens(body)}
        authors = {author[a] for a in answers if a in author}
        assert r["gold"] == sorted(u for u in authors if rep[u] < n)
        assert n == sorted(rep[u] for u in authors)[k - 1] + 1
        assert r["stratum"] == f"rank {k:02d}"
    words = [r["params"]["word"] for r in recs]
    assert len(words) == len(set(words))


def test_h3_rank_collisions_give_one_candidate():
    c = Ctx(None, 1, T, HH)
    c._cache.update({"fc_h3_vocab": {"w": frozenset(range(1, 50))},
                     "fc_answer_author": {a: a for a in range(1, 50)},
                     "reputation": {u: 1 for u in range(1, 50)}})
    out = fc.h3_sample(c, 10)
    assert [(p["rank"], p["number"]) for p in out] == [(5, 2)]
    assert c.notes["H3 rank collision (same NUMBER as a smaller rank)"] == 2


# ---------------------------------------------------------------- H4 and H4P

def test_h4_rotation_variants_numbers_and_gold(db, synthetic_years):
    recs, reps = gen(db, [fc.H4], 12, HH)
    assert len(recs) == 12, (reps[0].notes, reps[0].rejected)
    upv = dict(db.execute("SELECT id, up_votes FROM v_users"))
    for i, r in enumerate(recs):
        y, k, v = r["params"]["year"], r["params"]["rank"], r["params"]["variant"]
        assert v == (fc.H4_RANKS.index(k) + fc.H4_YEARS.index(y)) % 3
        assert r["text"] == r["paraphrases"][v]
        assert pk.wording_variant(r["question_id"], r) == v
        askers = {u for u, q in db.execute("SELECT user_id, question_id FROM v_asked")
                  if q in {x for x, d in db.execute("SELECT id, creation_date FROM v_questions") if int(d[:4]) == y}}
        vals = sorted((upv.get(u) or 0 for u in askers), reverse=True)
        assert r["params"]["number"] == vals[k - 1]
        assert r["gold"] == sorted(u for u in askers if (upv.get(u) or 0) >= r["params"]["number"])
    years = [r["params"]["year"] for r in recs]
    assert years == sorted(years) and all(years.count(y) == 3 for y in fc.H4_YEARS)
    assert {(r["params"]["year"], r["params"]["variant"]) for r in recs} == {(y, v) for y in fc.H4_YEARS for v in range(3)}


def test_an_unavailable_h4_cell_is_not_replaced(db, synthetic_years, monkeypatch):
    monkeypatch.setattr(fc, "H4_RANKS", (2, 4, 10_000))
    recs, reps = gen(db, [fc.H4], 12, HH)
    assert len(recs) == 8                              # four cells of rank 10,000 dropped, nothing in their place
    assert all(r["text"] == r["paraphrases"][r["params"]["variant"]] for r in recs)   # wording follows the cell
    assert sum(v for k, v in reps[0].notes.items() if "unavailable: not replaced" in k) == 4


def test_h4p_is_generated_only_when_named(path, synthetic_years, tmp_path):
    out = tmp_path / "p.jsonl"
    assert questions.main(["--oracle", path, "--only", "H4P,S2", "--per-template", "2", "--out", str(out),
                           "--t-independent", str(T), "--h-cap", str(HH)]) == 0
    got = {json.loads(l)["template"] for l in out.read_text().splitlines()}
    assert got == {"H4P", "S2"}
    names = [t.id for t in questions.all_templates() if t.split == "dev" or (t.split == "probe" and False)]
    assert "H4P" not in names and "H4" in names


# ---------------------------------------------------------------- reference queries on fake stores

class _Stores:
    def __init__(self, path):
        self.path = path

    def executor(self):
        return Executor({s: self._client(s) for s in (Store.GRAPH, Store.SQL, Store.DOC)})

    def _client(self, store):
        outer = self

        class C:
            def fetch(self, step, query, binds, limit):
                db = union.open_union(outer.path)
                try:
                    ids = sorted(self._ids(db, query, binds))
                    return FetchResult(ids[:limit], exhausted=len(ids) <= limit)
                finally:
                    db.close()

            def _ids(self, db, q, binds):
                bound = set((binds or {}).get("a") or ())
                if store is Store.GRAPH:
                    if "ANSWERED" in q:
                        return {u for u, a in db.execute("SELECT user_id, answer_id FROM v_answered") if a in bound}
                    if "WHERE u.id IN $a" in q:
                        return {x for u, x in db.execute("SELECT user_id, question_id FROM v_asked") if u in bound}
                    if "WHERE q.id IN $a" in q:
                        return {u for u, x in db.execute("SELECT user_id, question_id FROM v_asked") if x in bound}
                    raise AssertionError(q)
                if store is Store.SQL:
                    if "FROM badges" in q:
                        name = re.search(r"name = '((?:[^']|'')*)'", q).group(1).replace("''", "'")
                        return {u for (u,) in db.execute("SELECT user_id FROM badges WHERE name = ?", (name,))}
                    if "EXTRACT(YEAR" in q:
                        y = int(re.search(r"creation_date\) = (\d+)", q).group(1))
                        return {i for i, d in db.execute("SELECT id, creation_date FROM v_questions") if int(d[:4]) == y}
                    if "reputation <" in q:
                        n = int(re.search(r"< (\d+)", q).group(1))
                        return {u for u, r in db.execute("SELECT id, reputation FROM v_users") if r < n}
                    if "up_votes >=" in q:
                        n = int(re.search(r">= (\d+)", q).group(1))
                        return {u for u, v in db.execute("SELECT id, up_votes FROM v_users") if (v or 0) >= n}
                    raise AssertionError(q)
                w = json.loads(q)["query"]["bool"]["filter"][1]["match"]["body"]["query"]
                return {a for a, b in db.execute("SELECT id, body FROM posts WHERE post_type_id = 2 AND body <> ''")
                        if w in union.tokens(b)}
        return C()


def test_h1_h3_h4_validate_on_stores_that_agree_with_the_oracle(db, path, synthetic_years):
    recs, _ = gen(db, [fc.H1, fc.H3, fc.H4], 4, HH)
    assert {r["template"] for r in recs} == {"H1", "H3", "H4"}
    ex = _Stores(path).executor()
    for r in recs:
        out = validate_record(r, ex)
        assert out.status == "match", (r["question_id"], out)


# ---------------------------------------------------------------- supply report and word list

def test_pool_counts_for_h1_h3_h4(db, synthetic_years):
    for t in (fc.H1, fc.H3, fc.H4):
        r = pool_counts.count(t, Ctx(db, 3, T, HH), set(), T, HH)
        assert r["pool"] > 0 and "step_sizes" in r, t.id
    r = pool_counts.count(fc.H3, Ctx(db, 3, T, HH), set(), T, HH)
    assert "artificial_threshold" in r


def test_h3_word_list(path, tmp_path):
    out = tmp_path / "h3.tsv"
    assert fc.main(["--oracle", path, "--words", "H3", "--out", str(out)]) == 0
    rows = out.read_text().splitlines()
    assert rows[0] == "word\tanswers" and any(r.startswith(WORD + "\t") for r in rows)


def test_h4_ranks_with_the_same_number_drop_the_cell():
    c = Ctx(None, 1, T, HH)
    c._cache.update({"fc_questions_of_year": {2015: {1, 2, 3, 4}}, "asker_of": {1: 10, 2: 11, 3: 12, 4: 13},
                     "fc_up_votes": {10: 9, 11: 9, 12: 5, 13: 1}})
    out = fc._h4_cells(c, (2015,), (1, 2, 3), "H4")
    assert [(p["rank"], p["number"]) for p in out] == [(3, 5)]      # ranks 1 and 2 both give 9: dropped
    assert sum(c.notes.values()) == 2


def test_h1_takes_one_question_per_badge_across_strata():
    from benchmark_v2.questions import Gold
    rows = {"over-cap": [({"stratum": "over-cap", "badge": "b", "year": 2015}, Gold(3)),
                         ({"stratum": "over-cap", "badge": "c", "year": 2016}, Gold(3))],
            "within-cap": [({"stratum": "within-cap", "badge": "b", "year": 2011}, Gold(3))]}
    chosen = questions.select(fc.H1, rows, 3)
    assert sorted(p["badge"] for p, _ in chosen) == ["b", "c"]
