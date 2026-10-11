"""
benchmark_v2/catalogue_sp.py
============================
Templates for classes S (single store) and P (independent sub-queries, each small).

Every gold function is written against the virtual union views (``benchmark_v2.union``).
Samplers draw parameters in strata taken from the oracle profile and return more candidates
than needed; the generator validates them and takes the requested number round-robin.

Dev / test split (decision D3, spec section 8): S03, S07 and P03 are test templates.
They are assigned here, before any system has been run.
"""

from __future__ import annotations

from . import union
from .questions import Ctx, Gold, MAX_SET, Step, Template, T_INDEPENDENT, pick


def _bucket(n: int, bands: list[tuple[str, int, int]]) -> str | None:
    for name, lo, hi in bands:
        if lo <= n <= hi:
            return name
    return None


# ============================================================================
# Class S: a single store suffices
# ============================================================================

# ---- S01 (sql) reputation of a user ----------------------------------------

def s01_sample(c: Ctx, n: int):
    out = []
    for name, cond in (("rep=1", "reputation = 1"), ("rep101-999", "reputation BETWEEN 101 AND 999"),
                       ("rep>=1000", "reputation >= 1000")):
        ids = c.col(f"SELECT id FROM v_users WHERE {cond} ORDER BY id")
        out += [{"stratum": name, "user": u} for u in pick(ids, n, c.rng)]
    return out


def s01_gold(c: Ctx, p: dict) -> Gold:
    return Gold(c.one("SELECT reputation FROM v_users WHERE id = ?", p["user"]))


S01 = Template(
    "S01", "S", "dev", "scalar", "number",
    ("What is the reputation of the user with ID {user}?",
     "How much reputation does user {user} have?",
     "Give the reputation score of user ID {user}."),
    (Step("a", "retrieve", "sql", question="user reputation", output="number", kind="scalar"),),
    s01_sample, s01_gold)


# ---- S02 (sql) number of badges of a user ----------------------------------

def s02_sample(c: Ctx, n: int):
    rows = c.rows("SELECT user_id, count(*) FROM badges WHERE user_id IN (SELECT id FROM v_users) "
                  "GROUP BY user_id")
    bands = [("1-2 badges", 1, 2), ("3-10 badges", 3, 10), ("11+ badges", 11, 10**9)]
    groups: dict[str, list[int]] = {}
    for u, cnt in rows:
        b = _bucket(cnt, bands)
        if b:
            groups.setdefault(b, []).append(u)
    out = []
    for name, ids in sorted(groups.items()):
        out += [{"stratum": name, "user": u} for u in pick(sorted(ids), n, c.rng)]
    return out


def s02_gold(c: Ctx, p: dict) -> Gold:
    return Gold(c.one("SELECT count(*) FROM badges WHERE user_id = ?", p["user"]))


S02 = Template(
    "S02", "S", "dev", "scalar", "number",
    ("How many badges has the user with ID {user} earned?",
     "What is the total number of badges awarded to user {user}?",
     "Count the badges held by user ID {user}."),
    (Step("a", "retrieve", "sql", question="count badges of user", output="number", kind="scalar"),),
    s02_sample, s02_gold)


# ---- S03 (sql, test) top-k questions by views in a year --------------------

def _top_views(c: Ctx, year: str, k: int):
    return c.rows("SELECT id, coalesce(view_count, 0) FROM v_questions "
                  "WHERE substr(creation_date, 1, 4) = ? ORDER BY 2 DESC, id ASC LIMIT ?", year, k + 1)


def s03_sample(c: Ctx, n: int):
    years = [y for y, cnt in c.rows("SELECT substr(creation_date, 1, 4), count(*) FROM v_questions "
                                    "GROUP BY 1") if cnt >= 50]
    out = []
    for _ in range(n * 3):
        k, y = c.rng.choice((5, 10)), c.rng.choice(years)
        out.append({"stratum": f"k={k}", "year": y, "k": k})
    return out


def s03_gold(c: Ctx, p: dict) -> Gold:
    rows = _top_views(c, p["year"], p["k"])
    k = p["k"]
    if len(rows) <= k or rows[k - 1][1] == rows[k][1]:
        return Gold(None)            # tie at the boundary: rejected by the generator
    return Gold([r[0] for r in rows[:k]])


S03 = Template(
    "S03", "S", "test", "ranked", "question",
    ("Which {k} questions created in {year} have the highest view count?",
     "List the top {k} most viewed questions that were asked in {year}.",
     "Among questions posted in {year}, which {k} have been viewed the most?"),
    (Step("a", "retrieve", "sql", question="top-k by view_count within year", output="question", kind="ranked"),),
    s03_sample, s03_gold, k_param="k")


# ---- S05 (graph) answers of a question --------------------------------------

def s05_sample(c: Ctx, n: int):
    rows = c.rows("SELECT question_id, count(*) FROM v_answers_of GROUP BY question_id")
    bands = [("1-2 answers", 1, 2), ("3-5 answers", 3, 5), ("6+ answers", 6, 50)]
    groups: dict[str, list[int]] = {}
    for q, cnt in rows:
        b = _bucket(cnt, bands)
        if b:
            groups.setdefault(b, []).append(q)
    out = []
    for name, ids in sorted(groups.items()):
        out += [{"stratum": name, "question": q} for q in pick(sorted(ids), n, c.rng)]
    return out


def s05_gold(c: Ctx, p: dict) -> Gold:
    return Gold(c.col("SELECT answer_id FROM v_answers_of WHERE question_id = ? ORDER BY answer_id",
                      p["question"]))


S05 = Template(
    "S05", "S", "dev", "ids", "answer",
    ("Which answers were posted for the question with ID {question}?",
     "List the IDs of all answers to question {question}.",
     "What are the answers to question ID {question}?"),
    (Step("a", "retrieve", "graph", question="answers of question", output="answer"),),
    s05_sample, s05_gold)


# ---- S06 (graph) duplicates of a question -----------------------------------

def s06_sample(c: Ctx, n: int):
    rows = c.rows("SELECT original_id, count(*) FROM v_duplicate_of GROUP BY original_id")
    bands = [("1 duplicate", 1, 1), ("2-4 duplicates", 2, 4), ("5+ duplicates", 5, 50)]
    groups: dict[str, list[int]] = {}
    for q, cnt in rows:
        b = _bucket(cnt, bands)
        if b:
            groups.setdefault(b, []).append(q)
    out = []
    for name, ids in sorted(groups.items()):
        out += [{"stratum": name, "question": q} for q in pick(sorted(ids), n, c.rng)]
    return out


def s06_gold(c: Ctx, p: dict) -> Gold:
    return Gold(c.col("SELECT question_id FROM v_duplicate_of WHERE original_id = ? "
                      "ORDER BY question_id", p["question"]))


S06 = Template(
    "S06", "S", "dev", "ids", "question",
    ("Which questions are marked as duplicates of the question with ID {question}?",
     "List the IDs of questions that were closed as duplicates of question {question}.",
     "What questions duplicate question ID {question}?"),
    (Step("a", "retrieve", "graph", question="duplicate-of in-neighbours", output="question"),),
    s06_sample, s06_gold)


# ---- S07 (graph, test) most co-occurring tags --------------------------------

def _cooc_top(c: Ctx, tag: str, k: int):
    union.ensure_cooccurrence(c.db)
    return c.rows("""SELECT other, weight FROM (
                       SELECT tag2 AS other, weight FROM v_cooc WHERE tag1 = ?
                       UNION ALL SELECT tag1, weight FROM v_cooc WHERE tag2 = ?)
                     ORDER BY weight DESC, other ASC LIMIT ?""", tag, tag, k + 1)


def s07_sample(c: Ctx, n: int):
    sizes = c.tag_sizes()
    bands = [("tag 1000+", 1000, 10**9), ("tag 200-999", 200, 999), ("tag 50-199", 50, 199)]
    groups: dict[str, list[str]] = {}
    for tag, cnt in sizes.items():
        b = _bucket(cnt, bands)
        if b:
            groups.setdefault(b, []).append(tag)
    out = []
    for name, tags in sorted(groups.items()):
        for tag in pick(sorted(tags), n * 2, c.rng):
            out.append({"stratum": name, "tag": tag, "k": c.rng.choice((5, 10))})
    return out


def s07_gold(c: Ctx, p: dict) -> Gold:
    rows, k = _cooc_top(c, p["tag"], p["k"]), p["k"]
    if len(rows) <= k or rows[k - 1][1] == rows[k][1]:
        return Gold(None)
    return Gold([r[0] for r in rows[:k]])


S07 = Template(
    "S07", "S", "test", "ranked", "tag",
    ("Which {k} tags appear together most often with the tag \"{tag}\"?",
     "What are the top {k} tags that co-occur with \"{tag}\", most frequent first?",
     "List the {k} tags most frequently used together with \"{tag}\"."),
    (Step("a", "retrieve", "graph", question="top-k co-occurrence weight", output="tag", kind="ranked"),),
    s07_sample, s07_gold, k_param="k")


# ---- S08 (graph) tags of a question ------------------------------------------

def s08_sample(c: Ctx, n: int):
    rows = c.rows("SELECT question_id, count(*) FROM v_tagged GROUP BY question_id")
    bands = [("1-2 tags", 1, 2), ("3 tags", 3, 3), ("4-5 tags", 4, 5)]
    groups: dict[str, list[int]] = {}
    for q, cnt in rows:
        b = _bucket(cnt, bands)
        if b:
            groups.setdefault(b, []).append(q)
    out = []
    for name, ids in sorted(groups.items()):
        out += [{"stratum": name, "question": q} for q in pick(sorted(ids), n, c.rng)]
    return out


def s08_gold(c: Ctx, p: dict) -> Gold:
    return Gold(c.tags_of(p["question"]))


S08 = Template(
    "S08", "S", "dev", "ids", "tag",
    ("Which tags are attached to the question with ID {question}?",
     "List the tags of question {question}.",
     "What tags does question ID {question} carry?"),
    (Step("a", "retrieve", "graph", question="tags of question", output="tag"),),
    s08_sample, s08_gold)


# ---- S09 (doc) questions whose title contains a word ---------------------------

def _word_strata(c: Ctx, bands: list[tuple[str, int, int]], n: int) -> dict[str, list[str]]:
    post = c.title_postings()
    groups: dict[str, list[str]] = {}
    for w, ids in post.items():
        b = _bucket(len(ids), bands)
        if b:
            groups.setdefault(b, []).append(w)
    return {b: pick(sorted(ws), n, c.rng) for b, ws in groups.items()}


def s09_sample(c: Ctx, n: int):
    bands = [("df 2-5", 2, 5), ("df 6-20", 6, 20), ("df 21-50", 21, 50)]
    return [{"stratum": b, "word": w} for b, ws in sorted(_word_strata(c, bands, n).items()) for w in ws]


def s09_gold(c: Ctx, p: dict) -> Gold:
    return Gold(list(c.title_postings()[p["word"]]))


S09 = Template(
    "S09", "S", "dev", "ids", "question",
    ("Which questions have the word \"{word}\" in their title?",
     "List the IDs of questions whose title contains \"{word}\".",
     "Find all questions with \"{word}\" in the title."),
    (Step("a", "retrieve", "doc", "filter", "title contains word", output="question"),),
    s09_sample, s09_gold)


# ============================================================================
# Class P: independent sub-queries on several stores, each small (<= T)
# ============================================================================

# ---- P01 (graph and doc) tagged T and title word W ----------------------------

P01_BANDS = ((1, 1, "gold 1"), (2, 5, "gold 2-5"), (6, 20, "gold 6-20"), (21, MAX_SET, "gold 21-50"))


def p01_sample(c: Ctx, n: int):
    """Tag-first sampling, stratified by the actual size of the answer.

    For a tag with at most T questions, count for every title word (with at most T questions
    overall) how many of the tag's questions contain it. That count is the answer size, so the
    strata are exact and a single-item answer is only one stratum among four."""
    from collections import Counter
    post, members = c.title_postings(), c.tag_members()
    qwords = c.title_words()
    tags = sorted(t for t, qs in members.items() if 8 <= len(qs) <= T_INDEPENDENT)
    c.rng.shuffle(tags)
    pools: dict[str, list[dict]] = {name: [] for _, _, name in P01_BANDS}
    want = n * 4
    for tag in tags[:1500]:
        counts = Counter(w for qid in members[tag] for w in qwords.get(qid, ())
                         if len(post[w]) <= T_INDEPENDENT)
        for lo, hi, name in P01_BANDS:
            if len(pools[name]) >= want:
                continue
            words = sorted(w for w, k in counts.items() if lo <= k <= hi)
            if words:
                pools[name].append({"stratum": name, "tag": tag, "word": c.rng.choice(words)})
        if all(len(v) >= want for v in pools.values()):
            break
    return [x for v in pools.values() for x in v]


def p01_gold(c: Ctx, p: dict) -> Gold:
    g, d = set(c.tagged(p["tag"])), set(c.title_postings()[p["word"]])
    return Gold(sorted(g & d), {"graph": len(g), "doc": len(d)})


P01 = Template(
    "P01", "P", "dev", "ids", "question",
    ("Which questions are tagged \"{tag}\" and also have the word \"{word}\" in their title?",
     "List the questions that carry the tag \"{tag}\" and whose title contains \"{word}\".",
     "Find questions with tag \"{tag}\" and \"{word}\" in the title."),
    (Step("a", "retrieve", "graph", question="questions tagged", output="question"),
     Step("b", "retrieve", "doc", "filter", "title contains word", output="question"),
     Step("c", "intersect", inputs=("a", "b"), output="question")),
    p01_sample, p01_gold)


# ---- P02 (sql and graph) answerers of a question with high reputation ----------

def _reputation_tiers(c: Ctx):
    reps = c.cached("rep_desc", lambda: c.col("SELECT reputation FROM v_users ORDER BY reputation DESC"))
    tiers = []
    for size in (60, 120, 200):
        if len(reps) > size:
            r = reps[size - 1]                      # count(rep >= r) can exceed size on ties
            cnt = c.one("SELECT count(*) FROM v_users WHERE reputation >= ?", r)
            if cnt <= T_INDEPENDENT:
                tiers.append((f"{size} users", r))
    return tiers


P02_BANDS = ((1, 1, "gold 1"), (2, 3, "gold 2-3"), (4, MAX_SET, "gold 4+"))


def p02_sample(c: Ctx, n: int):
    """Stratified by the size of the answer: a question has few answerers, so sizes above one
    are rarer and are drawn first from each reputation tier."""
    from collections import Counter
    pools: dict[str, list[dict]] = {name: [] for _, _, name in P02_BANDS}
    for _tier, r in _reputation_tiers(c):
        rows = c.rows("""SELECT DISTINCT ao.question_id, an.user_id FROM v_answered an
                         JOIN v_answers_of ao ON ao.answer_id = an.answer_id
                         JOIN v_users u ON u.id = an.user_id
                         WHERE u.reputation >= ?""", r)
        size = Counter(q for q, _ in rows)
        for lo, hi, name in P02_BANDS:
            qs = sorted(q for q, k in size.items() if lo <= k <= hi)
            pools[name] += [{"stratum": name, "min_reputation": r, "question": q}
                            for q in pick(qs, n * 2, c.rng)]
    return [x for v in pools.values() for x in v]


def p02_gold(c: Ctx, p: dict) -> Gold:
    s = c.one("SELECT count(*) FROM v_users WHERE reputation >= ?", p["min_reputation"])
    ans = c.col("""SELECT DISTINCT an.user_id FROM v_answers_of ao
                   JOIN v_answered an ON an.answer_id = ao.answer_id
                   WHERE ao.question_id = ? ORDER BY an.user_id""", p["question"])
    hi = c.col("""SELECT DISTINCT an.user_id FROM v_answers_of ao
                  JOIN v_answered an ON an.answer_id = ao.answer_id
                  JOIN v_users u ON u.id = an.user_id
                  WHERE ao.question_id = ? AND u.reputation >= ? ORDER BY an.user_id""",
               p["question"], p["min_reputation"])
    return Gold(hi, {"sql": s, "graph": len(ans)})


P02 = Template(
    "P02", "P", "dev", "ids", "user",
    ("Which users with a reputation of at least {min_reputation} answered the question with ID {question}?",
     "Among the users who answered question {question}, which have reputation {min_reputation} or more?",
     "List the answerers of question ID {question} whose reputation is {min_reputation} or higher."),
    (Step("a", "retrieve", "sql", question="users with reputation >= R", output="user"),
     Step("b", "retrieve", "graph", question="answerers of question", output="user"),
     Step("c", "intersect", inputs=("a", "b"), output="user")),
    p02_sample, p02_gold)


# ---- P03 (sql and doc, test) high-score questions with a title word -----------

def p03_sample(c: Ctx, n: int):
    post = c.title_postings()
    scores = c.cached("q_scores", lambda: c.col("SELECT coalesce(score, 0) FROM v_questions ORDER BY 1 DESC"))
    if len(scores) <= T_INDEPENDENT:
        return []
    s_lo = scores[T_INDEPENDENT] + 1                 # smallest threshold with <= T questions
    top = c.rows("SELECT q.id, coalesce(q.score, 0), d.title FROM v_question_docs d "
                 "JOIN v_questions q ON q.id = d.post_id WHERE coalesce(q.score, 0) >= ?", s_lo)
    out = []
    for _ in range(n * 8):
        if not top:
            break
        q0, sc0, title = c.rng.choice(top)
        words = [w for w in set(union.tokens(title)) if w in post and 2 <= len(post[w]) <= 50]
        if not words:
            continue
        w = c.rng.choice(sorted(words))
        s = c.rng.randint(s_lo, max(s_lo, sc0))
        df = len(post[w])
        out.append({"stratum": "df<=10" if df <= 10 else "df11-50", "word": w, "min_score": s})
    return out


def p03_gold(c: Ctx, p: dict) -> Gold:
    d = c.title_postings()[p["word"]]
    s_n = c.one("SELECT count(*) FROM v_questions WHERE coalesce(score, 0) >= ?", p["min_score"])
    hit = c.col(f"SELECT id FROM v_questions WHERE coalesce(score, 0) >= ? AND id IN ({','.join('?' * len(d))}) "
                "ORDER BY id", p["min_score"], *d)
    return Gold(hit, {"sql": s_n, "doc": len(d)})


P03 = Template(
    "P03", "P", "test", "ids", "question",
    ("Which questions with the word \"{word}\" in their title have a score of at least {min_score}?",
     "List questions whose title contains \"{word}\" and whose score is {min_score} or higher.",
     "Find the questions titled with \"{word}\" that scored at least {min_score}."),
    (Step("a", "retrieve", "sql", question="questions with score >= s", output="question"),
     Step("b", "retrieve", "doc", "filter", "title contains word", output="question"),
     Step("c", "intersect", inputs=("a", "b"), output="question")),
    p03_sample, p03_gold)


# ---- P04 (doc and graph) title word but not tagged T --------------------------

def p04_sample(c: Ctx, n: int):
    post, sizes = c.title_postings(), c.tag_sizes()
    words = sorted(w for w, ids in post.items() if 6 <= len(ids) <= 50)
    out = []
    for _ in range(n * 8):
        w = c.rng.choice(words)
        ids = post[w]
        counts: dict[str, int] = {}
        for q in ids:
            for t in c.tags_of(q):
                counts[t] = counts.get(t, 0) + 1
        cands = [t for t, k in counts.items() if 1 <= k < len(ids) and sizes.get(t, 0) <= 150]
        if not cands:
            continue
        t = c.rng.choice(sorted(cands))
        frac = counts[t] / len(ids)
        out.append({"stratum": "overlap<1/3" if frac < 1 / 3 else ("overlap<2/3" if frac < 2 / 3 else "overlap>=2/3"),
                    "word": w, "tag": t})
    return out


def p04_gold(c: Ctx, p: dict) -> Gold:
    d, g = set(c.title_postings()[p["word"]]), set(c.tagged(p["tag"]))
    return Gold(sorted(d - g), {"doc": len(d), "graph": len(g)})


P04 = Template(
    "P04", "P", "dev", "ids", "question",
    ("Which questions with the word \"{word}\" in their title are not tagged \"{tag}\"?",
     "List questions whose title contains \"{word}\" but that do not carry the tag \"{tag}\".",
     "Among questions titled with \"{word}\", which lack the tag \"{tag}\"?"),
    (Step("a", "retrieve", "doc", "filter", "title contains word", output="question"),
     Step("b", "retrieve", "graph", question="questions tagged", output="question"),
     Step("c", "subtract", inputs=("a", "b"), output="question")),
    p04_sample, p04_gold)


TEMPLATES: list[Template] = [S01, S02, S03, S05, S06, S07, S08, S09, P01, P02, P03, P04]
