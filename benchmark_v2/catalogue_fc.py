"""
benchmark_v2/catalogue_fc.py
============================
First-contact templates (pre-registration section 3; build item 3; plan approved by Siva on 2026-11-26 with
changes, session log of 2026-11-26 sections 29 to 31). Class F: the registered cap rules checked step by step,
no lower threshold T (``questions._check_f``).

Batch 104: S1, S2 and H2 (104b: fixes after the first probe). Batch 105: H1, H3, H4 and the off-grid H4 probe
template H4P.

S1  questions marked as duplicates of a question tagged TAG with a score of at least NUMBER (authors' wording,
    item 6 of the decisions). Gold plan in the merged form (A2): one graph step for the duplicates of the tag's
    questions, one SQL step for the score, intersect. NUMBER is the score of the k-th best candidate, k in
    ``S1_RANKS``; NUMBER below 1 is not used (Batch 104b: "at least 0" restricts almost nothing, and its SQL step
    exceeds the validator's cap); two ranks that give the same NUMBER for one tag give one candidate (the smaller
    rank). Stratum = rank; quota 8, 8, 7, 7 for ranks 2, 5, 10, 20 (of 30), and a shortfall of rank 20 is filled
    from rank 10 and recorded (decision of 2026-11-26, second round).
S2  users who asked a question tagged TAG_A and have never asked one tagged TAG_B (authors' wording): both asker
    sets within H, the sets overlap, 1 to 50 users remain. TAG_B among the 40 largest tags with at most H askers,
    as in the Batch 97 pilot.
H2  questions tagged TAG with the word WORD in the title or an accepted answer (Seema Aarella's wording, v1.2,
    verbatim). Gold plan in the conditions-kept-together form (A3): (tagged and word) union (tagged with an
    accepted answer). Selection rule (v1.5, restored): the word contributes at least 3 questions outside the
    accepted-answer part and the accepted-answer part at least 3 outside the word part. The exclusive-reading gold
    is stored (``gold_exclusive``), and the gold function checks that the two forms of the plan give one set.
    Batch 104b: the word may not be a component of the tag name (the tag split on hyphens) nor clash with one under
    the title guard of N1 (so "regressions" is excluded for a tag containing "regression"); ``H2_EXCLUDED`` holds
    the one misspelling found in the seeded sample of 200 (read on 2026-11-26).

H1  questions asked in YEAR by users who hold the badge BADGE, counted (Seema Aarella's wording, v1.2, verbatim).
    Gold plan: SQL badge holders; graph questions they asked (bound); SQL questions of the year; intersect; count.
    Badges whose holders number 1 to H and whose holders' questions number 1 to H. Years: over-cap are the years
    with more than H questions, within-cap the others; 2009 is excluded as degenerate (decision of 2026-11-16).
    Quota 20 over-cap and 10 within-cap (of 30); at most two questions per over-cap year and three per within-cap
    year; one per badge. The count is at least 1. Over-cap years are exact only through the restriction of the
    year step.
H3  users with a reputation under NUMBER who wrote an answer containing WORD (Seema Aarella's wording, v1.2,
    verbatim). Gold plan: document answers whose body contains the word; graph their authors (bound); SQL users
    with reputation under NUMBER; intersect. The word is chosen from cleaned answer bodies (``clean_body``: markup,
    code, TeX removed), in at least ``H3_MIN_BODIES`` answers, written in lower case in at least 90% of its
    occurrences there, not in the N1 exclusion list nor ``H3_EXCLUDED``; the gold set of answers is computed on the
    raw body, as the store indexes it (raw HTML through the standard tokenizer). NUMBER is one more than the k-th
    lowest reputation among the authors (k in ``H3_RANKS``); two ranks that give the same NUMBER for one word give
    one candidate (the smaller rank). Stratum = rank; one question per word.
H4  users who asked a question in YEAR and have cast at least NUMBER up-votes (Seema Aarella's wording, v1.2,
    verbatim). Gold plan: SQL questions of the year; graph their askers (bound); SQL users with at least NUMBER
    up-votes; intersect. A fixed set of 12 (decided 2026-11-16): years 2010, 2011, 2012, 2024; NUMBER the up_votes
    of the 5th, 20th and 40th highest asker of the year (askers as the graph has them, missing up_votes as 0);
    the wording variant rotated as variant = (rank index + year index) mod 3 and stored in the parameters. A cell
    with no valid NUMBER is not replaced (failure rule of 2026-11-26).
H4P the off-grid probe of H4 (decision 7 of 2026-11-26): ranks 10 and 30 in 2011 and 2024, split "probe", so
    it is generated only when named with --only and is never part of the fresh set.

Draw caps (``group_caps``): one question per tag (S1, H2), one per word (H2, H3), one per TAG_A and at most two per
TAG_B (S2), one per badge (H1) with the per-year caps above.

The word rule of H2 is the D05 rule (titles, alphabetic, 5 letters or more, not a stop word). The document store's
analyzer is the standard tokenizer with English stop words and no stemming (session log of 2026-11-26, section
28); its stop words of 5 letters or more are exactly ``questions.STOPWORDS``. ``H2_EXCLUDED`` is fixed once from
the reading of the eligible list (``python -m benchmark_v2.catalogue_fc --oracle ... --words H2 --out ...``).

Like the rest of the generator, this module does not import the system under test.
"""

from __future__ import annotations

import argparse
import random
from bisect import bisect_left

from . import union
from collections import Counter

from .catalogue_d import _askers_by_tag, _badge_holders, _questions_by_user, _reputation
from .catalogue_n1 import _TOKEN_ANY_CASE, BODY_MIN_LOWER, EXCLUDED_WORDS, TitleGuard, _indexable, clean_body
from .questions import MAX_SET, Bind, Ctx, Gold, Step, Template

S1_RANKS = (2, 5, 10, 20)
S1_MIN_NUMBER = 1            # Batch 104b
S2_B_CANDIDATES = 40
S2_PER_TAG_A = 2             # candidates offered per TAG_A (the draw cap keeps one)
H2_MIN_ARM = 3               # selection rule: items that each arm contributes alone
WORD_SAMPLE = 200            # words printed for reading when the eligible list is long
WORD_SAMPLE_SEED = 20261126

# Fixed once from the reading of the seeded sample of 200 of the 1,613 eligible words (2026-11-26). The sample's
# misspelling rate (1 in 200, about 0.5%) gives about 0.15 expected misspelled words among 30 H2 questions; the
# outside wording check on the generated questions is the backstop (a flagged word is replaced by a seeded redraw
# under the issue B rule).
H2_EXCLUDED: dict[str, str] = {"multicolinearity": "misspelling"}


# ---------------------------------------------------------------- shared structures

def _duplicates_of(c: Ctx) -> dict[int, set[int]]:
    """original question -> questions marked as duplicates of it (the graph's DUPLICATE_OF edges)."""
    def build():
        out: dict[int, set[int]] = {}
        for q, orig in c.db.execute("SELECT question_id, original_id FROM v_duplicate_of"):
            out.setdefault(orig, set()).add(q)
        return out
    return c.cached("fc_duplicates_of", build)


def _question_score(c: Ctx) -> dict[int, int]:
    """question -> score, missing scores as 0 (the SQL reference query uses coalesce(score, 0))."""
    return c.cached("fc_question_score", lambda: {
        q: (s or 0) for q, s in c.db.execute("SELECT id, score FROM v_questions")})


def _scores_sorted(c: Ctx) -> list[int]:
    return c.cached("fc_scores_sorted", lambda: sorted(_question_score(c).values()))


def _accepted(c: Ctx) -> frozenset[int]:
    """questions with an accepted answer (the graph's ACCEPTED edges)."""
    return c.cached("fc_accepted", lambda: frozenset(q for (q,) in c.db.execute(
        "SELECT DISTINCT question_id FROM v_accepted")))


# ---------------------------------------------------------------- S1

def _s1_candidates(c: Ctx, tag: str) -> set[int]:
    dups = _duplicates_of(c)
    out: set[int] = set()
    for q in c.tag_members().get(tag, ()):
        out |= dups.get(q, set())
    return out


def s1_sample(c: Ctx, n: int) -> list[dict]:
    members, score = c.tag_members(), _question_score(c)
    tags = sorted(t for t, qs in members.items() if len(qs) <= c.h_cap)
    c.rng.shuffle(tags)
    out: list[dict] = []
    for tag in tags:
        cand = _s1_candidates(c, tag)
        if not cand:
            continue
        c.notes["S1 tags with a candidate"] += 1
        scores = sorted((score.get(q, 0) for q in cand), reverse=True)
        used: set[int] = set()
        available = sum(k <= len(scores) for k in S1_RANKS)
        collided = False
        for k in S1_RANKS:
            if k > len(scores):
                c.notes[f"S1 rank {k:02d} unavailable"] += 1
                continue
            number = scores[k - 1]
            if number < S1_MIN_NUMBER:
                c.notes[f"S1 threshold below {S1_MIN_NUMBER} not used"] += 1
                continue
            if number in used:
                c.notes["S1 rank collision (same NUMBER as a smaller rank)"] += 1
                collided = True
                continue
            used.add(number)
            out.append({"stratum": f"rank {k:02d}", "tag": tag, "number": number, "rank": k})
        if available >= 2:
            c.notes["S1 tags with two or more ranks"] += 1
            c.notes["S1 tags with a collision"] += collided
    return out


def s1_gold(c: Ctx, p: dict) -> Gold:
    cand, score = _s1_candidates(c, p["tag"]), _question_score(c)
    scores = _scores_sorted(c)
    n_b = len(scores) - bisect_left(scores, p["number"])
    hit = sorted(q for q in cand if score.get(q, 0) >= p["number"])
    return Gold(hit, {"graph": len(cand), "sql": n_b}, steps={"a": len(cand), "b": n_b})


def s1_steps(c: Ctx, p: dict) -> dict[str, set]:
    return {"a": _s1_candidates(c, p["tag"]),
            "b": {q for q, s in _question_score(c).items() if s >= p["number"]}}


S1 = Template(
    "S1", "F", "dev", "ids", "question",
    ("Which questions that are marked as duplicates of a question tagged \"{tag}\" have a score of at least {number}?",
     "Among the questions marked as duplicates of a question tagged \"{tag}\", list those with a score of {number} or more.",
     "Find questions with a score of at least {number} that are marked as duplicates of a question carrying the tag \"{tag}\"."),
    (Step("a", "retrieve", "graph", question="questions marked as duplicates of a question tagged TAG", output="question"),
     Step("b", "retrieve", "sql", question="questions with a score of at least NUMBER", output="question"),
     Step("c", "intersect", inputs=("a", "b"), output="question")),
    s1_sample, s1_gold, step_sets=s1_steps, group_caps=(("tag", 1),),
    quota=(("rank 02", 8), ("rank 05", 8), ("rank 10", 7), ("rank 20", 7)), fallback=(("rank 20", "rank 10"),))


# ---------------------------------------------------------------- S2

def s2_sample(c: Ctx, n: int) -> list[dict]:
    askers = _askers_by_tag(c)
    usable = sorted(t for t, us in askers.items() if 3 <= len(us) <= c.h_cap)
    big = sorted(usable, key=lambda t: (-len(askers[t]), t))[:S2_B_CANDIDATES]
    tags_a = list(usable)
    c.rng.shuffle(tags_a)
    out: list[dict] = []
    for a in tags_a:
        bs = list(big)
        c.rng.shuffle(bs)
        found = 0
        for b in bs:
            if b == a:
                continue
            sa, sb = askers[a], askers[b]
            if sa & sb and 1 <= len(sa - sb) <= MAX_SET:
                out.append({"stratum": "all", "tag_a": a, "tag_b": b})
                found += 1
                if found >= S2_PER_TAG_A:
                    break
    return out


def s2_gold(c: Ctx, p: dict) -> Gold:
    askers = _askers_by_tag(c)
    sa, sb = askers.get(p["tag_a"], set()), askers.get(p["tag_b"], set())
    return Gold(sorted(sa - sb), {}, steps={"a": len(sa), "b": len(sb)})


def s2_steps(c: Ctx, p: dict) -> dict[str, set]:
    askers = _askers_by_tag(c)
    return {"a": set(askers.get(p["tag_a"], set())), "b": set(askers.get(p["tag_b"], set()))}


S2 = Template(
    "S2", "F", "dev", "ids", "user",
    ("Which users who asked a question tagged \"{tag_a}\" have not asked a question tagged \"{tag_b}\"?",
     "List the users who have asked a question with the tag \"{tag_a}\" but never a question with the tag \"{tag_b}\".",
     "Find askers of questions tagged \"{tag_a}\" who have never asked a question tagged \"{tag_b}\"."),
    (Step("a", "retrieve", "graph", question="users who asked a question tagged TAG_A", output="user"),
     Step("b", "retrieve", "graph", question="users who asked a question tagged TAG_B", output="user"),
     Step("c", "subtract", inputs=("a", "b"), output="user")),
    s2_sample, s2_gold, step_sets=s2_steps, group_caps=(("tag_a", 1), ("tag_b", 2)))


# ---------------------------------------------------------------- H2

def h2_words_for_tag(c: Ctx, tag: str) -> list[str]:
    """Words of the tag's titles that satisfy the H2 selection rule, sorted."""
    qs = set(c.tag_members().get(tag, ()))
    acc = qs & _accepted(c)
    if not (H2_MIN_ARM <= len(acc) <= MAX_SET - H2_MIN_ARM):
        return []
    tw = c.title_words()
    parts = [x for x in tag.split("-") if x]
    guard = TitleGuard(parts)
    with_word: dict[str, set[int]] = {}
    for q in qs:
        for w in tw.get(q, ()):
            with_word.setdefault(w, set()).add(q)
    return sorted(w for w, qw in with_word.items()
                  if len(qw - acc) >= H2_MIN_ARM and len(acc - qw) >= H2_MIN_ARM and len(qw | acc) <= MAX_SET
                  and w not in H2_EXCLUDED and w not in parts and not guard.clashes(w))


def h2_sample(c: Ctx, n: int) -> list[dict]:
    members = c.tag_members()
    tags = sorted(t for t, qs in members.items() if 10 <= len(qs) <= c.h_cap)
    c.rng.shuffle(tags)
    out: list[dict] = []
    for tag in tags:
        words = h2_words_for_tag(c, tag)
        if not words:
            continue
        out.append({"stratum": "all", "tag": tag, "word": c.rng.choice(words)})
    return out


def h2_gold(c: Ctx, p: dict) -> Gold:
    qs = set(c.tagged(p["tag"]))
    word = set(c.title_postings().get(p["word"], ()))
    acc_all = _accepted(c)
    natural = qs & (word | acc_all)                      # the hand-in's logic: tag AND (word OR accepted)
    merged = (qs & word) | (qs & acc_all)                # the gold plan (A3)
    exclusive = (qs & word) ^ (qs & acc_all)             # the exclusive reading of "either ... or"
    return Gold(sorted(merged), {}, steps={"a": len(qs), "b": len(word), "c": len(qs & acc_all)},
                extra={"gold_exclusive": sorted(exclusive), "forms_agree": natural == merged})


def h2_steps(c: Ctx, p: dict) -> dict[str, set]:
    qs = set(c.tagged(p["tag"]))
    return {"a": qs, "b": set(c.title_postings().get(p["word"], ())), "c": qs & _accepted(c)}


H2 = Template(
    "H2", "F", "dev", "ids", "question",
    ("Which questions tagged with \"{tag}\" have either the word \"{word}\" in their title or an accepted answer?",
     "List questions carrying the tag \"{tag}\" that either feature the word \"{word}\" in their title or possess an accepted answer.",
     "Find any question with the \"{tag}\" tag that has an accepted answer or includes the word \"{word}\" in its title."),
    (Step("a", "retrieve", "graph", question="questions tagged TAG", output="question"),
     Step("b", "retrieve", "doc", "filter", question="questions whose title contains WORD", output="question"),
     Step("c", "retrieve", "graph", question="questions tagged TAG that have an accepted answer", output="question"),
     Step("d", "intersect", inputs=("a", "b"), output="question"),
     Step("e", "union", inputs=("d", "c"), output="question")),
    h2_sample, h2_gold, step_sets=h2_steps, group_caps=(("tag", 1), ("word", 1)))


# ---------------------------------------------------------------- shared: years, up-votes, answers

def _question_year(c: Ctx) -> dict[int, int]:
    """question -> year of creation (the first four characters of the UTC date, as the SQL EXTRACT(YEAR ...))."""
    return c.cached("fc_question_year", lambda: {
        q: int(d[:4]) for q, d in c.db.execute("SELECT id, creation_date FROM v_questions") if d})


def _questions_of_year(c: Ctx) -> dict[int, set[int]]:
    def build():
        out: dict[int, set[int]] = {}
        for q, y in _question_year(c).items():
            out.setdefault(y, set()).add(q)
        return out
    return c.cached("fc_questions_of_year", build)


def _up_votes(c: Ctx) -> dict[int, int]:
    """user -> up_votes cast, missing as 0."""
    return c.cached("fc_up_votes", lambda: {u: (v or 0) for u, v in c.db.execute("SELECT id, up_votes FROM v_users")})


def _askers_of(c: Ctx, questions) -> set[int]:
    from .catalogue_d import _asker_of
    asker = _asker_of(c)
    return {asker[q] for q in questions if q in asker}


# ---------------------------------------------------------------- H1

H1_EXCLUDED_YEARS = (2009,)


def _h1_year_class(c: Ctx) -> dict[int, str]:
    sizes = {y: len(qs) for y, qs in _questions_of_year(c).items() if y not in H1_EXCLUDED_YEARS}
    return {y: ("over-cap" if n > c.h_cap else "within-cap") for y, n in sizes.items()}


def _h1_questions(c: Ctx, badge: str) -> set[int]:
    qbu = _questions_by_user(c)
    return {q for u in _badge_holders(c).get(badge, ()) for q in qbu.get(u, ())}


def h1_sample(c: Ctx, n: int) -> list[dict]:
    holders, year, cls = _badge_holders(c), _question_year(c), _h1_year_class(c)
    badges = sorted(b for b, us in holders.items() if 1 <= len(us) <= c.h_cap)
    c.rng.shuffle(badges)
    out: list[dict] = []
    for b in badges:
        qs = _h1_questions(c, b)
        if not qs or len(qs) > c.h_cap:
            c.notes["H1 badges whose holders' questions are 0 or above H"] += 1
            continue
        by_year = Counter(year[q] for q in qs if q in year)
        for stratum in ("over-cap", "within-cap"):
            ys = sorted(y for y, k in cls.items() if k == stratum and by_year.get(y, 0) >= 1)
            if ys:
                out.append({"stratum": stratum, "badge": b, "year": c.rng.choice(ys)})
    return out


def h1_gold(c: Ctx, p: dict) -> Gold:
    holders = _badge_holders(c).get(p["badge"], set())
    qs, yq = _h1_questions(c, p["badge"]), _questions_of_year(c).get(p["year"], set())
    return Gold(len(qs & yq), {}, bound={"a": len(holders)},
                steps={"a": len(holders), "b": len(qs), "c": len(yq)})


def h1_steps(c: Ctx, p: dict) -> dict[str, set]:
    return {"a": set(_badge_holders(c).get(p["badge"], set())), "b": _h1_questions(c, p["badge"]),
            "c": set(_questions_of_year(c).get(p["year"], set()))}


H1 = Template(
    "H1", "F", "dev", "scalar", "number",
    ("How many questions were asked in {year} by users who hold the badge \"{badge}\"?",
     "Give the total count of questions created in {year} authored by users with the \"{badge}\" badge.",
     "What is the number of questions created during {year} by people awarded the \"{badge}\" badge?"),
    (Step("a", "retrieve", "sql", question="users holding the badge BADGE", output="user"),
     Step("b", "retrieve", "graph", question="questions asked by these users", inputs=("a",),
          bind=Bind("a", "user", "a"), output="question"),
     Step("c", "retrieve", "sql", question="questions created in YEAR", output="question"),
     Step("d", "intersect", inputs=("b", "c"), output="question"),
     Step("e", "count", inputs=("d",), output="number", kind="scalar")),
    h1_sample, h1_gold, step_sets=h1_steps,
    quota=(("over-cap", 2), ("within-cap", 1)),
    group_caps=(("badge", 1), ("year", {"over-cap": 2, "within-cap": 3})))


# ---------------------------------------------------------------- H3

H3_RANKS = (5, 20, 40)
H3_MIN_BODIES = 20           # as the N1 body-word rule
# Fixed once from the reading of the seeded sample of 200 of the eligible H3 words (decision 3 of 2026-11-26,
# second round); empty until that reading.
H3_EXCLUDED: dict[str, str] = {}


def _answer_author(c: Ctx) -> dict[int, int]:
    return c.cached("fc_answer_author", lambda: dict(c.rows("SELECT answer_id, user_id FROM v_answered")))


def h3_vocab(c: Ctx) -> dict[str, frozenset[int]]:
    """eligible word -> answers whose raw body contains it (as the store indexes the body)."""
    def build():
        df: Counter = Counter()
        seen: Counter = Counter()
        lower: Counter = Counter()
        bodies = c.rows("SELECT id, body FROM posts WHERE post_type_id = 2 AND body <> ''")
        for _, body in bodies:
            words = set()
            for raw in _TOKEN_ANY_CASE.findall(clean_body(body)):
                w = raw.lower()
                if _indexable(w):
                    words.add(w)
                    seen[w] += 1
                    lower[w] += raw == w
            df.update(words)
        eligible = {w for w, n in df.items() if n >= H3_MIN_BODIES and lower[w] >= BODY_MIN_LOWER * seen[w]
                    and w not in EXCLUDED_WORDS and w not in H3_EXCLUDED}
        post: dict[str, set[int]] = {}
        for aid, body in bodies:
            for w in set(union.tokens(body or "")) & eligible:
                post.setdefault(w, set()).add(aid)
        return {w: frozenset(v) for w, v in post.items()}
    return c.cached("fc_h3_vocab", build)


def _h3_authors(c: Ctx, word: str) -> set[int]:
    author = _answer_author(c)
    return {author[a] for a in h3_vocab(c).get(word, ()) if a in author}


def h3_sample(c: Ctx, n: int) -> list[dict]:
    vocab, rep = h3_vocab(c), _reputation(c)
    words = sorted(w for w, ans in vocab.items() if len(ans) <= c.h_cap)
    c.rng.shuffle(words)
    out: list[dict] = []
    for w in words:
        authors = _h3_authors(c, w)
        if not authors or len(authors) > c.h_cap:
            continue
        reps = sorted(rep.get(u, 0) for u in authors)
        used: set[int] = set()
        available = sum(k <= len(reps) for k in H3_RANKS)
        collided = False
        for k in H3_RANKS:
            if k > len(reps):
                c.notes[f"H3 rank {k:02d} unavailable"] += 1
                continue
            number = reps[k - 1] + 1
            if number in used:
                c.notes["H3 rank collision (same NUMBER as a smaller rank)"] += 1
                collided = True
                continue
            used.add(number)
            out.append({"stratum": f"rank {k:02d}", "word": w, "number": number, "rank": k})
        if available >= 2:
            c.notes["H3 words with two or more ranks"] += 1
            c.notes["H3 words with a collision"] += collided
    return out


def _users_under(c: Ctx, n: int) -> set[int]:
    return {u for u, r in _reputation(c).items() if r < n}


def h3_gold(c: Ctx, p: dict) -> Gold:
    answers, authors = h3_vocab(c).get(p["word"], frozenset()), _h3_authors(c, p["word"])
    rep = _reputation(c)
    hit = sorted(u for u in authors if rep.get(u, 0) < p["number"])
    n_c = sum(1 for r in rep.values() if r < p["number"])
    return Gold(hit, {}, bound={"a": len(answers)}, steps={"a": len(answers), "b": len(authors), "c": n_c})


def h3_steps(c: Ctx, p: dict) -> dict[str, set]:
    return {"a": set(h3_vocab(c).get(p["word"], ())), "b": _h3_authors(c, p["word"]),
            "c": _users_under(c, p["number"])}


H3 = Template(
    "H3", "F", "dev", "ids", "user",
    ("Which users with a reputation under {number} wrote an answer containing \"{word}\"?",
     "List the users whose reputation score is less than {number} who authored an answer featuring the word \"{word}\".",
     "Find authors of answers containing \"{word}\" who have a reputation below {number}."),
    (Step("a", "retrieve", "doc", "filter", question="answers whose text contains WORD", output="answer"),
     Step("b", "retrieve", "graph", question="users who wrote these answers", inputs=("a",),
          bind=Bind("a", "answer", "a"), output="user"),
     Step("c", "retrieve", "sql", question="users with a reputation under NUMBER", output="user"),
     Step("d", "intersect", inputs=("b", "c"), output="user")),
    h3_sample, h3_gold, step_sets=h3_steps, group_caps=(("word", 1),))


# ---------------------------------------------------------------- H4

H4_YEARS = (2010, 2011, 2012, 2024)
H4_RANKS = (5, 20, 40)


def h4_number(c: Ctx, year: int, k: int) -> int | None:
    """up_votes of the k-th highest asker of the year, or None if there is no such asker or it is 0."""
    upv = _up_votes(c)
    vals = sorted((upv.get(u, 0) for u in _askers_of(c, _questions_of_year(c).get(year, ()))), reverse=True)
    if k > len(vals) or vals[k - 1] <= 0:
        return None
    return vals[k - 1]


def _h4_cells(c: Ctx, years, ranks, name: str) -> list[dict]:
    """The cells in rotation order: for year index yi and wording variant v, the rank of index (v - yi) mod 3, so
    that variant = (rank index + year index) mod 3. A cell without a valid NUMBER, or whose NUMBER equals that of
    another rank of the same year (the same question), is dropped and noted, not replaced."""
    out: list[dict] = []
    for yi, y in enumerate(years):
        numbers = {k: h4_number(c, y, k) for k in ranks}
        for v in range(3):
            k = ranks[(v - yi) % 3]
            number = numbers[k]
            if number is None or list(numbers.values()).count(number) > 1:
                c.notes[f"{name} cell ({y}, rank {k}) unavailable: not replaced"] += 1
                continue
            out.append({"stratum": "all", "year": y, "number": number, "rank": k, "variant": v})
    return out


def h4_sample(c: Ctx, n: int) -> list[dict]:
    return _h4_cells(c, H4_YEARS, H4_RANKS, "H4")


def h4_gold(c: Ctx, p: dict) -> Gold:
    yq = _questions_of_year(c).get(p["year"], set())
    askers, upv = _askers_of(c, yq), _up_votes(c)
    hit = sorted(u for u in askers if upv.get(u, 0) >= p["number"])
    n_c = sum(1 for v in upv.values() if v >= p["number"])
    return Gold(hit, {}, bound={"a": len(yq)}, steps={"a": len(yq), "b": len(askers), "c": n_c})


def h4_steps(c: Ctx, p: dict) -> dict[str, set]:
    yq = set(_questions_of_year(c).get(p["year"], set()))
    return {"a": yq, "b": _askers_of(c, yq), "c": {u for u, v in _up_votes(c).items() if v >= p["number"]}}


H4 = Template(
    "H4", "F", "dev", "ids", "user",
    ("Which users who asked a question in {year} have cast at least {number} up-votes?",
     "List the authors of questions created during {year} who have cast {number} or more up-votes.",
     "Find users who authored a question in the year {year} and have cast a minimum of {number} up-votes."),
    (Step("a", "retrieve", "sql", question="questions created in YEAR", output="question"),
     Step("b", "retrieve", "graph", question="users who asked these questions", inputs=("a",),
          bind=Bind("a", "question", "a"), output="user"),
     Step("c", "retrieve", "sql", question="users who have cast at least NUMBER up-votes", output="user"),
     Step("d", "intersect", inputs=("b", "c"), output="user")),
    h4_sample, h4_gold, step_sets=h4_steps)

H4P_YEARS = (2011, 2024)
H4P_RANKS = (10, 30)


def h4p_sample(c: Ctx, n: int) -> list[dict]:
    out = []
    for y, k, v in ((H4P_YEARS[0], H4P_RANKS[0], 0), (H4P_YEARS[1], H4P_RANKS[1], 1)):
        number = h4_number(c, y, k)
        if number is None:
            c.notes[f"H4P cell ({y}, rank {k}) unavailable"] += 1
            continue
        out.append({"stratum": "all", "year": y, "number": number, "rank": k, "variant": v})
    return out


H4P = Template("H4P", "F", "probe", H4.answer_type, H4.entity, H4.paraphrases, H4.plan, h4p_sample, h4_gold,
               step_sets=h4_steps)


TEMPLATES: list[Template] = [S1, S2, H2, H1, H3, H4]
PROBE_TEMPLATES: list[Template] = [H4P]


# ---------------------------------------------------------------- word lists for reading

def h2_word_table(c: Ctx) -> list[tuple[str, int]]:
    """(word, number of tags for which it is eligible), for every word eligible for H2."""
    counts: dict[str, int] = {}
    for tag, qs in c.tag_members().items():
        if 10 <= len(qs) <= c.h_cap:
            for w in h2_words_for_tag(c, tag):
                counts[w] = counts.get(w, 0) + 1
    return sorted(counts.items())


def main(argv: list[str] | None = None) -> int:
    """Write the eligible word list of a shape (all of it) and, if it is longer than ``WORD_SAMPLE``, a seeded sample
    of ``WORD_SAMPLE`` words for reading. Read-only on the oracle."""
    from .questions import H_BENCH, T_INDEPENDENT
    ap = argparse.ArgumentParser(description="Eligible word lists of the first-contact shapes, for reading")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--words", required=True, choices=["H2", "H3"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    db = union.open_union(args.oracle)
    ctx = Ctx(db, 0, T_INDEPENDENT, H_BENCH)
    if args.words == "H2":
        rows, unit = h2_word_table(ctx), "tags"
    else:
        rows, unit = sorted((w, len(a)) for w, a in h3_vocab(ctx).items()), "answers"
    with open(args.out, "w") as fh:
        fh.write(f"word\t{unit}\n")
        for w, n in rows:
            fh.write(f"{w}\t{n}\n")
    msg = f"{len(rows)} eligible {args.words} words written to {args.out}"
    if len(rows) > WORD_SAMPLE:
        sample = sorted(random.Random(WORD_SAMPLE_SEED).sample(rows, WORD_SAMPLE))
        sample_path = args.out.rsplit(".", 1)[0] + f".sample{WORD_SAMPLE}.tsv"
        with open(sample_path, "w") as fh:
            fh.write(f"word\t{unit}\n")
            for w, n in sample:
                fh.write(f"{w}\t{n}\n")
        msg += f"; a seeded sample of {WORD_SAMPLE} (seed {WORD_SAMPLE_SEED}) written to {sample_path}"
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
