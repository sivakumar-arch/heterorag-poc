"""
benchmark_v2/catalogue_d.py
===========================
Templates for class D: an independent formulation needs a sub-result above T, or none exists.

The gold plan is the *logical* plan. It contains a bind only where a store lacks the predicate
(D04); where the stores could answer each part alone (D01, D05) the plan is independent and
the choice between independent and dependent execution belongs to the system (design B).
The benchmark records the sub-result sizes, so every instance can be placed on the
cardinality axis that decides that choice:

    independent sub-result in (T, H]   independent execution works but reads many rows
    independent sub-result above H     independent execution is truncated by the hard cap

Every instance must still be answerable exactly: the smaller set (D01, D05) or the injected
set (D04) has at most H entities, checked in ``questions.check_gold``.

Like the rest of the generator, this module does not import the system under test.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter

from .questions import MAX_SET, Bind, Ctx, Gold, Step, Template, pick


# ---------------------------------------------------------------- shared derived structures

def _asker_of(c: Ctx) -> dict[int, int]:
    """question id -> asking user (loaded users only)."""
    return c.cached("asker_of", lambda: dict(c.rows("SELECT question_id, user_id FROM v_asked")))


def _askers_by_tag(c: Ctx) -> dict[str, set[int]]:
    """tag -> users who asked a question carrying it."""
    def build():
        asker, out = _asker_of(c), {}
        for qid, tag in c.db.execute("SELECT question_id, tag FROM v_tagged"):
            u = asker.get(qid)
            if u is not None:
                out.setdefault(tag, set()).add(u)
        return out
    return c.cached("askers_by_tag", build)


def _badge_holders(c: Ctx) -> dict[str, set[int]]:
    """badge name -> user ids that hold it (one scan of the badges table)."""
    def build():
        out: dict[str, set[int]] = {}
        for name, uid in c.db.execute("SELECT name, user_id FROM badges"):
            out.setdefault(name, set()).add(uid)
        return out
    return c.cached("badge_holders", build)


def _questions_by_user(c: Ctx) -> dict[int, list[int]]:
    def build():
        out: dict[int, list[int]] = {}
        for q, u in _asker_of(c).items():
            out.setdefault(u, []).append(q)
        return out
    return c.cached("questions_by_user", build)


def _tags_by_question(c: Ctx) -> dict[int, list[str]]:
    def build():
        out: dict[int, list[str]] = {}
        for tag, qs in c.tag_members().items():
            for q in qs:
                out.setdefault(q, []).append(tag)
        return out
    return c.cached("tags_by_question", build)


def _reputation(c: Ctx) -> dict[int, int]:
    return c.cached("reputation", lambda: dict(c.rows("SELECT id, reputation FROM v_users")))


def _reputations_sorted(c: Ctx) -> list[int]:
    return c.cached("reputations_sorted", lambda: sorted(_reputation(c).values()))


def _users_at_least(c: Ctx, r: int) -> int:
    reps = _reputations_sorted(c)
    return len(reps) - bisect_left(reps, r)


def _band(c: Ctx, n: int) -> str:
    return f"{c.t_independent}-{c.h_cap}" if n <= c.h_cap else f">{c.h_cap}"


def _fill(pools: dict[str, list[dict]], want: int) -> bool:
    return bool(pools) and all(len(v) >= want for v in pools.values())


# ---- D01 (graph and sql) askers of a tag with a reputation floor -----------------------

D01_THRESHOLD_RANKS = (1, 2, 3, 5, 10, 20, 30)


def d01_sample(c: Ctx, n: int):
    """Tag-first. For a tag, the reputation of the j-th best asker is a threshold that keeps at
    least j askers, so the answer size is controlled. The strata are the side that is large
    (the user table under the reputation filter, or the askers of a big tag) and the size band
    of that side against T and H."""
    askers, rep = _askers_by_tag(c), _reputation(c)
    tags = sorted(t for t, us in askers.items() if len(us) >= 3)
    c.rng.shuffle(tags)
    want = n * 4
    pools: dict[str, list[dict]] = {}
    for tag in tags[:2500]:
        us = askers[tag]
        reps = sorted((rep[u] for u in us if u in rep), reverse=True)
        for j in D01_THRESHOLD_RANKS:
            if j > len(reps):
                break
            r = reps[j - 1]
            inter = sum(1 for x in reps if x >= r)
            n_sql = _users_at_least(c, r)
            big = max(len(us), n_sql)
            if not (1 <= inter <= MAX_SET) or big <= c.t_independent or min(len(us), n_sql) > c.h_cap:
                continue
            side = "sql" if n_sql >= len(us) else "graph"
            name = f"{side} large {_band(c, big)}"
            if len(pools.setdefault(name, [])) < want:
                pools[name].append({"stratum": name, "tag": tag, "min_reputation": r})
        if _fill({k: v for k, v in pools.items()}, want) and len(pools) >= 4:
            break
    return [x for v in pools.values() for x in v]


def d01_gold(c: Ctx, p: dict) -> Gold:
    us, rep, r = _askers_by_tag(c).get(p["tag"], set()), _reputation(c), p["min_reputation"]
    hit = sorted(u for u in us if rep.get(u, -1) >= r)
    return Gold(hit, {"graph": len(us), "sql": _users_at_least(c, r)})


D01 = Template(
    "D01", "D", "dev", "ids", "user",
    ("Which users who asked a question tagged \"{tag}\" have a reputation of at least {min_reputation}?",
     "Among the users who have asked questions with the tag \"{tag}\", which have reputation {min_reputation} or more?",
     "List the askers of \"{tag}\" questions whose reputation is {min_reputation} or higher."),
    (Step("a", "retrieve", "graph", question="users who asked a question with the tag", output="user"),
     Step("b", "retrieve", "sql", question="users with reputation >= R", output="user"),
     Step("c", "intersect", inputs=("a", "b"), output="user")),
    d01_sample, d01_gold)


# ---- D04 (sql then graph) tags used by the questions of badge holders -------------------

D04_BANDS = (("1-2 holders", 1, 2), ("3-10 holders", 3, 10), ("11-50 holders", 11, 50))


def d04_sample(c: Ctx, n: int):
    holders = _badge_holders(c)
    out = []
    for name, lo, hi in D04_BANDS:
        names = sorted(b for b, us in holders.items() if lo <= len(us) <= hi)
        out += [{"stratum": name, "badge": b} for b in pick(names, n * 4, c.rng)]
    return out


def d04_gold(c: Ctx, p: dict) -> Gold:
    holders = _badge_holders(c).get(p["badge"], set())
    by_user, by_q = _questions_by_user(c), _tags_by_question(c)
    tags = sorted({t for u in holders for q in by_user.get(u, ()) for t in by_q.get(q, ())})
    return Gold(tags, {"sql": len(holders)}, bound={"a": len(holders)}, independent=False)


D04 = Template(
    "D04", "D", "dev", "ids", "tag",
    ("Which tags appear on questions asked by users who hold the \"{badge}\" badge?",
     "List the tags used on questions posted by holders of the \"{badge}\" badge.",
     "What tags do the questions of users with the \"{badge}\" badge carry?"),
    (Step("a", "retrieve", "sql", question="users holding the badge", output="user"),
     Step("b", "retrieve", "graph", question="tags of the questions asked by these users",
          inputs=("a",), bind=Bind("a", "user", "a"), output="tag")),
    d04_sample, d04_gold)


# ---- D05 (graph and doc) tagged T, title contains a common word -------------------------

def d05_sample(c: Ctx, n: int):
    """Tag-first, as P01, but the word is common (more than T questions). The strata are the
    size band of the word's postings against T and H."""
    post, members, qwords = c.title_postings(), c.tag_members(), c.title_words()
    tags = sorted(t for t, qs in members.items() if 8 <= len(qs) <= min(1000, c.h_cap))
    c.rng.shuffle(tags)
    want = n * 4
    pools: dict[str, list[dict]] = {}
    for tag in tags[:2500]:
        counts = Counter(w for qid in members[tag] for w in qwords.get(qid, ())
                         if len(post[w]) > c.t_independent)
        by_band: dict[str, list[str]] = {}
        for w, k in counts.items():
            if 1 <= k <= MAX_SET:
                by_band.setdefault(f"df {_band(c, len(post[w]))}", []).append(w)
        for name, words in by_band.items():
            if len(pools.setdefault(name, [])) < want:
                pools[name].append({"stratum": name, "tag": tag, "word": c.rng.choice(sorted(words))})
        if len(pools) >= 2 and _fill(pools, want):
            break
    return [x for v in pools.values() for x in v]


def d05_gold(c: Ctx, p: dict) -> Gold:
    g, d = set(c.tagged(p["tag"])), set(c.title_postings()[p["word"]])
    return Gold(sorted(g & d), {"graph": len(g), "doc": len(d)})


D05 = Template(
    "D05", "D", "dev", "ids", "question",
    ("Which questions tagged \"{tag}\" have the word \"{word}\" in their title?",
     "List the questions with the tag \"{tag}\" whose title contains \"{word}\".",
     "Find questions carrying the tag \"{tag}\" with \"{word}\" in the title."),
    (Step("a", "retrieve", "graph", question="questions tagged", output="question"),
     Step("b", "retrieve", "doc", "filter", "title contains word", output="question"),
     Step("c", "intersect", inputs=("a", "b"), output="question")),
    d05_sample, d05_gold)


TEMPLATES: list[Template] = [D01, D04, D05]
