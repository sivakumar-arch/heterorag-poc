"""
benchmark_v2/catalogue_n1.py
============================
N1 correct-empty controls (pre-registration section 3; Batch 102, plan approved 2026-11-18).

Two templates built from the D01 and D05 shapes: the same three paraphrases, the same gold plan
(no ``must_exist`` on any step), parameters chosen so that the true answer is empty. Each has two
cells, the stratum of the instance:

  N1 final         both sub-results are non-empty; only their intersection is empty;
  N1 intermediate  the step ``b`` (D01E: the SQL reputation step; D05E: the document title step)
                   returns nothing; the other sub-result is within (T, H].

D01E
  final         R = the lowest reputation in the data that is above the tag's best asker (a real
                user's value, as in D01), so at least one user passes the SQL filter and no asker does.
  intermediate  R = the smallest multiple of 1,000 above the highest reputation in the data; every
                instance of the cell shares this R and differs by tag.

D05E
  final         a title word with more than T title postings (as in D05) that occurs in no title of
                the tag's questions;
  intermediate  a word that occurs in the bodies of at least ``BODY_MIN_DF`` questions (markup, code
                blocks and TeX removed) and in no title at all.

Stemming guard. The document store indexes titles with the ``english`` analyzer, which stems; the
oracle matches whole words. An empty answer is where the two differ most (the oracle finds no
"regression" in a title that says "regressions"; the store does). A D05E word is therefore rejected
when it clashes with any relevant title token (the tag's titles for the final cell, every title for
the intermediate cell): the word starts with the stem prefix of a title token, or a title token
starts with the word's stem prefix, where the stem prefix of x is its first max(4, len(x) - 3)
letters and title tokens of 4 letters or more are considered. The rule is conservative (it rejects
more words than the analyzer would match) but not exact (a 3-letter title token can still stem
together with a longer word); every generated instance is validated against the live stores
(``validate_instances``, which also checks the cell's condition step by step), and a mismatching
instance is dropped there.

Like the rest of the generator, this module does not import the system under test.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter

from . import union
from .catalogue_d import D01, D05, _askers_by_tag, _reputation, _reputations_sorted, d01_gold
from .questions import N1_FINAL, N1_INTERMEDIATE, STOPWORDS, Ctx, Gold, Template

BODY_MIN_DF = 20            # an intermediate-cell word occurs in at least this many question bodies
WORDS_TRIED_PER_TAG = 60    # final cell of D05E: random candidate words tried per tag
TAGS_SCANNED = 2500         # as in D01 and D05
MIN_PREFIX = 4              # title tokens shorter than this are not considered by the stemming guard

_DROP = re.compile(r"<pre\b.*?</pre>|<code\b.*?</code>|\$\$.*?\$\$|\$[^$]*\$", re.S | re.I)
_TAG = re.compile(r"<[^>]+>|&[a-z]+;|&#\d+;", re.I)


# ---------------------------------------------------------------- stemming guard

def stem_prefix(w: str) -> str:
    return w[:max(MIN_PREFIX, len(w) - 3)]


class TitleGuard:
    """The title tokens a word must not clash with, as two prefix sets."""

    def __init__(self, tokens) -> None:
        toks = {x for x in tokens if len(x) >= MIN_PREFIX}
        self.long = {x[:k] for x in toks for k in range(MIN_PREFIX, len(x) + 1)}
        self.stems = {stem_prefix(x) for x in toks}

    def clashes(self, word: str) -> bool:
        if stem_prefix(word) in self.long:                       # a title token starts with the word's stem
            return True
        return any(word[:k] in self.stems for k in range(MIN_PREFIX, len(word) + 1))


def _indexable(w: str) -> bool:
    return w.isalpha() and len(w) >= 5 and w not in STOPWORDS


def _title_tokens(c: Ctx) -> dict[int, frozenset[str]]:
    """question id -> every token of its title (the document store's questions only)."""
    def build():
        return {pid: frozenset(union.tokens(title))
                for pid, title in c.db.execute("SELECT post_id, title FROM v_question_docs")}
    return c.cached("n1_title_tokens", build)


def _global_guard(c: Ctx) -> TitleGuard:
    return c.cached("n1_global_guard", lambda: TitleGuard(x for ts in _title_tokens(c).values() for x in ts))


def clean_body(html: str | None) -> str:
    if not html:
        return ""
    return _TAG.sub(" ", _DROP.sub(" ", html))


def _body_words(c: Ctx) -> list[str]:
    """Words of at least ``BODY_MIN_DF`` question bodies that clash with no title, sorted."""
    def build():
        df: Counter = Counter()
        for (body,) in c.db.execute("SELECT body FROM v_questions WHERE body <> ''"):
            df.update({w for w in union.tokens(clean_body(body)) if _indexable(w)})
        guard = _global_guard(c)
        return sorted(w for w, n in df.items() if n >= BODY_MIN_DF and not guard.clashes(w))
    return c.cached("n1_body_words", build)


# ---------------------------------------------------------------- D01E

def d01e_sample(c: Ctx, n: int):
    askers, rep = _askers_by_tag(c), _reputation(c)
    reps = _reputations_sorted(c)
    if not reps:
        return []
    r_inter = (reps[-1] // 1000 + 1) * 1000
    tags = sorted(askers)
    c.rng.shuffle(tags)
    want = n * 4
    final: list[dict] = []
    inter: list[dict] = []
    for tag in tags[:TAGS_SCANNED]:
        us = askers[tag]
        known = [rep[u] for u in us if u in rep]
        if not known:
            continue
        if len(final) < want:
            i = bisect_right(reps, max(known))
            if i < len(reps):
                n_sql = len(reps) - i                    # users with reputation >= reps[i]
                if max(len(us), n_sql) > c.t_independent and min(len(us), n_sql) <= c.h_cap:
                    final.append({"stratum": N1_FINAL, "tag": tag, "min_reputation": reps[i]})
        if len(inter) < want and c.t_independent < len(us) <= c.h_cap:
            inter.append({"stratum": N1_INTERMEDIATE, "tag": tag, "min_reputation": r_inter})
        if len(final) >= want and len(inter) >= want:
            break
    return final + inter


D01E = Template("D01E", "D", "dev", "ids", "user", D01.paraphrases, D01.plan, d01e_sample, d01_gold,
                empty_step="b")


# ---------------------------------------------------------------- D05E

def d05e_sample(c: Ctx, n: int):
    post, members, titles = c.title_postings(), c.tag_members(), _title_tokens(c)
    common = sorted(w for w, ids in post.items() if len(ids) > c.t_independent)
    body_words = list(_body_words(c))
    c.rng.shuffle(body_words)
    tags = sorted(members)
    c.rng.shuffle(tags)
    want = n * 4
    final: list[dict] = []
    inter: list[dict] = []
    for tag in tags[:TAGS_SCANNED]:
        qs = members[tag]
        if len(final) < want and common and 8 <= len(qs) <= min(1000, c.h_cap):
            guard = TitleGuard(x for q in qs for x in titles.get(q, ()))
            tried = c.rng.sample(common, min(WORDS_TRIED_PER_TAG, len(common)))
            w = next((w for w in tried if not guard.clashes(w)), None)
            if w is not None:
                final.append({"stratum": N1_FINAL, "tag": tag, "word": w})
        if len(inter) < want and body_words and c.t_independent < len(qs) <= c.h_cap:
            inter.append({"stratum": N1_INTERMEDIATE, "tag": tag, "word": body_words[len(inter) % len(body_words)]})
        if len(final) >= want and len(inter) >= want:
            break
    return final + inter


def d05e_gold(c: Ctx, p: dict) -> Gold:
    g, d = set(c.tagged(p["tag"])), set(c.title_postings().get(p["word"], ()))
    return Gold(sorted(g & d), {"graph": len(g), "doc": len(d)})


D05E = Template("D05E", "D", "dev", "ids", "question", D05.paraphrases, D05.plan, d05e_sample, d05e_gold,
                empty_step="b")


TEMPLATES: list[Template] = [D01E, D05E]
