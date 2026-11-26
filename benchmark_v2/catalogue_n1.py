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
                blocks and TeX removed), is written in lower case in at least ``BODY_MIN_LOWER`` of its
                occurrences there (Batch 102b: this keeps out names, such as "aaron" in the first dev
                probe), and occurs in no title at all.

Excluded words (Batch 102d). The whole eligible list of the intermediate cell (116 words on Siva's oracle,
fingerprint 5723274bb899b21a) was read once on 2026-11-26, before any model saw an N1 question; the eight words
of ``EXCLUDED_WORDS`` were excluded by category (site user names, a URL fragment, misspellings, slang, a data
value), so that the cell asks about plain words. The list only removes words; it never adds one.

One cell per tag (Batch 102b). Within a template a tag is used by at most one cell, so two N1 questions
never share a tag. The samplers alternate which cell a tag is offered to first (even positions in the
shuffled tag order: final first; odd positions: intermediate first), so that neither cell is left with
only the tags the other could not use.

Title guard (Batch 102, corrected in Batch 103). Batch 102 assumed that the document store indexes titles with the
stemming ``english`` analyzer of Elasticsearch. It does not: ``heterorag/es_index.py`` defines its own analyzer
under that name, the standard tokenizer with English stop words and no stemming, which overrides the built-in one
for this index. A D05E word is still rejected when it clashes with a relevant title token (the tag's titles for
the final cell, every title for the intermediate cell): the word starts with the stem prefix of a title token, or
a title token starts with the word's stem prefix, where the stem prefix of x is its first max(4, len(x) - 3)
letters and title tokens of 4 letters or more are considered. Without stemming the guard is not needed for
correctness; it only narrows the vocabulary. It is kept because the vocabulary of the intermediate cell was read
in full with it in place (2026-11-26) and removing it would admit unread words. Every generated instance is still
validated against the live stores (``validate_instances``, which also checks the cell's condition step by step).

Like the rest of the generator, this module does not import the system under test.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter

from . import union
from .catalogue_d import D01, D05, _askers_by_tag, _reputation, _reputations_sorted, d01_gold, d01_steps, d05_steps
from .questions import N1_FINAL, N1_INTERMEDIATE, STOPWORDS, Ctx, Gold, Template

BODY_MIN_DF = 20            # an intermediate-cell word occurs in at least this many question bodies
BODY_MIN_LOWER = 0.9        # ... and is written in lower case in at least this share of its occurrences
WORDS_TRIED_PER_TAG = 60    # final cell of D05E: random candidate words tried per tag
TAGS_SCANNED = 2500         # as in D01 and D05
MIN_PREFIX = 4              # title tokens shorter than this are not considered by the stemming guard

# Batch 102d: excluded from the intermediate cell of D05E after reading the whole eligible list (2026-11-26).
EXCLUDED_WORDS: dict[str, str] = {
    "jbowman": "site user name",
    "ttnphns": "site user name",
    "noredirect": "URL fragment",
    "aswell": "misspelling",
    "beacuse": "misspelling",
    "beeing": "misspelling",
    "wanna": "slang",
    "setosa": "data value (iris species)",
}

_DROP = re.compile(r"<pre\b.*?</pre>|<code\b.*?</code>|\$\$.*?\$\$|\$[^$]*\$"
                   r"|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{([a-z*]+)\}.*?\\end\{\1\}", re.S | re.I)
_TEXCMD = re.compile(r"\\[A-Za-z]+")       # a TeX command left outside a recognised math span (Batch 102c: "vdots")
_TAG = re.compile(r"<[^>]+>|&[a-z]+;|&#\d+;", re.I)
_TOKEN_ANY_CASE = re.compile(union._TOKEN.pattern, re.I)   # the tokenizer of ``union.tokens``, case kept


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
    return _TAG.sub(" ", _TEXCMD.sub(" ", _DROP.sub(" ", html)))


def body_word_table(c: Ctx) -> list[tuple[str, int, float]]:
    """(word, number of question bodies, share of its body occurrences in lower case) for every word eligible for
    the intermediate cell of D05E, sorted by word. Listed in full by ``python -m benchmark_v2.catalogue_n1``."""
    def build():
        df: Counter = Counter()
        seen: Counter = Counter()
        lower: Counter = Counter()
        for (body,) in c.db.execute("SELECT body FROM v_questions WHERE body <> ''"):
            words = set()
            for raw in _TOKEN_ANY_CASE.findall(clean_body(body)):
                w = raw.lower()
                if _indexable(w):
                    words.add(w)
                    seen[w] += 1
                    lower[w] += raw == w
            df.update(words)
        guard = _global_guard(c)
        return sorted((w, n, lower[w] / seen[w]) for w, n in df.items()
                      if n >= BODY_MIN_DF and lower[w] >= BODY_MIN_LOWER * seen[w] and not guard.clashes(w)
                      and w not in EXCLUDED_WORDS)
    return c.cached("n1_body_table", build)


def _body_words(c: Ctx) -> list[str]:
    """Words of at least ``BODY_MIN_DF`` question bodies, written in lower case in at least ``BODY_MIN_LOWER``
    of their occurrences there, that clash with no title; sorted."""
    return [w for w, _, _ in body_word_table(c)]


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
    for pos, tag in enumerate(tags[:TAGS_SCANNED]):
        us = askers[tag]
        known = [rep[u] for u in us if u in rep]
        if not known:
            continue

        def as_final():
            if len(final) >= want:
                return False
            i = bisect_right(reps, max(known))
            if i >= len(reps):
                return False
            n_sql = len(reps) - i                        # users with reputation >= reps[i]
            if not (max(len(us), n_sql) > c.t_independent and min(len(us), n_sql) <= c.h_cap):
                return False
            final.append({"stratum": N1_FINAL, "tag": tag, "min_reputation": reps[i]})
            return True

        def as_inter():
            if len(inter) >= want or not (c.t_independent < len(us) <= c.h_cap):
                return False
            inter.append({"stratum": N1_INTERMEDIATE, "tag": tag, "min_reputation": r_inter})
            return True

        _ = (as_final() or as_inter()) if pos % 2 == 0 else (as_inter() or as_final())
        if len(final) >= want and len(inter) >= want:
            break
    return final + inter


D01E = Template("D01E", "D", "dev", "ids", "user", D01.paraphrases, D01.plan, d01e_sample, d01_gold,
                empty_step="b", step_sets=d01_steps)


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
    for pos, tag in enumerate(tags[:TAGS_SCANNED]):
        qs = members[tag]

        def as_final():
            if len(final) >= want or not common or not (8 <= len(qs) <= min(1000, c.h_cap)):
                return False
            guard = TitleGuard(x for q in qs for x in titles.get(q, ()))
            tried = c.rng.sample(common, min(WORDS_TRIED_PER_TAG, len(common)))
            w = next((w for w in tried if not guard.clashes(w)), None)
            if w is None:
                return False
            final.append({"stratum": N1_FINAL, "tag": tag, "word": w})
            return True

        def as_inter():
            if len(inter) >= want or not body_words or not (c.t_independent < len(qs) <= c.h_cap):
                return False
            inter.append({"stratum": N1_INTERMEDIATE, "tag": tag, "word": body_words[len(inter) % len(body_words)]})
            return True

        _ = (as_final() or as_inter()) if pos % 2 == 0 else (as_inter() or as_final())
        if len(final) >= want and len(inter) >= want:
            break
    return final + inter


def d05e_gold(c: Ctx, p: dict) -> Gold:
    g, d = set(c.tagged(p["tag"])), set(c.title_postings().get(p["word"], ()))
    return Gold(sorted(g & d), {"graph": len(g), "doc": len(d)})


D05E = Template("D05E", "D", "dev", "ids", "question", D05.paraphrases, D05.plan, d05e_sample, d05e_gold,
                empty_step="b", step_sets=d05_steps)


TEMPLATES: list[Template] = [D01E, D05E]


def main(argv: list[str] | None = None) -> int:
    """List every word eligible for the intermediate cell of D05E, for a reading of the whole list before the
    freeze. Read-only on the oracle; writes one tab-separated file."""
    import argparse

    from .questions import H_BENCH, T_INDEPENDENT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    db = union.open_union(args.oracle)
    rows = body_word_table(Ctx(db, 0, T_INDEPENDENT, H_BENCH))
    with open(args.out, "w") as fh:
        fh.write("word\tbodies\tlower_share\n")
        for w, n, share in rows:
            fh.write(f"{w}\t{n}\t{share:.3f}\n")
    print(f"{len(rows)} eligible words (bodies >= {BODY_MIN_DF}, lower case >= {BODY_MIN_LOWER:.0%}, no title clash) "
          f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
