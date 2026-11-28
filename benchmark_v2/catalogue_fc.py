"""
benchmark_v2/catalogue_fc.py
============================
First-contact templates (pre-registration section 3; build item 3; plan approved by Siva on 2026-11-26 with
changes, session log of 2026-11-26 sections 29 to 31). Class F: the registered cap rules checked step by step,
no lower threshold T (``questions._check_f``).

Batch 104: S1, S2 and H2. Batch 105 adds H1, H3 and H4.

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

Draw caps (``group_caps``): one question per tag (S1, H2), one per word (H2), one per TAG_A and at most two per
TAG_B (S2).

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
from .catalogue_d import _askers_by_tag
from .catalogue_n1 import TitleGuard
from .questions import MAX_SET, Ctx, Gold, Step, Template

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


TEMPLATES: list[Template] = [S1, S2, H2]


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
    ap.add_argument("--words", required=True, choices=["H2"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    db = union.open_union(args.oracle)
    rows = h2_word_table(Ctx(db, 0, T_INDEPENDENT, H_BENCH))
    with open(args.out, "w") as fh:
        fh.write("word\ttags\n")
        for w, n in rows:
            fh.write(f"{w}\t{n}\n")
    msg = f"{len(rows)} eligible {args.words} words written to {args.out}"
    if len(rows) > WORD_SAMPLE:
        sample = sorted(random.Random(WORD_SAMPLE_SEED).sample(rows, WORD_SAMPLE))
        sample_path = args.out.rsplit(".", 1)[0] + f".sample{WORD_SAMPLE}.tsv"
        with open(sample_path, "w") as fh:
            fh.write("word\ttags\n")
            for w, n in sample:
                fh.write(f"{w}\t{n}\n")
        msg += f"; a seeded sample of {WORD_SAMPLE} (seed {WORD_SAMPLE_SEED}) written to {sample_path}"
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
