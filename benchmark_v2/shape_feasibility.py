"""
benchmark_v2/shape_feasibility.py
=================================
Pilot check for question shapes written by an outside reader (Reader Kit, Job 1, section 3.6): for each shape,
can blank values be found that give an answer of the size the study needs, and do two readings of the wording
give different answers? Reads only the oracle SQLite file; no system output, no model.

Size rules used (from the generator): a list answer should have 1 to MAX_SET (50) items; every intermediate set
must fit under the hard cap of one retrieve step (H_CAP, 10,000); a count may be any size.

Shapes (names of the first outside hand-in, 2026-11-14; the wording is the reader's, the reading is stated here):
  H1  count of questions created in YEAR by users who hold badge BADGE
  H2  questions tagged TAG that contain WORD in the title (or body, with --body) or have an accepted answer,
      read as: tagged TAG and (contains WORD or accepted)
  H3  users with reputation below NUMBER who wrote an answer containing WORD (reading A); reading B takes the
      answered question (its title) to contain the word
  H4  users who asked a question in YEAR and have at least NUMBER up-votes: reading A the user's own up-vote
      count (the `up_votes` field, votes cast); reading B up-votes received on those questions

    python -m benchmark_v2.shape_feasibility --oracle data/oracle_stats.sqlite > shape_feasibility.txt
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict
from statistics import median

from benchmark_v2.questions import STOPWORDS
from benchmark_v2.union import tokens

MAX_SET, H_CAP = 50, 10_000


def _words(text: str | None) -> set[str]:
    return {w for w in tokens(text) if w.isalpha() and len(w) >= 5 and w not in STOPWORDS}


def _q(values: list[int]) -> str:
    if not values:
        return "none"
    v = sorted(values)
    pick = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return f"min {v[0]}, 25% {pick(.25)}, median {pick(.5)}, 75% {pick(.75)}, max {v[-1]}"


def h1(db) -> list[str]:
    holders = defaultdict(set)
    for name, uid in db.execute("SELECT name, user_id FROM badges WHERE user_id IS NOT NULL"):
        holders[name].add(uid)
    years_of = defaultdict(list)
    for uid, y in db.execute("SELECT owner_user_id, substr(creation_date,1,4) FROM posts "
                             "WHERE post_type_id = 1 AND owner_user_id IS NOT NULL"):
        years_of[uid].append(y)
    elig, counts, over_cap, too_big = 0, [], 0, 0
    for name, users in holders.items():
        if len(users) > H_CAP:
            too_big += 1
            continue
        per = Counter(y for u in users for y in years_of.get(u, ()))
        if sum(per.values()) > H_CAP:
            over_cap += 1
            continue
        if per:
            elig += 1
            counts += list(per.values())
    return [f"H1: {len(holders)} badges; {too_big} have more than {H_CAP} holders and {over_cap} have more than "
            f"{H_CAP} questions among their holders (the cap would be hit); {elig} badges give at least one (badge, year) count.",
            f"    counts per (badge, year): {_q(counts)}; {sum(1 for c in counts if c <= MAX_SET)} of {len(counts)} are at most {MAX_SET} "
            "(a count may be larger, the intermediate set is what the cap limits)."]


def h2(db, use_body: bool) -> list[str]:
    tag_q = defaultdict(set)
    for pid, tag in db.execute("SELECT post_id, tag FROM post_tags"):
        tag_q[tag].add(pid)
    accepted = {r[0] for r in db.execute("SELECT id FROM posts WHERE post_type_id = 1 AND accepted_answer_id IS NOT NULL")}
    text = {pid: _words((t or "") + (" " + (b or "") if use_body else ""))
            for pid, t, b in db.execute("SELECT id, title, body FROM posts WHERE post_type_id = 1")}
    sizes, tags_ok, tags_n, accepted_big = [], 0, 0, 0
    for tag, qs in tag_q.items():
        if not 10 <= len(qs) <= 300:
            continue
        tags_n += 1
        acc = qs & accepted
        if len(acc) > MAX_SET:
            accepted_big += 1
        vocab = set().union(*(text.get(q, set()) for q in qs))
        hit = 0
        for w in vocab:
            with_w = {q for q in qs if w in text.get(q, ())}
            n = len(with_w | acc)
            sizes.append(n)
            hit += 1 <= n <= MAX_SET
        tags_ok += hit > 0
    return [f"H2: {tags_n} tags with 10 to 300 questions; in {accepted_big} of them the accepted-answer part alone exceeds {MAX_SET} "
            f"(no word can bring the union under {MAX_SET}); {tags_ok} tags have at least one word giving 1 to {MAX_SET} items.",
            f"    union sizes over (tag, word): {_q(sizes)}; {sum(1 for s in sizes if 1 <= s <= MAX_SET)} of {len(sizes)} within 1 to {MAX_SET}. "
            f"Text searched: {'title and body' if use_body else 'title only'}."]


def h3(db) -> list[str]:
    n_body = db.execute("SELECT count(*) FROM posts WHERE post_type_id = 2 AND body IS NOT NULL AND body <> ''").fetchone()[0]
    if not n_body:
        return ["H3: answers have no text in this oracle (built with --no-text); cannot be checked here."]
    rep = dict(db.execute("SELECT id, reputation FROM users WHERE reputation IS NOT NULL"))
    by_word_a = defaultdict(set)
    for owner, body in db.execute("SELECT owner_user_id, body FROM posts WHERE post_type_id = 2 AND owner_user_id IS NOT NULL"):
        for w in _words(body):
            by_word_a[w].add(owner)
    title_words = {pid: _words(t) for pid, t in db.execute("SELECT id, title FROM posts WHERE post_type_id = 1")}
    by_word_b = defaultdict(set)
    for owner, parent in db.execute("SELECT owner_user_id, parent_id FROM posts WHERE post_type_id = 2 AND owner_user_id IS NOT NULL"):
        for w in title_words.get(parent, ()):
            by_word_b[w].add(owner)
    words = [w for w, s in by_word_a.items() if 5 <= len(s) <= 5000]
    cuts = (10, 50, 101, 200, 500, 1000, 2000)
    sizes = [sum(1 for u in by_word_a[w] if rep.get(u, 0) < n) for w in words for n in cuts]
    jac = []
    for w in sorted(set(words) & set(by_word_b))[:200]:
        a, b = by_word_a[w], by_word_b[w]
        jac.append(len(a & b) / len(a | b))
    return [f"H3 (reading A, the answer contains the word): {len(words)} words with 5 to 5000 authors; result sizes over (word, NUMBER in {cuts}): {_q(sizes)}; "
            f"{sum(1 for s in sizes if 1 <= s <= MAX_SET)} of {len(sizes)} within 1 to {MAX_SET}.",
            f"    reading B (the answered question's title contains the word) against reading A on {len(jac)} words: median overlap "
            f"(Jaccard) of the author sets {median(jac):.2f}" if jac else "    reading B: no words in common to compare."]


def h4(db) -> list[str]:
    upv = dict(db.execute("SELECT id, up_votes FROM users WHERE up_votes IS NOT NULL"))
    askers = defaultdict(set)
    qowner = {}
    for pid, owner, y in db.execute("SELECT id, owner_user_id, substr(creation_date,1,4) FROM posts "
                                    "WHERE post_type_id = 1 AND owner_user_id IS NOT NULL"):
        askers[y].add(owner)
        qowner[pid] = (owner, y)
    recv = defaultdict(Counter)          # year -> user -> up-votes received on that year's questions
    for pid, n in db.execute("SELECT post_id, count(*) FROM votes WHERE vote_type_id = 2 GROUP BY post_id"):
        if pid in qowner:
            o, y = qowner[pid]
            recv[y][o] += n
    cuts = (5, 10, 50, 100, 500, 1000, 5000)
    sa, sb, jac, big = [], [], [], 0
    for y, users in askers.items():
        if len(users) > H_CAP:
            big += 1
        for n in cuts:
            a = {u for u in users if upv.get(u, 0) >= n}
            b = {u for u in users if recv[y][u] >= n}
            sa.append(len(a)), sb.append(len(b))
            if a | b:
                jac.append(len(a & b) / len(a | b))
    return [f"H4: {len(askers)} years; {big} have more than {H_CAP} askers (the cap would be hit).",
            f"    reading A (own up_votes field) result sizes over (year, NUMBER in {cuts}): {_q(sa)}; {sum(1 for s in sa if 1 <= s <= MAX_SET)} of {len(sa)} within 1 to {MAX_SET}.",
            f"    reading B (up-votes received on that year's questions): {_q(sb)}; {sum(1 for s in sb if 1 <= s <= MAX_SET)} of {len(sb)} within 1 to {MAX_SET}.",
            f"    the two readings give different sets: median overlap (Jaccard) {median(jac):.2f}" if jac else "    no overlap to compare."]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--body", action="store_true", help="H2 also searches the question body")
    ap.add_argument("--only", choices=["H1", "H2", "H3", "H4"])
    args = ap.parse_args(argv)
    db = sqlite3.connect(f"file:{args.oracle}?mode=ro", uri=True)
    try:
        for name, fn in (("H1", lambda: h1(db)), ("H2", lambda: h2(db, args.body)), ("H3", lambda: h3(db)), ("H4", lambda: h4(db))):
            if args.only in (None, name):
                print("\n".join(fn()))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
