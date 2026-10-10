"""
benchmark_v2/profile_oracle.py
==============================
Describe the data in the oracle so question templates and their parameter strata
(selectivity, intermediate cardinality, link-chain depth) are chosen from facts,
not guesses.

    python -m benchmark_v2.profile_oracle --db data/oracle_stats.sqlite [--json out.json]

Read-only. Runs in seconds to a couple of minutes on the full `stats` oracle.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterable


def _quantiles(values: list[int], qs=(0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)) -> dict[str, int]:
    if not values:
        return {}
    v = sorted(values)
    return {f"p{int(q * 100)}": v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))] for q in qs}


def _bin(count: int, edges: tuple[int, ...]) -> str:
    lo = 1
    for e in edges:
        if count < e:
            return f"{lo}-{e - 1}"
        lo = e
    return f"{lo}+"


def profile(db_path: Path, seed_sample: int = 500) -> dict:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    one = lambda sql, *a: con.execute(sql, a).fetchone()[0]
    out: dict = {}

    # --- posts ---------------------------------------------------------------
    out["posts_by_type"] = dict(con.execute(
        "SELECT post_type_id, count(*) FROM posts GROUP BY 1 ORDER BY 1"))
    q = one("SELECT count(*) FROM posts WHERE post_type_id=1")
    out["questions"] = q
    out["questions_with_accepted_answer"] = one(
        "SELECT count(*) FROM posts WHERE post_type_id=1 AND accepted_answer_id IS NOT NULL")
    out["questions_with_no_answer"] = one(
        "SELECT count(*) FROM posts WHERE post_type_id=1 AND answer_count=0")
    out["answer_count_quantiles"] = _quantiles(
        [r[0] for r in con.execute("SELECT answer_count FROM posts WHERE post_type_id=1")])
    out["question_score_quantiles"] = _quantiles(
        [r[0] for r in con.execute("SELECT score FROM posts WHERE post_type_id=1")])
    out["questions_without_owner"] = one(
        "SELECT count(*) FROM posts WHERE post_type_id=1 AND owner_user_id IS NULL")
    out["questions_with_owner_not_in_users"] = one(
        "SELECT count(*) FROM posts p WHERE post_type_id=1 AND owner_user_id IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM users u WHERE u.id=p.owner_user_id)")

    # --- tags: selectivity strata and bind-set sizes -------------------------------
    edges = (10, 50, 200, 1000)
    tag_counts = con.execute(
        "SELECT t.tag, count(*) c FROM post_tags t JOIN posts p ON p.id=t.post_id "
        "WHERE p.post_type_id=1 GROUP BY t.tag ORDER BY c DESC, t.tag").fetchall()
    out["tag_count"] = len(tag_counts)
    out["tags_per_question_quantiles"] = _quantiles([r[0] for r in con.execute(
        "SELECT count(*) FROM post_tags GROUP BY post_id")])
    bins: dict[str, list] = {}
    for tag, c in tag_counts:
        bins.setdefault(_bin(c, edges), []).append((tag, c))
    out["tags_by_question_count_bin"] = {k: len(v) for k, v in bins.items()}
    out["top_tags"] = tag_counts[:15]
    examples = {}
    for k, v in bins.items():
        tag, c = v[len(v) // 2]                       # median tag of the bin, deterministic
        askers = one("SELECT count(DISTINCT p.owner_user_id) FROM post_tags t JOIN posts p "
                     "ON p.id=t.post_id WHERE t.tag=? AND p.post_type_id=1", tag)
        answerers = one("SELECT count(DISTINCT a.owner_user_id) FROM post_tags t JOIN posts a "
                        "ON a.parent_id=t.post_id WHERE t.tag=? AND a.post_type_id=2", tag)
        examples[k] = {"tag": tag, "questions": c, "distinct_askers": askers,
                       "distinct_answerers": answerers}
    out["bind_set_examples_by_bin"] = examples

    # --- links: degree and two-hop neighbourhoods ----------------------------------
    out["links_by_type"] = dict(con.execute(
        "SELECT link_type_id, count(*) FROM post_links GROUP BY 1 ORDER BY 1"))
    adj: dict[int, set[int]] = {}
    for a, b in con.execute("SELECT post_id, related_post_id FROM post_links"):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    deg = [len(v) for v in adj.values()]
    out["posts_with_a_link"] = len(adj)
    out["link_degree_quantiles"] = _quantiles(deg)
    seeds = sorted(adj)[:: max(1, len(adj) // seed_sample)][:seed_sample]
    sizes = []
    for s in seeds:
        hop1 = adj[s]
        hop2 = set().union(*(adj.get(n, set()) for n in hop1)) - hop1 - {s}
        sizes.append(len(hop1 | hop2))
    out["two_hop_neighbourhood_sample_size"] = len(seeds)
    out["two_hop_neighbourhood_quantiles"] = _quantiles(sizes)
    out["two_hop_bins"] = {
        "1-5": sum(1 for x in sizes if 1 <= x <= 5),
        "6-20": sum(1 for x in sizes if 6 <= x <= 20),
        "21+": sum(1 for x in sizes if x >= 21),
    }

    # --- users ---------------------------------------------------------------------
    out["users"] = one("SELECT count(*) FROM users")
    out["reputation_quantiles"] = _quantiles([r[0] for r in con.execute(
        "SELECT reputation FROM users WHERE reputation IS NOT NULL")])
    out["users_with_question"] = one(
        "SELECT count(DISTINCT owner_user_id) FROM posts WHERE post_type_id=1")
    out["users_with_answer"] = one(
        "SELECT count(DISTINCT owner_user_id) FROM posts WHERE post_type_id=2")
    out["questions_per_asker_quantiles"] = _quantiles([r[0] for r in con.execute(
        "SELECT count(*) FROM posts WHERE post_type_id=1 AND owner_user_id IS NOT NULL "
        "GROUP BY owner_user_id")])
    out["badges_per_user_quantiles"] = _quantiles([r[0] for r in con.execute(
        "SELECT count(*) FROM badges GROUP BY user_id")])
    out["users_with_about_me"] = one(
        "SELECT count(*) FROM users WHERE about_me IS NOT NULL AND about_me <> ''")

    # --- text ----------------------------------------------------------------------
    out["posts_with_body"] = one("SELECT count(*) FROM posts WHERE body IS NOT NULL AND body <> ''")
    out["comments"] = one("SELECT count(*) FROM comments")
    out["tags_with_wiki_excerpt"] = one("SELECT count(*) FROM tags WHERE excerpt_post_id IS NOT NULL")
    con.close()
    return out


def render(p: dict) -> str:
    L: list[str] = []
    add = L.append

    def kv(title, d):
        add(f"{title}: " + ", ".join(f"{k}={v}" for k, v in d.items()))

    add("== POSTS")
    kv("posts_by_type", p["posts_by_type"])
    for k in ("questions", "questions_with_accepted_answer", "questions_with_no_answer",
              "questions_without_owner", "questions_with_owner_not_in_users"):
        add(f"{k}: {p[k]:,}")
    kv("answer_count_quantiles", p["answer_count_quantiles"])
    kv("question_score_quantiles", p["question_score_quantiles"])
    add("== TAGS (questions only)")
    add(f"distinct_tags: {p['tag_count']:,}")
    kv("tags_per_question_quantiles", p["tags_per_question_quantiles"])
    kv("tags_by_question_count_bin", p["tags_by_question_count_bin"])
    add("top_tags: " + ", ".join(f"{t}={c}" for t, c in p["top_tags"]))
    for b, e in p["bind_set_examples_by_bin"].items():
        add(f"bin {b}: tag={e['tag']} questions={e['questions']} "
            f"distinct_askers={e['distinct_askers']} distinct_answerers={e['distinct_answerers']}")
    add("== LINKS")
    kv("links_by_type", p["links_by_type"])
    add(f"posts_with_a_link: {p['posts_with_a_link']:,}")
    kv("link_degree_quantiles", p["link_degree_quantiles"])
    add(f"two_hop sample of {p['two_hop_neighbourhood_sample_size']} seeds")
    kv("two_hop_neighbourhood_quantiles", p["two_hop_neighbourhood_quantiles"])
    kv("two_hop_bins", p["two_hop_bins"])
    add("== USERS")
    for k in ("users", "users_with_question", "users_with_answer", "users_with_about_me"):
        add(f"{k}: {p[k]:,}")
    kv("reputation_quantiles", p["reputation_quantiles"])
    kv("questions_per_asker_quantiles", p["questions_per_asker_quantiles"])
    kv("badges_per_user_quantiles", p["badges_per_user_quantiles"])
    add("== TEXT")
    for k in ("posts_with_body", "comments", "tags_with_wiki_excerpt"):
        add(f"{k}: {p[k]:,}")
    return "\n".join(L)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None, help="also write the profile as JSON")
    args = ap.parse_args(list(argv) if argv is not None else None)
    p = profile(args.db)
    print(render(p))
    if args.json:
        args.json.write_text(json.dumps(p, indent=1, default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
