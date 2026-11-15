"""Appendix of real example values for the Reader Kit (Job 1), printed as Markdown.

Reads only the oracle SQLite file (never a system result file). It lists example tag names,
badge names and typical ranges so that an outside reader can write shapes whose blanks can be
filled. Deterministic: no randomness, the examples are spread evenly over each size band.

    python -m benchmark_v2.kit_appendix --oracle PATH_TO_ORACLE.sqlite > kit_appendix.md
"""
from __future__ import annotations

import argparse
import sqlite3

TAG_BANDS = ((10, 50), (51, 200), (201, 1000), (1001, 10 ** 9))


def _spread(items: list, n: int) -> list:
    """n items spread evenly over the list (first and last included when n >= 2)."""
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def _quantiles(values: list[int], qs=(0.25, 0.5, 0.75, 0.9, 0.99)) -> list[tuple[float, int]]:
    values = sorted(values)
    if not values:
        return []
    return [(q, values[min(len(values) - 1, int(q * len(values)))]) for q in qs]


def render(db: sqlite3.Connection) -> str:
    out: list[str] = ["# Appendix: example values from the data (for Job 1)", "",
                      "These are real values from the community copy, to help you write shapes whose blanks "
                      "can be filled. They are not the values the study will use, and they say nothing about "
                      "the system. Do not hand in a shape that only works for one of these values.", ""]

    tags = db.execute("SELECT tag, count(*) FROM post_tags GROUP BY tag").fetchall()
    out += ["## Tags (name, number of questions carrying it)", ""]
    for lo, hi in TAG_BANDS:
        band = sorted((t for t in tags if lo <= t[1] <= hi), key=lambda t: (t[1], t[0]))
        label = f"{lo} to {hi}" if hi < 10 ** 9 else f"{lo} or more"
        if not band:
            continue
        picks = _spread(band, 6)
        out.append(f"* Tags with {label} questions ({len(band)} tags in this band): "
                   + ", ".join(f"\"{t}\" ({n})" for t, n in picks))
    out.append("")

    badges = db.execute("SELECT name, count(DISTINCT user_id) FROM badges GROUP BY name").fetchall()
    badges.sort(key=lambda b: (b[1], b[0]))
    out += ["## Badge names (name, number of users who hold it)", ""]
    for lo, hi in ((1, 10), (11, 100), (101, 1000), (1001, 10 ** 9)):
        band = [b for b in badges if lo <= b[1] <= hi]
        label = f"{lo} to {hi}" if hi < 10 ** 9 else f"{lo} or more"
        if not band:
            continue
        out.append(f"* Held by {label} users ({len(band)} badges): "
                   + ", ".join(f"\"{n}\" ({c})" for n, c in _spread(band, 6)))
    out.append("")

    rep = [r[0] for r in db.execute("SELECT reputation FROM users WHERE reputation IS NOT NULL")]
    score = [r[0] for r in db.execute("SELECT score FROM posts WHERE post_type_id = 1 AND score IS NOT NULL")]
    yrs = db.execute("SELECT min(substr(creation_date,1,4)), max(substr(creation_date,1,4)) FROM posts").fetchone()
    out += ["## Typical ranges", "",
            "* User reputation (share of users at or below the value): "
            + ", ".join(f"{int(q * 100)}%: {v}" for q, v in _quantiles(rep)),
            "* Question score (share of questions at or below the value): "
            + ", ".join(f"{int(q * 100)}%: {v}" for q, v in _quantiles(score)),
            f"* Posts were created between {yrs[0]} and {yrs[1]}.",
            "* A shape that combines a rare tag with a high reputation threshold tends to give a short answer; "
            "a common tag with a low threshold gives a very long one. We pick values to land between 1 and about 50 items.",
            ""]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", required=True)
    args = ap.parse_args(argv)
    db = sqlite3.connect(f"file:{args.oracle}?mode=ro", uri=True)
    try:
        print(render(db))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
