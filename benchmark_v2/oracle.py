"""
benchmark_v2/oracle.py
======================
The ground-truth oracle for Benchmark v2.

The oracle is the *virtual union* of the data: every entity, attribute,
relationship and text in the Stack Exchange dump, held in one canonical SQLite
database built directly from the XML files. Gold answers are computed against it.

Independence is the point. This module does not import anything from the
``heterorag`` package, does not read PostgreSQL, Neo4j or Elasticsearch, and has
its own XML reading and tag parsing. A bug in a loader, a Flyway view or a
service descriptor therefore cannot make the oracle agree with a wrong system.

Usage
-----
    python -m benchmark_v2.oracle build  --data-dir data/stats --db data/oracle_stats.sqlite
    python -m benchmark_v2.oracle verify --data-dir data/stats --db data/oracle_stats.sqlite

``build`` streams each XML file (constant memory) and writes one table per file plus
``post_tags``. ``verify`` compares every table's row count with an independent raw
line count of the XML file (a count of ``<row`` lines that uses no XML parser).

PostHistory.xml is not part of the virtual union and is not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

BATCH = 50_000

# ----------------------------------------------------------------------------
# Table specifications: (xml attribute, column name, sqlite type)
# ----------------------------------------------------------------------------

_I, _T = "INTEGER", "TEXT"


@dataclass(frozen=True)
class TableSpec:
    table: str
    xml_file: str
    columns: tuple[tuple[str, str, str], ...]   # (xml attr, column, type)
    primary_key: str = "id"


SPECS: tuple[TableSpec, ...] = (
    TableSpec("users", "Users.xml", (
        ("Id", "id", _I), ("Reputation", "reputation", _I), ("CreationDate", "creation_date", _T),
        ("DisplayName", "display_name", _T), ("LastAccessDate", "last_access_date", _T),
        ("WebsiteUrl", "website_url", _T), ("Location", "location", _T),
        ("AboutMe", "about_me", _T), ("Views", "views", _I), ("UpVotes", "up_votes", _I),
        ("DownVotes", "down_votes", _I), ("AccountId", "account_id", _I))),
    TableSpec("posts", "Posts.xml", (
        ("Id", "id", _I), ("PostTypeId", "post_type_id", _I),
        ("AcceptedAnswerId", "accepted_answer_id", _I), ("ParentId", "parent_id", _I),
        ("CreationDate", "creation_date", _T), ("Score", "score", _I),
        ("ViewCount", "view_count", _I), ("Body", "body", _T),
        ("OwnerUserId", "owner_user_id", _I), ("OwnerDisplayName", "owner_display_name", _T),
        ("LastEditorUserId", "last_editor_user_id", _I), ("LastEditDate", "last_edit_date", _T),
        ("LastActivityDate", "last_activity_date", _T), ("Title", "title", _T),
        ("Tags", "tags_raw", _T), ("AnswerCount", "answer_count", _I),
        ("CommentCount", "comment_count", _I), ("FavoriteCount", "favorite_count", _I),
        ("ClosedDate", "closed_date", _T), ("CommunityOwnedDate", "community_owned_date", _T),
        ("ContentLicense", "content_license", _T))),
    TableSpec("votes", "Votes.xml", (
        ("Id", "id", _I), ("PostId", "post_id", _I), ("VoteTypeId", "vote_type_id", _I),
        ("UserId", "user_id", _I), ("CreationDate", "creation_date", _T),
        ("BountyAmount", "bounty_amount", _I))),
    TableSpec("badges", "Badges.xml", (
        ("Id", "id", _I), ("UserId", "user_id", _I), ("Name", "name", _T),
        ("Date", "date", _T), ("Class", "class", _I), ("TagBased", "tag_based", _T))),
    TableSpec("comments", "Comments.xml", (
        ("Id", "id", _I), ("PostId", "post_id", _I), ("Score", "score", _I),
        ("Text", "text", _T), ("CreationDate", "creation_date", _T), ("UserId", "user_id", _I),
        ("UserDisplayName", "user_display_name", _T))),
    TableSpec("post_links", "PostLinks.xml", (
        ("Id", "id", _I), ("CreationDate", "creation_date", _T), ("PostId", "post_id", _I),
        ("RelatedPostId", "related_post_id", _I), ("LinkTypeId", "link_type_id", _I))),
    TableSpec("tags", "Tags.xml", (
        ("Id", "id", _I), ("TagName", "tag_name", _T), ("Count", "count", _I),
        ("ExcerptPostId", "excerpt_post_id", _I), ("WikiPostId", "wiki_post_id", _I))),
)

INDEXES: tuple[str, ...] = (
    "CREATE INDEX idx_posts_type   ON posts(post_type_id)",
    "CREATE INDEX idx_posts_owner  ON posts(owner_user_id)",
    "CREATE INDEX idx_posts_parent ON posts(parent_id)",
    "CREATE INDEX idx_post_tags_tag ON post_tags(tag)",
    "CREATE INDEX idx_post_tags_post ON post_tags(post_id)",
    "CREATE INDEX idx_votes_post   ON votes(post_id)",
    "CREATE INDEX idx_votes_user   ON votes(user_id)",
    "CREATE INDEX idx_badges_user  ON badges(user_id)",
    "CREATE INDEX idx_comments_post ON comments(post_id)",
    "CREATE INDEX idx_comments_user ON comments(user_id)",
    "CREATE INDEX idx_links_post   ON post_links(post_id)",
    "CREATE INDEX idx_links_rel    ON post_links(related_post_id)",
)

# ----------------------------------------------------------------------------
# Own helpers (deliberately not shared with the heterorag package)
# ----------------------------------------------------------------------------

_ANGLE = re.compile(r"<([^<>]+)>")


def split_tags(raw: str | None) -> list[str]:
    """Tag names from either dump encoding: '|a|b|' (2023 onward) or '<a><b>' (legacy)."""
    if not raw:
        return []
    raw = raw.strip()
    if "|" in raw:
        return [t for t in (p.strip() for p in raw.split("|")) if t]
    return [t.strip() for t in _ANGLE.findall(raw) if t.strip()]


def _convert(value: str | None, sql_type: str):
    if value is None:
        return None
    if sql_type == _I:
        try:
            return int(value)
        except ValueError:
            return None
    return value


def iter_rows(xml_path: Path) -> Iterator[dict[str, str]]:
    """Stream the attribute dict of every <row> element in constant memory."""
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event == "end" and elem.tag == "row":
            yield dict(elem.attrib)
            elem.clear()
            root.clear()


def raw_row_count(xml_path: Path) -> int:
    """Count '<row ' lines without an XML parser (independent check on the parser)."""
    marker = b"<row "
    n = 0
    with xml_path.open("rb") as f:
        for line in f:
            if line.lstrip().startswith(marker):
                n += 1
    return n


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------

def _create_table(con: sqlite3.Connection, spec: TableSpec) -> None:
    cols = ", ".join(
        f"{c} {t}{' PRIMARY KEY' if c == spec.primary_key else ''}" for _, c, t in spec.columns)
    con.execute(f"CREATE TABLE {spec.table} ({cols})")


def _load_table(con: sqlite3.Connection, spec: TableSpec, xml_path: Path,
                with_text: bool, log: Callable[[str], None]) -> int:
    text_cols = {"body", "about_me", "text"}
    cols = [c for _, c, _ in spec.columns]
    placeholders = ",".join("?" for _ in cols)
    insert = f"INSERT INTO {spec.table} ({','.join(cols)}) VALUES ({placeholders})"
    tag_insert = "INSERT INTO post_tags (post_id, tag) VALUES (?, ?)"
    batch: list[tuple] = []
    tag_batch: list[tuple] = []
    n = 0
    for attrs in iter_rows(xml_path):
        row = []
        for attr, col, typ in spec.columns:
            if col in text_cols and not with_text:
                row.append(None)
            else:
                row.append(_convert(attrs.get(attr), typ))
        batch.append(tuple(row))
        if spec.table == "posts":
            pid = _convert(attrs.get("Id"), _I)
            for tag in split_tags(attrs.get("Tags")):
                tag_batch.append((pid, tag))
        n += 1
        if len(batch) >= BATCH:
            con.executemany(insert, batch); batch.clear()
            if tag_batch:
                con.executemany(tag_insert, tag_batch); tag_batch.clear()
            if n % (BATCH * 10) == 0:
                log(f"  {spec.table}: {n:,} rows")
    if batch:
        con.executemany(insert, batch)
    if tag_batch:
        con.executemany(tag_insert, tag_batch)
    return n


def build(data_dir: Path, db_path: Path, with_text: bool = True, hash_files: bool = True,
          log: Callable[[str], None] = print) -> dict[str, int]:
    """Build the oracle database. Refuses to overwrite an existing file."""
    data_dir, db_path = Path(data_dir), Path(db_path)
    if db_path.exists():
        raise FileExistsError(f"{db_path} exists; delete it to rebuild")
    missing = [s.xml_file for s in SPECS if not (data_dir / s.xml_file).exists()]
    if missing:
        raise FileNotFoundError(f"missing in {data_dir}: {', '.join(missing)}")

    t0 = time.time()
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    counts: dict[str, int] = {}
    try:
        for spec in SPECS:
            _create_table(con, spec)
        con.execute("CREATE TABLE post_tags (post_id INTEGER NOT NULL, tag TEXT NOT NULL)")
        con.execute("CREATE TABLE oracle_meta (key TEXT PRIMARY KEY, value TEXT)")
        for spec in SPECS:
            log(f"loading {spec.xml_file} ...")
            counts[spec.table] = _load_table(con, spec, data_dir / spec.xml_file, with_text, log)
            con.commit()
        counts["post_tags"] = con.execute("SELECT count(*) FROM post_tags").fetchone()[0]
        log("creating indexes ...")
        for stmt in INDEXES:
            con.execute(stmt)
        meta = {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "with_text": str(with_text),
            "row_counts": json.dumps(counts, sort_keys=True),
            "files": json.dumps({
                s.xml_file: {
                    "bytes": (data_dir / s.xml_file).stat().st_size,
                    "sha256": sha256_of(data_dir / s.xml_file) if hash_files else None,
                } for s in SPECS}, sort_keys=True),
        }
        con.executemany("INSERT INTO oracle_meta VALUES (?, ?)", meta.items())
        con.commit()
    finally:
        con.close()
    log(f"built {db_path} in {time.time() - t0:.0f}s: {counts}")
    return counts


# ----------------------------------------------------------------------------
# Verify
# ----------------------------------------------------------------------------

def verify(data_dir: Path, db_path: Path, log: Callable[[str], None] = print) -> bool:
    """Compare every table's row count with an independent raw line count of its XML."""
    ok = True
    con = sqlite3.connect(str(db_path))
    try:
        for spec in SPECS:
            raw = raw_row_count(Path(data_dir) / spec.xml_file)
            got = con.execute(f"SELECT count(*) FROM {spec.table}").fetchone()[0]
            status = "OK " if raw == got else "MISMATCH"
            ok &= raw == got
            log(f"{status} {spec.table:<11} xml_lines={raw:>10,}  oracle_rows={got:>10,}")
        # tag rows must be consistent with the raw tag strings
        posts_with_tags = con.execute(
            "SELECT count(*) FROM posts WHERE tags_raw IS NOT NULL AND tags_raw <> ''").fetchone()[0]
        tagged_posts = con.execute("SELECT count(DISTINCT post_id) FROM post_tags").fetchone()[0]
        status = "OK " if posts_with_tags == tagged_posts else "MISMATCH"
        ok &= posts_with_tags == tagged_posts
        log(f"{status} post_tags    posts_with_tag_string={posts_with_tags:,}  posts_in_post_tags={tagged_posts:,}")
    finally:
        con.close()
    log("VERIFY PASSED" if ok else "VERIFY FAILED")
    return ok


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the oracle database from the XML files")
    b.add_argument("--data-dir", required=True, type=Path)
    b.add_argument("--db", required=True, type=Path)
    b.add_argument("--no-text", action="store_true", help="skip Body/Text/AboutMe (smaller database)")
    b.add_argument("--no-hash", action="store_true", help="skip SHA-256 of the XML files")
    v = sub.add_parser("verify", help="compare oracle row counts with raw XML line counts")
    v.add_argument("--data-dir", required=True, type=Path)
    v.add_argument("--db", required=True, type=Path)
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "build":
        build(args.data_dir, args.db, with_text=not args.no_text, hash_files=not args.no_hash)
        return 0
    return 0 if verify(args.data_dir, args.db) else 1


if __name__ == "__main__":
    sys.exit(main())
