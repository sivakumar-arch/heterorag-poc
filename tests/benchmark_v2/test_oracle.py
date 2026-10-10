"""Tests for the independent ground-truth oracle (benchmark_v2/oracle.py)."""
import sqlite3
from pathlib import Path

from benchmark_v2 import oracle

HEADER = '<?xml version="1.0" encoding="utf-8"?>\n<{root}>\n'


def _write(path: Path, root: str, rows: list[str]) -> None:
    path.write_text(HEADER.format(root=root) + "\n".join(rows) + f"\n</{root}>\n", encoding="utf-8")


def make_dump(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    _write(d / "Users.xml", "users", [
        '  <row Id="-1" Reputation="1" CreationDate="2010-01-01T00:00:00.000" DisplayName="Community" Views="0" UpVotes="0" DownVotes="0" />',
        '  <row Id="2" Reputation="1500" CreationDate="2011-02-03T04:05:06.000" DisplayName="Ann" AboutMe="I use &lt;b&gt;R&lt;/b&gt; &amp; Python" Views="10" UpVotes="5" DownVotes="1" AccountId="77" />',
        '  <row Id="3" Reputation="20" CreationDate="2012-02-03T04:05:06.000" DisplayName="Bo" Views="1" UpVotes="0" DownVotes="0" />',
    ])
    _write(d / "Posts.xml", "posts", [
        '  <row Id="1" PostTypeId="1" AcceptedAnswerId="2" CreationDate="2020-01-01T00:00:00.000" Score="7" ViewCount="100" Body="&lt;p&gt;What is PCA?&lt;/p&gt;" OwnerUserId="2" Title="What is PCA?" Tags="|pca|python|" AnswerCount="1" CommentCount="1" />',
        '  <row Id="2" PostTypeId="2" ParentId="1" CreationDate="2020-01-02T00:00:00.000" Score="5" Body="Rotation of axes." OwnerUserId="3" CommentCount="0" />',
        '  <row Id="3" PostTypeId="1" CreationDate="2020-02-01T00:00:00.000" Score="0" ViewCount="5" Body="old tags" OwnerUserId="3" Title="Legacy" Tags="&lt;r&gt;&lt;regression&gt;" AnswerCount="0" CommentCount="0" />',
        '  <row Id="4" PostTypeId="5" CreationDate="2020-03-01T00:00:00.000" Score="0" Body="wiki" OwnerUserId="2" />',
        '  <row Id="5" PostTypeId="1" CreationDate="2020-04-01T00:00:00.000" Score="1" ViewCount="1" Body="no tags" OwnerUserId="2" Title="Untagged" AnswerCount="0" CommentCount="0" />',
    ])
    _write(d / "Votes.xml", "votes", [
        '  <row Id="1" PostId="1" VoteTypeId="2" CreationDate="2020-01-03T00:00:00.000" />',
        '  <row Id="2" PostId="1" VoteTypeId="8" UserId="2" CreationDate="2020-01-04T00:00:00.000" BountyAmount="50" />',
    ])
    _write(d / "Badges.xml", "badges", [
        '  <row Id="1" UserId="2" Name="Teacher" Date="2020-01-05T00:00:00.000" Class="3" TagBased="False" />',
    ])
    _write(d / "Comments.xml", "comments", [
        '  <row Id="1" PostId="1" Score="2" Text="Good &amp; clear" CreationDate="2020-01-06T00:00:00.000" UserId="3" />',
    ])
    _write(d / "PostLinks.xml", "postlinks", [
        '  <row Id="1" CreationDate="2020-05-01T00:00:00.000" PostId="3" RelatedPostId="1" LinkTypeId="1" />',
        '  <row Id="2" CreationDate="2020-05-02T00:00:00.000" PostId="5" RelatedPostId="1" LinkTypeId="3" />',
    ])
    _write(d / "Tags.xml", "tags", [
        '  <row Id="1" TagName="pca" Count="1" ExcerptPostId="4" WikiPostId="4" />',
        '  <row Id="2" TagName="python" Count="1" />',
    ])
    return d


def _build(tmp_path: Path, **kw):
    dump = make_dump(tmp_path / "dump")
    db = tmp_path / "oracle.sqlite"
    counts = oracle.build(dump, db, log=lambda s: None, **kw)
    return dump, db, counts


def test_split_tags_both_encodings_and_blanks():
    assert oracle.split_tags("|pca|python|") == ["pca", "python"]
    assert oracle.split_tags("<r><regression>") == ["r", "regression"]
    assert oracle.split_tags("") == [] and oracle.split_tags(None) == []
    assert oracle.split_tags("|a||b|") == ["a", "b"]


def test_row_counts_match_the_fixture(tmp_path):
    _, _, counts = _build(tmp_path)
    assert counts["users"] == 3 and counts["posts"] == 5 and counts["votes"] == 2
    assert counts["badges"] == 1 and counts["comments"] == 1
    assert counts["post_links"] == 2 and counts["tags"] == 2
    assert counts["post_tags"] == 4          # pca, python, r, regression


def test_values_types_and_nulls(tmp_path):
    _, db, _ = _build(tmp_path)
    con = sqlite3.connect(db)
    u = con.execute("SELECT reputation, display_name, about_me FROM users WHERE id=2").fetchone()
    assert u == (1500, "Ann", "I use <b>R</b> & Python")          # XML entities decoded
    assert con.execute("SELECT accepted_answer_id, parent_id FROM posts WHERE id=1").fetchone() == (2, None)
    assert con.execute("SELECT parent_id FROM posts WHERE id=2").fetchone() == (1,)
    assert con.execute("SELECT typeof(score), typeof(creation_date) FROM posts WHERE id=1").fetchone() == ("integer", "text")
    assert con.execute("SELECT user_id, bounty_amount FROM votes WHERE id=1").fetchone() == (None, None)
    assert con.execute("SELECT bounty_amount FROM votes WHERE id=2").fetchone() == (50,)


def test_tags_normalised_from_both_encodings_and_untagged_post(tmp_path):
    _, db, _ = _build(tmp_path)
    con = sqlite3.connect(db)
    assert sorted(r[0] for r in con.execute("SELECT tag FROM post_tags WHERE post_id=1")) == ["pca", "python"]
    assert sorted(r[0] for r in con.execute("SELECT tag FROM post_tags WHERE post_id=3")) == ["r", "regression"]
    assert con.execute("SELECT count(*) FROM post_tags WHERE post_id=5").fetchone()[0] == 0


def test_verify_passes_and_detects_a_missing_row(tmp_path):
    dump, db, _ = _build(tmp_path)
    assert oracle.verify(dump, db, log=lambda s: None) is True
    con = sqlite3.connect(db)
    con.execute("DELETE FROM votes WHERE id=2"); con.commit(); con.close()
    assert oracle.verify(dump, db, log=lambda s: None) is False


def test_raw_count_is_independent_of_the_xml_parser(tmp_path):
    dump = make_dump(tmp_path / "dump")
    assert oracle.raw_row_count(dump / "Posts.xml") == 5
    assert oracle.raw_row_count(dump / "Tags.xml") == 2


def test_no_text_option_drops_bodies_but_keeps_structure(tmp_path):
    _, db, _ = _build(tmp_path, with_text=False)
    con = sqlite3.connect(db)
    assert con.execute("SELECT body, title FROM posts WHERE id=1").fetchone()[0] is None
    assert con.execute("SELECT owner_user_id FROM posts WHERE id=1").fetchone() == (2,)


def test_metadata_records_file_hashes_and_counts(tmp_path):
    import json
    dump, db, _ = _build(tmp_path)
    con = sqlite3.connect(db)
    meta = dict(con.execute("SELECT key, value FROM oracle_meta"))
    files = json.loads(meta["files"])
    assert files["Posts.xml"]["sha256"] == oracle.sha256_of(dump / "Posts.xml")
    assert json.loads(meta["row_counts"])["posts"] == 5


def test_refuses_to_overwrite_and_reports_missing_files(tmp_path):
    dump, db, _ = _build(tmp_path)
    for fn in (lambda: oracle.build(dump, db, log=lambda s: None),):
        try:
            fn()
        except FileExistsError:
            pass
        else:
            raise AssertionError("expected FileExistsError")
    (dump / "Votes.xml").unlink()
    try:
        oracle.build(dump, tmp_path / "other.sqlite", log=lambda s: None)
    except FileNotFoundError as e:
        assert "Votes.xml" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_oracle_does_not_import_the_system_under_test():
    import ast
    tree = ast.parse(Path(oracle.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"heterorag", "psycopg2", "neo4j", "elasticsearch"})


# ------------------------------------------------ profile

def test_profile_on_the_fixture(tmp_path):
    from benchmark_v2 import profile_oracle
    _, db, _ = _build(tmp_path)
    p = profile_oracle.profile(db)
    assert p["posts_by_type"] == {1: 3, 2: 1, 5: 1}
    assert p["questions"] == 3 and p["questions_with_accepted_answer"] == 1
    assert p["questions_with_no_answer"] == 2
    assert p["links_by_type"] == {1: 1, 3: 1}
    assert p["posts_with_a_link"] == 3                 # posts 1, 3, 5
    assert dict(p["top_tags"])["pca"] == 1 and p["tag_count"] == 4
    assert p["users_with_about_me"] == 1
    text = profile_oracle.render(p)
    assert "== LINKS" in text and "bin 1-9" in text
