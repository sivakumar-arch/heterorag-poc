"""The pilot check reads only the oracle file and reports sizes and reading differences."""
import sqlite3

from benchmark_v2.shape_feasibility import h1, h2, h3, h4, h4_grid, s1, s2


def _db(with_body=True):
    db = sqlite3.connect(":memory:")
    db.executescript("""
    create table posts(id int, post_type_id int, accepted_answer_id int, parent_id int, owner_user_id int,
                       creation_date text, title text, body text);
    create table post_tags(post_id int, tag text);
    create table badges(name text, user_id int);
    create table users(id int, reputation int, up_votes int);
    create table votes(post_id int, vote_type_id int);
    """)
    for u in range(1, 21):
        db.execute("insert into users values(?,?,?)", (u, u * 10, u * 3))
        db.execute("insert into badges values(?,?)", ("Teacher" if u % 2 else "Editor", u))
    for q in range(1, 41):
        db.execute("insert into posts values(?,?,?,?,?,?,?,?)",
                   (q, 1, q + 100 if q % 2 else None, None, 1 + q % 20, f"20{10 + q % 3}-05-01",
                    "estimating variance components" if q % 5 else "bootstrapping residuals", None))
        db.execute("insert into post_tags values(?,?)", (q, "variance" if q <= 30 else "bootstrap"))
        db.execute("insert into votes values(?,2)", (q,))
        db.execute("insert into posts values(?,?,?,?,?,?,?,?)",
                   (q + 100, 2, None, q, 1 + (q * 7) % 20, "2012-01-01", None,
                    "use bootstrapping resampling here" if with_body else None))
    return db


def test_all_four_shapes_report():
    db = _db()
    assert "H1" in h1(db)[0] and "H2" in h2(db, False)[0]
    assert "reading A" in h3(db)[0] and "reading B" in h3(db)[1]
    out = h4(db)
    assert out[0].startswith("H4") and "different sets" in out[-1]


def test_h3_without_text_says_so():
    assert "no text" in h3(_db(with_body=False))[0]


def test_h2_flags_tags_whose_accepted_part_is_too_large():
    db = sqlite3.connect(":memory:")
    db.executescript("create table posts(id int, post_type_id int, accepted_answer_id int, title text, body text);"
                     "create table post_tags(post_id int, tag text);")
    for q in range(1, 121):
        db.execute("insert into posts values(?,?,?,?,?)", (q, 1, q + 1000, "something general", None))
        db.execute("insert into post_tags values(?,?)", (q, "big"))
    assert "accepted-answer part alone exceeds 50" in h2(db, False)[0]


def _db6():
    db = _db()
    db.executescript("create table post_links(post_id int, related_post_id int, link_type_id int);")
    for q in range(2, 12):
        db.execute("insert into post_links values(?,?,3)", (q, 1))      # q is a duplicate of question 1
    db.execute("alter table posts add column score int")
    db.execute("update posts set score = id")
    return db


def test_h4_grid_gives_usable_pairs_by_construction():
    out = h4_grid(_db())
    assert out[0].startswith("H4 finer grid") and " 0 (year, NUMBER)" not in out[0]


def test_s1_counts_pairs_for_duplicates_and_links():
    out = s1(_db6())
    assert out[0].startswith("S1 (DUPLICATE_OF): 10 edges") and "5 (tag, NUMBER) pairs" in out[0]
    assert out[1].startswith("S1 (LINKED_TO): 0 edges")


def test_s2_reports_difference_pairs():
    out = s2(_db())
    assert out[0].startswith("S2:") and "non-trivial" in out[0]
