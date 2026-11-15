"""The Reader Kit appendix reads only the oracle file and is deterministic."""
import sqlite3

from benchmark_v2.kit_appendix import _quantiles, _spread, render


def _db():
    db = sqlite3.connect(":memory:")
    db.executescript("""
    create table post_tags(post_id int, tag text);
    create table badges(name text, user_id int);
    create table users(id int, reputation int);
    create table posts(id int, post_type_id int, score int, creation_date text);
    """)
    for t in range(8):
        for q in range(60 + t):
            db.execute("insert into post_tags values(?,?)", (q, f"tag{t}"))
    for i in range(200):
        db.execute("insert into badges values(?,?)", (f"b{i % 3}", i))
    for i in range(100):
        db.execute("insert into users values(?,?)", (i, i * 10))
        db.execute("insert into posts values(?,?,?,?)", (i, 1, i % 7, f"20{10 + i % 5}-06-01"))
    return db


def test_spread_is_even_and_keeps_ends():
    spread = _spread(list(range(10)), 3)
    assert spread[0] == 0 and spread[-1] == 9 and len(spread) == 3
    assert _spread([1, 2], 5) == [1, 2]


def test_quantiles_of_empty_is_empty():
    assert _quantiles([]) == []


def test_render_is_deterministic_and_lists_values():
    a, b = render(_db()), render(_db())
    assert a == b
    assert "Tags with 51 to 200 questions (8 tags in this band)" in a
    assert '"b0"' in a and "User reputation" in a and "between 2010 and 2014" in a
