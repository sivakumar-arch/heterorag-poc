"""
Data-ownership rule (Benchmark v2 spec D2, D10, D11) enforced on the loader, the SQL
bootstrap, the service descriptors and the shared Elasticsearch mapping.

Attributes live in PostgreSQL, relationships in Neo4j, text in Elasticsearch; identifiers
are the only shared keys. These tests fail if any of the three definitions of a store
(bootstrap SQL, descriptor, fidelity checker) drifts from the others, or if a loader writes
a field the store does not own. Live-store behaviour is covered by benchmark_v2.fidelity.
"""
import io
import json
import re
from pathlib import Path

import pytest

from benchmark_v2 import fidelity
from tests.test_loader_tag_fixes import _load_loader

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ SQL bootstrap

def _bootstrap_columns() -> dict[str, set[str]]:
    sql = (ROOT / "sql" / "init" / "00_bootstrap.sql").read_text()
    tables: dict[str, set[str]] = {}
    for m in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", sql, re.S):
        cols = set()
        for line in m.group(2).splitlines():
            line = line.split("--")[0].strip()
            if not line:
                continue
            cols.add(line.split()[0])
        tables[m.group(1)] = cols
    return tables


def test_bootstrap_sql_matches_the_ownership_definition():
    found = _bootstrap_columns()
    found.pop("_load_progress")
    assert set(found) == set(fidelity.PG_COLUMNS)
    for table, cols in fidelity.PG_COLUMNS.items():
        assert found[table] == set(cols), table


def test_sql_descriptor_columns_are_a_subset_of_the_owned_columns():
    from heterorag.layer1.poc_descriptors import USER_ACTIVITY_DESCRIPTOR
    tables = {t.table_name: {c.name for c in t.columns}
              for t in USER_ACTIVITY_DESCRIPTOR.schema_spec.tables}
    assert set(tables) == set(fidelity.PG_COLUMNS)
    for table, cols in tables.items():
        assert cols <= fidelity.PG_COLUMNS[table], (table, cols - fidelity.PG_COLUMNS[table])


def test_graph_descriptor_nodes_carry_identifiers_only():
    from heterorag.layer1.poc_descriptors import KNOWLEDGE_GRAPH_DESCRIPTOR
    for n in KNOWLEDGE_GRAPH_DESCRIPTOR.schema_spec.node_types:
        assert set(n.properties) == fidelity.NEO4J_NODE_PROPERTIES[n.label], n.label
    edges = {e.edge_type: set(e.properties) for e in KNOWLEDGE_GRAPH_DESCRIPTOR.schema_spec.edge_types}
    assert set(edges) == set(fidelity.REL_TYPES)
    assert edges["CO_OCCURS_WITH"] == {"weight"}


def test_document_descriptor_fields_match_the_stored_fields():
    from heterorag.layer1.poc_descriptors import CONTENT_SERVICE_DESCRIPTOR
    for dt in CONTENT_SERVICE_DESCRIPTOR.schema_spec.doc_types:
        assert {f.field_name for f in dt.fields} == set(fidelity.ES_DOC_FIELDS[dt.doc_type]), dt.doc_type


def test_shared_es_mapping_holds_no_duplicated_fields():
    from heterorag.es_index import FORBIDDEN_FIELDS, INDEX_BODY
    props = set(INDEX_BODY["mappings"]["properties"])
    assert props == set(fidelity.ES_FIELDS)
    assert not props & FORBIDDEN_FIELDS


def test_setup_index_script_uses_the_shared_mapping():
    text = (ROOT / "ground-truth" / "document" / "setup_index.py").read_text()
    assert "from heterorag.es_index import" in text and "INDEX_BODY" in text
    assert '"parent_id"' not in text and '"score"' not in text


def test_flyway_views_and_graph_fixtures_are_gone():
    assert not list((ROOT / "sql").glob("migrations/*.sql"))
    assert not (ROOT / "graph" / "migrations" / "001_ground_truth_fixtures.cypher").exists()
    assert "flyway" not in (ROOT / "docker-compose.yml").read_text().lower().replace("# ", "")


# ------------------------------------------------------------------ PostgreSQL loader

class _Cur:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def cursor(self):
        return _Cur()

    def commit(self):
        pass


def _capture_inserts(loader, monkeypatch):
    captured = []
    monkeypatch.setattr(loader.psycopg2.extras, "execute_values",
                        lambda cur, sql, rows, template=None, **k: captured.append((sql, list(rows), template)),
                        raising=False)
    monkeypatch.setattr(loader, "is_complete", lambda *a, **k: False)
    monkeypatch.setattr(loader, "mark_complete", lambda *a, **k: None)
    return captured


_POSTS = """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" AcceptedAnswerId="2" CreationDate="2020-01-01T00:00:00.000" Score="7" ViewCount="100" Body="b" OwnerUserId="2" Title="T" Tags="|a|b|" AnswerCount="1" CommentCount="1" />
  <row Id="2" PostTypeId="2" ParentId="1" CreationDate="2020-01-02T00:00:00.000" Score="5" Body="a" OwnerUserId="3" CommentCount="0" />
</posts>
"""


def test_postgres_posts_hold_metrics_only(tmp_path, monkeypatch):
    loader = _load_loader()
    cap = _capture_inserts(loader, monkeypatch)
    (tmp_path / "Posts.xml").write_text(_POSTS, encoding="utf-8")
    loader.load_posts_pg(_Conn(), tmp_path, 1000, dry_run=False)
    sql, rows, template = cap[0]
    cols = {c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].replace("\n", " ").split(",")}
    assert cols == set(fidelity.PG_COLUMNS["posts"])
    assert len(rows[0]) == len(cols) == template.count("%s")
    assert rows[0][:6] == (1, 1, 7, 100, 1, 1)           # id, type, score, views, answers, comments


def test_postgres_users_and_tags_carry_no_text_or_references(tmp_path, monkeypatch):
    loader = _load_loader()
    cap = _capture_inserts(loader, monkeypatch)
    (tmp_path / "Users.xml").write_text(
        '<users><row Id="-1" Reputation="1"/><row Id="2" Reputation="9" DisplayName="Ann" AboutMe="hi" Location="X" Views="1" UpVotes="2" DownVotes="3" AccountId="7"/></users>',
        encoding="utf-8")
    loader.load_users_pg(_Conn(), tmp_path, 1000, dry_run=False)
    sql, rows, template = cap[0]
    assert "about_me" not in sql
    assert len(rows) == 1 and rows[0][0] == 2            # id -1 is not loaded
    assert len(rows[0]) == template.count("%s")
    assert "hi" not in rows[0]


def test_no_post_links_loader_remains():
    loader = _load_loader()
    assert not hasattr(loader, "load_post_links_pg")


# ------------------------------------------------------------------ Neo4j loader

class _Session:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **kw):
        self.log.append((query, kw))


class _Driver:
    def __init__(self):
        self.log = []

    def session(self):
        return _Session(self.log)


def test_neo4j_nodes_carry_identifiers_only(tmp_path):
    loader = _load_loader()
    d = _Driver()
    (tmp_path / "Users.xml").write_text(
        '<users><row Id="-1"/><row Id="2" Reputation="9" DisplayName="Ann"/></users>', encoding="utf-8")
    (tmp_path / "Posts.xml").write_text(_POSTS, encoding="utf-8")
    (tmp_path / "Tags.xml").write_text('<tags><row Id="1" TagName="a" Count="5"/></tags>', encoding="utf-8")
    loader.load_users_neo4j(d, tmp_path, 1000, dry_run=False)
    loader.load_posts_neo4j(d, tmp_path, 1000, dry_run=False)
    loader.load_tags_neo4j(d, tmp_path, 1000, dry_run=False)
    for query, kw in d.log:
        assert not re.search(r"\bSET\b", query), query          # no node properties are written
        for rows in kw.values():
            for r in rows:
                if "id" in r and "user_id" not in r and "answer_id" not in r:
                    assert set(r) == {"id"}
                if "name" in r:
                    assert set(r) == {"name"}
    users = next(kw["rows"] for q, kw in d.log if "User {id" in q)
    assert users == [{"id": 2}]
    assert loader.NEO4J_INDEXES == []


# ------------------------------------------------------------------ Elasticsearch loader

def _capture_es(loader):
    captured = []
    loader._es_bulk_flush = lambda es, batch, dry_run: captured.extend(batch)
    return captured


def test_es_documents_hold_text_and_event_identifiers_only(tmp_path):
    loader = _load_loader()
    cap = _capture_es(loader)
    (tmp_path / "Posts.xml").write_text(
        _POSTS.replace("</posts>", '  <row Id="4" PostTypeId="5" Body="wiki body" />\n</posts>'),
        encoding="utf-8")
    (tmp_path / "Comments.xml").write_text(
        '<comments><row Id="1" PostId="1" Score="2" Text="nice" CreationDate="2020-01-06T00:00:00.000" UserId="3"/></comments>',
        encoding="utf-8")
    (tmp_path / "Users.xml").write_text(
        '<users><row Id="-1" AboutMe="bot"/><row Id="2" AboutMe="I use R"/></users>', encoding="utf-8")
    (tmp_path / "Tags.xml").write_text(
        '<tags><row Id="1" TagName="a" Count="1" ExcerptPostId="4"/></tags>', encoding="utf-8")
    loader.load_posts_es(None, tmp_path, 1000, dry_run=False)
    loader.load_comments_es(None, tmp_path, 1000, dry_run=False)
    loader.load_users_es(None, tmp_path, 1000, dry_run=False)
    loader.load_tag_wikis_es(None, tmp_path, 1000, dry_run=False)
    by_type = {}
    for d in cap:
        by_type.setdefault(d["doc_type"], []).append({k: v for k, v in d.items() if not k.startswith("_")})
    assert set(by_type) == set(fidelity.ES_DOC_FIELDS)
    for doc_type, docs in by_type.items():
        for doc in docs:
            assert set(doc) == set(fidelity.ES_DOC_FIELDS[doc_type]), (doc_type, set(doc))
    assert by_type["question"][0]["title"] == "T"
    assert by_type["comment"][0]["user_id"] == 3 and by_type["comment"][0]["post_id"] == 1
    assert [d["user_id"] for d in by_type["user_about"]] == [2]       # user -1 not indexed
    assert by_type["tag_wiki"][0]["tag_name"] == "a"


# ------------------------------------------------------------------ stale-store guards

class _GuardCur:
    def __init__(self, links, cols):
        self.links, self.cols, self._last = links, cols, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self._last = sql

    def fetchone(self):
        return (self.links,)

    def fetchall(self):
        return self.cols


class _GuardConn:
    def __init__(self, links=False, cols=()):
        self.links, self.cols = links, list(cols)

    def cursor(self):
        return _GuardCur(self.links, self.cols)

    def rollback(self):
        pass


def test_pg_guard_refuses_a_stale_schema_and_accepts_the_current_one():
    loader = _load_loader()
    loader.assert_pg_schema_current(_GuardConn(cols=[("posts", "id"), ("posts", "score")]))
    with pytest.raises(RuntimeError, match="posts.owner_user_id"):
        loader.assert_pg_schema_current(_GuardConn(cols=[("posts", "id"), ("posts", "owner_user_id")]))
    with pytest.raises(RuntimeError, match="post_links"):
        loader.assert_pg_schema_current(_GuardConn(links=True))


def test_neo4j_guard_refuses_nodes_with_attributes():
    loader = _load_loader()

    class S:
        def __init__(self, rows):
            self.rows = rows

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, q, **kw):
            rows = self.rows

            class R:
                def data(s):
                    return rows
            return R()

    class D:
        def __init__(self, rows):
            self.rows = rows

        def session(self):
            return S(self.rows)

    loader.assert_neo4j_current(D([]))
    with pytest.raises(RuntimeError, match="Question"):
        loader.assert_neo4j_current(D([{"label": "Question", "props": ["id", "score"]}]))


def test_es_guard_refuses_a_stale_mapping(monkeypatch):
    loader = _load_loader()
    import urllib.request

    def fake(mapping):
        monkeypatch.setattr(loader, "_es_index_exists", lambda *a: True)
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda url, timeout=0: io.StringIO(json.dumps(mapping)))
    fake({"heterorag_content": {"mappings": {"properties": {"body": {}, "doc_type": {}}}}})
    loader.assert_es_current("http://es")
    fake({"heterorag_content": {"mappings": {"properties": {"body": {}, "tags": {}, "score": {}}}}})
    with pytest.raises(RuntimeError, match="score"):
        loader.assert_es_current("http://es")
    monkeypatch.setattr(loader, "_es_index_exists", lambda *a: False)
    loader.assert_es_current("http://es")             # no index yet: nothing to refuse


# ------------------------------------------------------------------ ANSWERS edge ordering

_OUT_OF_ORDER_POSTS = """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="2" ParentId="3" OwnerUserId="2" Score="1" Body="answer first" />
  <row Id="2" PostTypeId="1" OwnerUserId="2" Score="1" Body="q2" Title="Q2" />
  <row Id="3" PostTypeId="1" OwnerUserId="2" Score="1" Body="q3" Title="Q3" />
</posts>
"""


class _CountingTx:
    def __init__(self, log):
        self.log = log

    def run(self, query, **kw):
        self.log.append((query, kw))

        class R:
            def consume(s):
                import types
                return types.SimpleNamespace(counters=types.SimpleNamespace(relationships_created=len(kw["rows"])))
        return R()


class _CountingSession(_Session):
    def execute_write(self, fn):
        return fn(_CountingTx(self.log))


class _CountingDriver(_Driver):
    def session(self):
        return _CountingSession(self.log)


def test_answers_edge_is_created_even_when_the_question_comes_later(tmp_path):
    """Regression: with batch size 1 the answer (row 1) is flushed before its parent
    question (row 3) exists. ANSWERS must therefore be created in a separate pass."""
    loader = _load_loader()
    (tmp_path / "Posts.xml").write_text(_OUT_OF_ORDER_POSTS, encoding="utf-8")
    d = _CountingDriver()
    loader.load_posts_neo4j(d, tmp_path, 1, dry_run=False)
    assert not any("ANSWERS" in q for q, _ in d.log)       # not attempted while nodes are missing
    loader.load_answer_edges_neo4j(d, tmp_path, 1000, dry_run=False)
    rows = [r for q, kw in d.log if "ANSWERS" in q for r in kw["rows"]]
    assert rows == [{"answer_id": 1, "question_id": 3}]
