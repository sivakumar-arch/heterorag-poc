"""Store clients, tested against fakes of the three drivers."""
import json
from contextlib import contextmanager
from decimal import Decimal

import pytest

from heterorag.federation import Kind, Mode, Op, Step, Store
from heterorag.federation.clients import (ElasticsearchClient, Neo4jClient, PostgresClient,
                                          substitute_binds)


def step(store=Store.SQL, kind=Kind.SET, output="question", mode=None, k=None):
    return Step("s1", Op.RETRIEVE, store, mode, "q", output=output, kind=kind, k=k)


# ------------------------------------------------------------------ PostgreSQL

class FakeCursor:
    def __init__(self, conn, name):
        self.conn, self.name = conn, name
        self._pos = 0

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def execute(self, sql, params=None):
        self.conn.executed.append((self.name, sql, params))
        self._pos = 0

    def fetchmany(self, n):
        if self.conn.fail_after is not None and self._pos >= self.conn.fail_after:
            raise TimeoutError("canceling statement due to statement timeout")
        stop = self._pos + n
        if self.conn.fail_after is not None:
            stop = min(stop, self.conn.fail_after)
        chunk = self.conn.rows[self._pos:stop]
        self._pos = stop
        return chunk


class FakeConn:
    def __init__(self, rows, fail_after=None):
        self.rows, self.fail_after = [(r,) if not isinstance(r, tuple) else r for r in rows], fail_after
        self.executed, self.readonly, self.closed, self.rolled_back = [], None, False, False

    def set_session(self, readonly=None): self.readonly = readonly
    def cursor(self, name=None): return FakeCursor(self, name)
    def rollback(self): self.rolled_back = True
    def close(self): self.closed = True


def pg(rows, **kw):
    conn = FakeConn(rows, kw.pop("fail_after", None))
    return PostgresClient(lambda: conn, page=kw.pop("page", 1000), **kw), conn


def test_pg_limit_and_exhausted_semantics():
    for limit, n_expected, exhausted in [(10, 5, True), (5, 5, True), (4, 4, False), (1, 1, False)]:
        c, _ = pg([1, 2, 3, 4, 5], page=2)
        r = c.fetch(step(), "SELECT 1", {}, limit)
        assert len(r.values) == n_expected and r.exhausted is exhausted, (limit, r)


def test_pg_is_read_only_passes_binds_as_parameters_and_cleans_up():
    c, conn = pg([7, 8])
    r = c.fetch(step(), "SELECT id FROM t WHERE x = ANY(%(s1)s)", {"s1": (3, 4)}, 10)
    assert r.values == [7, 8] and conn.readonly is True
    named = [e for e in conn.executed if e[0] == "federation_read"]
    assert named[0][1].endswith("ANY(%(s1)s)") and named[0][2] == {"s1": [3, 4]}
    assert any("statement_timeout" in e[1] for e in conn.executed)
    assert conn.closed and conn.rolled_back


def test_pg_scalar_is_coerced_and_checked():
    c, _ = pg([(Decimal("42"),)])
    r = c.fetch(step(kind=Kind.SCALAR, output="number"), "SELECT 1", {}, 10)
    assert r.values == [42] and isinstance(r.values[0], int) and r.exhausted
    c, _ = pg([(Decimal("2.5"),)])
    assert c.fetch(step(kind=Kind.SCALAR), "q", {}, 10).values == [2.5]
    c, _ = pg([(1,), (2,)])
    assert c.fetch(step(kind=Kind.SCALAR), "q", {}, 10).exhausted is False     # more than one row


def test_pg_error_after_some_rows_is_reported_as_partial():
    c, conn = pg(list(range(10)), fail_after=4, page=2)
    r = c.fetch(step(), "q", {}, 100)
    assert r.values == [0, 1, 2, 3] and r.exhausted is False and "timeout" in r.error
    assert conn.closed


def test_pg_error_with_no_rows_is_raised():
    c, conn = pg([1], fail_after=0)
    with pytest.raises(TimeoutError):
        c.fetch(step(), "q", {}, 100)
    assert conn.closed


# ------------------------------------------------------------------ Neo4j

class FakeTx:
    def __init__(self, rows, log): self.rows, self.log = rows, log
    def run(self, query, params):
        # the real driver accepts only text inside a managed transaction
        if not isinstance(query, str):
            raise TypeError("Query object is only supported for session.run")
        self.log.append((query, params))
        return iter(self.rows)


class FakeSession:
    def __init__(self, rows, log): self.rows, self.log = rows, log
    def execute_read(self, work): return work(FakeTx(self.rows, self.log))


def neo(rows):
    log = []

    @contextmanager
    def factory():
        yield FakeSession([(r,) for r in rows], log)
    def wrapper(timeout):
        def deco(fn):
            fn.timeout = timeout
            return fn
        return deco
    return Neo4jClient(factory, work_wrapper=wrapper), log


def test_neo4j_limit_exhausted_and_parameters():
    c, log = neo([1, 2, 3])
    r = c.fetch(step(Store.GRAPH), "MATCH (a) WHERE a.id IN $s1 RETURN a.id", {"s1": (10, 11)}, 3)
    assert r.values == [1, 2, 3] and r.exhausted
    assert log[0][1] == {"s1": [10, 11]} and log[0][0] == "MATCH (a) WHERE a.id IN $s1 RETURN a.id"
    r = c.fetch(step(Store.GRAPH), "q", {}, 2)
    assert r.values == [1, 2] and r.exhausted is False


def test_neo4j_timeout_is_attached_to_the_transaction_function():
    c, _ = neo([1])
    seen = []
    orig = c._wrap
    c._wrap = lambda t: (seen.append(t), orig(t))[1]
    c.fetch(step(Store.GRAPH), "q", {}, 5)
    assert seen == [c._timeout]


def test_neo4j_scalar():
    c, _ = neo([5])
    r = c.fetch(step(Store.GRAPH, Kind.SCALAR), "RETURN 5", {}, 10)
    assert r.values == [5] and r.exhausted
    c, _ = neo([5, 6])
    assert c.fetch(step(Store.GRAPH, Kind.SCALAR), "q", {}, 10).exhausted is False


# ------------------------------------------------------------------ Elasticsearch

class FakeES:
    def __init__(self, ids, field="post_id", fail_on_call=None):
        self.docs = [{field: i} for i in ids]
        self.field, self.calls, self.closed, self.opened = field, [], False, 0
        self.fail_on_call = fail_on_call

    def open_point_in_time(self, index, keep_alive):
        self.opened += 1
        return {"id": "PIT1"}

    def close_point_in_time(self, id): self.closed = True

    def search(self, index=None, body=None):
        self.calls.append(body)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise ConnectionError("shard failure")
        start = body["search_after"][0] + 1 if "search_after" in body else 0
        if "pit" not in body:
            take = list(enumerate(self.docs))[: body["size"]]
        else:
            take = list(enumerate(self.docs))[start:start + body["size"]]
        return {"hits": {"hits": [{"_source": d, "sort": [i]} for i, d in take]}}


def dsl(**q): return json.dumps({"query": q})


def es(ids, **kw):
    f = FakeES(ids, **kw)
    return ElasticsearchClient(f, "idx", page=3), f


DOC = step(Store.DOC, mode=Mode.FILTER)


def test_es_filter_enumerates_every_match_across_pages_and_closes_the_pit():
    c, f = es(range(10))
    r = c.fetch(DOC, dsl(term={"doc_type": "question"}), {}, 100)
    assert r.values == list(range(10)) and r.exhausted and f.closed
    assert all("sort" in b and b["sort"] == [{"_shard_doc": "asc"}] for b in f.calls)
    assert f.calls[0]["_source"] == ["post_id"] and "search_after" not in f.calls[0]


def test_es_filter_limit_semantics():
    c, _ = es(range(10))
    r = c.fetch(DOC, dsl(match_all={}), {}, 10)
    assert len(r.values) == 10 and r.exhausted
    c, _ = es(range(10))
    r = c.fetch(DOC, dsl(match_all={}), {}, 4)
    assert r.values == [0, 1, 2, 3] and r.exhausted is False


def test_es_filter_partial_on_mid_error_and_pit_closed_on_failure():
    c, f = es(range(10), fail_on_call=2)
    r = c.fetch(DOC, dsl(match_all={}), {}, 100)
    assert r.values == [0, 1, 2] and r.exhausted is False and "shard failure" in r.error and f.closed
    c, f = es(range(10), fail_on_call=1)
    with pytest.raises(ConnectionError):
        c.fetch(DOC, dsl(match_all={}), {}, 100)
    assert f.closed


def test_es_bind_sentinel_is_replaced_and_only_the_query_is_used():
    c, f = es([1])
    c.fetch(DOC, dsl(bool={"filter": [{"terms": {"post_id": "$bind:s1"}}, {"term": {"doc_type": "answer"}}]}),
            {"s1": (5, 6, 7)}, 10)
    sent = f.calls[0]["query"]["bool"]["filter"]
    assert sent[0] == {"terms": {"post_id": [5, 6, 7]}} and sent[1]["term"]["doc_type"] == "answer"
    with pytest.raises(ValueError, match="unknown bind"):
        c.fetch(DOC, dsl(terms={"post_id": "$bind:zz"}), {"s1": (1,)}, 10)


def test_es_search_mode_is_an_explicit_top_k_without_a_pit():
    c, f = es(range(10))
    r = c.fetch(step(Store.DOC, mode=Mode.SEARCH, k=4), dsl(match={"body": "pandas"}), {}, 100)
    assert r.values == [0, 1, 2, 3] and r.exhausted and f.opened == 0 and f.calls[0]["size"] == 4


@pytest.mark.parametrize("query,msg", [
    ("not json", "not valid JSON"),
    (json.dumps({"query": {}, "aggs": {}}), "exactly one key"),
    (json.dumps({"size": 5}), "exactly one key"),
    (json.dumps([1]), "exactly one key"),
])
def test_es_rejects_queries_that_are_not_a_single_query_object(query, msg):
    c, _ = es([1])
    with pytest.raises(ValueError, match=msg):
        c.fetch(DOC, query, {}, 10)


def test_es_entity_to_identifier_field():
    f = FakeES([1, 2], field="user_id")
    c = ElasticsearchClient(f, "idx", page=3)
    assert c.fetch(step(Store.DOC, output="user", mode=Mode.FILTER), dsl(match_all={}), {}, 10).values == [1, 2]
    assert f.calls[0]["_source"] == ["user_id"]
    with pytest.raises(ValueError, match="no document identifier field"):
        c.fetch(step(Store.DOC, output="comment", mode=Mode.FILTER), dsl(match_all={}), {}, 10)


def test_substitute_binds_is_structural():
    q = {"a": ["$bind:x", {"b": "$bind:x"}], "c": "plain"}
    assert substitute_binds(q, {"x": (1, 2)}) == {"a": [[1, 2], {"b": [1, 2]}], "c": "plain"}
