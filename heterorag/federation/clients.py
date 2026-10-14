"""
heterorag/federation/clients.py
===============================
Store clients for the federation executor: PostgreSQL, Neo4j, Elasticsearch.

Each client runs one native query and returns ``FetchResult``. The contract the executor relies on:

* at most ``limit`` values are returned, and ``exhausted`` is False exactly when more existed;
* the connection is read-only;
* bound identifiers arrive as parameters (SQL, Cypher) or replace a sentinel in the query body
  (Elasticsearch); they are never concatenated into query text by this code;
* an error after some rows were read is reported as a partial result, not hidden.

Connections are created through injectable factories, so the logic is unit-tested without a
live database. Live behaviour is checked by ``benchmark_v2.validate_instances``.
"""

from __future__ import annotations

import json
import time
from decimal import Decimal
from typing import Any, Callable, Mapping, Sequence

from .executor import Estimate, FetchResult
from .guarantee import Kind
from .plan import Mode, Step

# Elasticsearch field that holds the identifier of each entity type.
ENTITY_ID_FIELD = {"question": "post_id", "answer": "post_id", "user": "user_id", "tag": "tag_name"}

PAGE = 1000
PROBE_TIMEOUT_MS = 3_000      # a size probe must be much cheaper than the query it informs


def _plain(v: Any) -> Any:
    """Convert driver-specific numbers to plain Python values."""
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    return v


def _brief(ex: Exception, limit: int = 300) -> str:
    """One short line describing a driver error, for a repair prompt."""
    return " ".join(str(ex).split())[:limit]


def _scalar_result(values: list, exhausted: bool) -> FetchResult:
    return FetchResult(values[:1], exhausted and len(values) <= 1)


# --------------------------------------------------------------------------- PostgreSQL

class PostgresClient:
    def __init__(self, connect: Callable[[], Any], *, statement_timeout_ms: int = 20_000, page: int = PAGE,
                 is_query_error: Callable[[Exception], bool] | None = None):
        self._connect = connect
        self._timeout = int(statement_timeout_ms)
        self._page = page
        self._is_query_error = is_query_error or _pg_query_error

    def validate(self, step: Step, query: str) -> list[str]:
        """Ask the server to plan the query without running it (``EXPLAIN``). Returns the server's
        complaints, or an empty list. Errors that are not about the query (connection, timeout)
        are raised. No data is read."""
        conn = self._connect()
        try:
            conn.set_session(readonly=True)
            params = {step.bind.name: []} if step.bind else {}
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = %s", (self._timeout,))
                cur.execute("EXPLAIN " + query.strip().rstrip(";"), params)
            return []
        except Exception as ex:
            if self._is_query_error(ex):
                return [f"PostgreSQL rejected the query: {_brief(ex)}"]
            raise
        finally:
            try:
                conn.rollback()
            finally:
                conn.close()

    def estimate(self, step: Step, query: str, limit: int) -> Estimate | None:
        """Rows of an unbound set query. First a count bounded at ``limit + 1`` rows under a short
        timeout (exact, and it also tells whether the cap would truncate the query); if that does
        not finish, the server planner's row estimate. None when neither works."""
        if step.bind is not None or step.kind is not Kind.SET:
            return None
        inner = query.strip().rstrip(";")
        t0 = time.perf_counter()
        conn = self._connect()
        try:
            conn.set_session(readonly=True)
            try:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL statement_timeout = %s", (PROBE_TIMEOUT_MS,))
                    cur.execute(f"SELECT count(*) FROM (SELECT 1 FROM (\n{inner}\n) AS s LIMIT {int(limit) + 1}) AS c")
                    n = int(cur.fetchone()[0])
                return Estimate(n, "lower" if n > limit else "count", (time.perf_counter() - t0) * 1000.0)
            except Exception:
                conn.rollback()
            try:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL statement_timeout = %s", (PROBE_TIMEOUT_MS,))
                    cur.execute("EXPLAIN (FORMAT JSON) \n" + inner)
                    plan = cur.fetchone()[0]
                plan = json.loads(plan) if isinstance(plan, str) else plan
                rows = int(plan[0]["Plan"]["Plan Rows"])
                return Estimate(rows, "planner", (time.perf_counter() - t0) * 1000.0)
            except Exception:
                return None
        finally:
            try:
                conn.rollback()
            finally:
                conn.close()

    def fetch(self, step: Step, query: str, binds: Mapping[str, Sequence], limit: int) -> FetchResult:
        conn = self._connect()
        values: list = []
        try:
            conn.set_session(readonly=True)
            with conn.cursor() as c0:
                c0.execute("SET LOCAL statement_timeout = %s", (self._timeout,))
            with conn.cursor(name="federation_read") as cur:
                cur.execute(query, {k: list(v) for k, v in binds.items()})
                try:
                    exhausted = False
                    while len(values) <= limit:
                        chunk = cur.fetchmany(min(self._page, limit + 1 - len(values)))
                        if not chunk:
                            exhausted = True
                            break
                        values.extend(_plain(r[0]) for r in chunk)
                except Exception as ex:                         # error after some rows were read
                    if values:
                        return FetchResult(values[:limit], exhausted=False,
                                           error=f"{type(ex).__name__}: {ex}")
                    raise
            if step.kind is Kind.SCALAR:
                return _scalar_result(values, exhausted)
            return FetchResult(values[:limit], exhausted and len(values) <= limit)
        finally:
            try:
                conn.rollback()
            finally:
                conn.close()


def _pg_query_error(ex: Exception) -> bool:
    import psycopg2
    return isinstance(ex, (psycopg2.ProgrammingError, psycopg2.DataError))


# --------------------------------------------------------------------------- Neo4j

def _unit_of_work(timeout_s: float):
    """Server-side transaction timeout. The driver rejects ``Query`` objects inside a managed
    transaction (``tx.run`` takes plain text there), so the timeout is attached to the
    transaction function instead."""
    from neo4j import unit_of_work
    return unit_of_work(timeout=timeout_s)


def _neo4j_query_error(ex: Exception) -> bool:
    from neo4j.exceptions import ClientError
    return isinstance(ex, ClientError)


class Neo4jClient:
    """``session_factory()`` returns a context manager yielding an object with
    ``execute_read(work)``, as the Neo4j driver's session does."""

    def __init__(self, session_factory: Callable[[], Any], *, timeout_s: float = 20.0,
                 work_wrapper: Callable[[float], Callable[[Callable], Callable]] | None = None,
                 is_query_error: Callable[[Exception], bool] | None = None):
        self._session = session_factory
        self._timeout = timeout_s
        self._wrap = work_wrapper or _unit_of_work
        self._is_query_error = is_query_error or _neo4j_query_error

    def validate(self, step: Step, query: str) -> list[str]:
        """Plan the query without running it (``EXPLAIN``). Syntax errors and undefined variables
        are reported; connection errors are raised. No data is read."""
        params = {step.bind.name: []} if step.bind else {}

        def work(tx):
            tx.run("EXPLAIN " + query.strip().rstrip(";"), params).consume()

        try:
            with self._session() as session:
                session.execute_read(self._wrap(self._timeout)(work))
            return []
        except Exception as ex:
            if self._is_query_error(ex):
                return [f"Neo4j rejected the query: {_brief(ex)}"]
            raise

    def estimate(self, step: Step, query: str, limit: int) -> Estimate | None:
        """Rows of an unbound set query: a count over the query as a subquery, bounded at
        ``limit + 1`` rows under a short timeout; otherwise the planner's EstimatedRows from EXPLAIN."""
        if step.bind is not None or step.kind is not Kind.SET:
            return None
        inner = query.strip().rstrip(";")
        t0 = time.perf_counter()

        def count(tx):
            rec = tx.run(f"CALL {{\n{inner}\nLIMIT $probe_limit }} RETURN count(*) AS n",
                         {"probe_limit": int(limit) + 1}).single()
            return int(rec["n"])

        def explain(tx):
            plan = tx.run("EXPLAIN \n" + inner, {}).consume().plan
            return int(round(float(plan["args"]["EstimatedRows"])))

        wrap = self._wrap(PROBE_TIMEOUT_MS / 1000.0)
        for work, basis in ((count, "count"), (explain, "planner")):
            try:
                with self._session() as session:
                    n = session.execute_read(wrap(work))
                if basis == "count":
                    basis = "lower" if n > limit else "count"
                return Estimate(n, basis, (time.perf_counter() - t0) * 1000.0)
            except Exception:
                continue
        return None

    def fetch(self, step: Step, query: str, binds: Mapping[str, Sequence], limit: int) -> FetchResult:
        params = {k: list(v) for k, v in binds.items()}

        def work(tx):
            out: list = []
            for rec in tx.run(query, params):
                out.append(_plain(rec[0]))
                if len(out) > limit:
                    break
            return out

        with self._session() as session:
            values = session.execute_read(self._wrap(self._timeout)(work))
        if step.kind is Kind.SCALAR:
            return _scalar_result(values, True)
        return FetchResult(values[:limit], len(values) <= limit)


# --------------------------------------------------------------------------- Elasticsearch

def substitute_binds(node: Any, binds: Mapping[str, Sequence]) -> Any:
    """Replace the sentinel string ``"$bind:<name>"`` by the identifier list, anywhere in a
    query body. A sentinel with no matching bind is an error."""
    if isinstance(node, str) and node.startswith("$bind:"):
        name = node[len("$bind:"):]
        if name not in binds:
            raise ValueError(f"query refers to unknown bind {name!r}")
        return list(binds[name])
    if isinstance(node, list):
        return [substitute_binds(x, binds) for x in node]
    if isinstance(node, dict):
        return {k: substitute_binds(v, binds) for k, v in node.items()}
    return node


def _es_query_error(ex: Exception) -> bool:
    from elasticsearch import BadRequestError
    return isinstance(ex, BadRequestError)


class ElasticsearchClient:
    """``es`` is an Elasticsearch client (8.x). ``filter`` mode enumerates every match with a
    point in time and ``search_after``; ``search`` mode returns the explicit top-k by relevance."""

    def __init__(self, es: Any, index: str, *, page: int = PAGE, keep_alive: str = "1m",
                 is_query_error: Callable[[Exception], bool] | None = None):
        self._es, self._index, self._page, self._keep = es, index, page, keep_alive
        self._is_query_error = is_query_error or _es_query_error

    def validate(self, step: Step, query: str) -> list[str]:
        """Use the server's query validation (``_validate/query``), which parses the query against
        the index mapping without running it. No documents are read."""
        try:
            body = json.loads(query)
            q = substitute_binds(body["query"], {step.bind.name: []} if step.bind else {})
        except (ValueError, KeyError, TypeError) as ex:
            return [f"Elasticsearch query is malformed: {_brief(ex)}"]
        try:
            resp = self._es.indices.validate_query(index=self._index, body={"query": q}, explain=True)
        except Exception as ex:
            if self._is_query_error(ex):
                return [f"Elasticsearch rejected the query: {_brief(ex)}"]
            raise
        if resp.get("valid", True):
            return []
        why = "; ".join(e["error"] for e in resp.get("explanations", []) if e.get("error")) \
            or str(resp.get("error", "the query is not valid"))
        return [f"Elasticsearch rejected the query: {why[:300]}"]

    def estimate(self, step: Step, query: str, limit: int) -> Estimate | None:
        """Rows of an unbound filter query: the index's own count (exact, no documents read). The
        count is not bounded by ``limit``, so it is exact even above it; the basis is still "lower"
        in that case, because what matters to the caller is that a read capped at ``limit`` rows
        would be truncated (the same meaning the sql and graph probes give the label)."""
        if step.bind is not None or step.mode is not Mode.FILTER:
            return None
        t0 = time.perf_counter()
        try:
            q = json.loads(query)["query"]
            n = int(self._es.count(index=self._index, body={"query": q})["count"])
        except Exception:
            return None
        return Estimate(n, "lower" if n > limit else "count", (time.perf_counter() - t0) * 1000.0)

    def fetch(self, step: Step, query: str, binds: Mapping[str, Sequence], limit: int) -> FetchResult:
        try:
            body = json.loads(query)
        except ValueError as ex:
            raise ValueError(f"the document query is not valid JSON: {ex}") from ex
        if not isinstance(body, dict) or set(body) != {"query"}:
            raise ValueError("the document query must be an object with exactly one key, 'query'")
        field = ENTITY_ID_FIELD.get(step.output)
        if field is None:
            raise ValueError(f"no document identifier field for entity type {step.output!r}")
        q = substitute_binds(body["query"], binds)
        if step.mode is Mode.SEARCH:
            return self._search(q, field, min(int(step.k or limit), limit))
        return self._filter(q, field, limit)

    def _search(self, q: dict, field: str, k: int) -> FetchResult:
        resp = self._es.search(index=self._index, body={"query": q, "size": k, "_source": [field],
                                                          "track_total_hits": False})
        return FetchResult([h["_source"][field] for h in resp["hits"]["hits"]], True)

    def _filter(self, q: dict, field: str, limit: int) -> FetchResult:
        pit = self._es.open_point_in_time(index=self._index, keep_alive=self._keep)["id"]
        values: list = []
        after = None
        try:
            while len(values) <= limit:
                body: dict = {"query": q, "size": min(self._page, limit + 1 - len(values)),
                              "_source": [field], "sort": [{"_shard_doc": "asc"}],
                              "pit": {"id": pit, "keep_alive": self._keep}, "track_total_hits": False}
                if after is not None:
                    body["search_after"] = after
                try:
                    hits = self._es.search(body=body)["hits"]["hits"]
                except Exception as ex:
                    if values:
                        return FetchResult(values[:limit], exhausted=False,
                                           error=f"{type(ex).__name__}: {ex}")
                    raise
                if not hits:
                    return FetchResult(values, True)
                values.extend(h["_source"][field] for h in hits)
                after = hits[-1]["sort"]
            return FetchResult(values[:limit], False)
        finally:
            try:
                self._es.close_point_in_time(id=pit)
            except Exception:                                  # the point in time expires anyway
                pass


# --------------------------------------------------------------------------- live factories

def live_clients(env: Mapping[str, str] | None = None) -> dict:
    """Clients for the local docker stores, configured like the loader and the fidelity check."""
    import os
    e = env if env is not None else os.environ
    from .plan import Store
    from ..es_index import INDEX_NAME

    def pg():
        import psycopg2
        return psycopg2.connect(e.get("PG_DSN", "postgresql://heterorag:heterorag_secret@localhost:5432/heterorag"))

    from neo4j import READ_ACCESS, GraphDatabase
    driver = GraphDatabase.driver(e.get("NEO4J_URI", "bolt://localhost:7687"),
                                  auth=(e.get("NEO4J_USER", "neo4j"), e.get("NEO4J_PASSWORD", "heterorag_secret")))

    def neo4j_session():
        return driver.session(default_access_mode=READ_ACCESS)

    from elasticsearch import Elasticsearch
    es = Elasticsearch(e.get("ES_URL", "http://localhost:9200"), request_timeout=60)
    return {Store.SQL: PostgresClient(pg), Store.GRAPH: Neo4jClient(neo4j_session),
            Store.DOC: ElasticsearchClient(es, INDEX_NAME)}
