"""
heterorag/layer3/retrieval_executor.py
========================================
Step 6 — Parallel Retrieval Executor.

Consumes I₂ (List[ServiceQuery]) and fires each native query against its
target service concurrently via a ThreadPoolExecutor, collecting RawServiceResults.

Architecture decisions:
  - Truly parallel: all service queries are dispatched simultaneously.
    This is the key differentiator vs B3 (LLM Function-Calling, sequential).
    The latency gap between HeteroRAG and B3 is the primary empirical evidence
    for the parallel architecture contribution.
  - Connection-per-query: no persistent connection pool in the POC.
    Each executor thread opens, uses, and closes its connection.
    This is safe for the benchmark workload and avoids thread-safety complexity.
  - Timeout: each service query is bounded by per_service_timeout_ms.
    A timed-out service produces a RawServiceResult with succeeded=False.
    The pipeline continues with the remaining services (graceful degradation).
  - total_wall_ms is measured as wall-clock across the entire ThreadPoolExecutor.wait(),
    matching the RL metric definition in Foundation Doc v1.6 §7.6.

Service-specific executor classes:
  _SQLExecutor      — psycopg2, returns rows as list[dict]
  _GraphExecutor    — neo4j driver, returns records as list[dict]
  _DocumentExecutor — elasticsearch-py, returns BM25 hits as list[dict]

Reference: Foundation Doc v1.6 §5 (parallel retrieval), §7.6 (RL metric definition).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from typing import Any

from heterorag.layer1.models import ConnectionConfig
from heterorag.layer2.models import I2_QueryTranslationOutput, QueryLanguage, ServiceQuery

from .models import (
    RawServiceResult,
    RetrievalBatch,
    RetrievalError,
    SourceType,
)

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS   = 10_000   # 10 s per service
_DEFAULT_MAX_ROWS     = 50       # cap result set size to protect Layer 3 memory
_DEFAULT_ES_TOP_K     = 10       # BM25 top-k hits


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _pg_conn(cfg: ConnectionConfig):
    """Open a psycopg2 connection from a ConnectionConfig."""
    import psycopg2
    return psycopg2.connect(
        host     = cfg.host,
        port     = cfg.port,
        dbname   = cfg.database or "heterorag",
        user     = cfg.extra_params.get("user", "heterorag"),
        password = cfg.extra_params.get("password", "heterorag_secret"),
        connect_timeout = 5,
    )


def _neo4j_driver(cfg: ConnectionConfig):
    """Open a neo4j driver from a ConnectionConfig."""
    from neo4j import GraphDatabase
    uri = f"{cfg.extra_params.get('scheme', 'bolt')}://{cfg.host}:{cfg.port}"
    return GraphDatabase.driver(
        uri,
        auth=(
            cfg.extra_params.get("user", "neo4j"),
            cfg.extra_params.get("password", "heterorag_secret"),
        ),
    )


def _es_client(cfg: ConnectionConfig):
    """Open an Elasticsearch client from a ConnectionConfig."""
    from elasticsearch import Elasticsearch
    scheme = cfg.extra_params.get("scheme", "http")
    url    = f"{scheme}://{cfg.host}:{cfg.port}"
    return Elasticsearch(url, request_timeout=5)


# ---------------------------------------------------------------------------
# Per-service executors
# ---------------------------------------------------------------------------

class _SQLExecutor:
    def __init__(self, cfg: ConnectionConfig, max_rows: int):
        self._cfg      = cfg
        self._max_rows = max_rows

    def execute(self, query: str) -> list[dict[str, Any]]:
        conn = _pg_conn(self._cfg)
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                if cur.description is None:
                    return []
                cols  = [desc[0] for desc in cur.description]
                rows  = cur.fetchmany(self._max_rows)
                return [dict(zip(cols, row)) for row in rows]
        finally:
            conn.close()


class _GraphExecutor:
    def __init__(self, cfg: ConnectionConfig, max_rows: int):
        self._cfg      = cfg
        self._max_rows = max_rows

    def execute(self, query: str) -> list[dict[str, Any]]:
        driver = _neo4j_driver(self._cfg)
        try:
            with driver.session() as session:
                result  = session.run(query)
                records = result.fetch(self._max_rows)
                return [dict(r) for r in records]
        finally:
            driver.close()


class _DocumentExecutor:
    def __init__(self, cfg: ConnectionConfig, top_k: int):
        self._cfg   = cfg
        self._top_k = top_k

    def execute(self, query: str) -> list[dict[str, Any]]:
        es    = _es_client(self._cfg)
        index = self._cfg.database or "heterorag_content"
        body  = {
            "query": {
                "multi_match": {
                    "query":  query,
                    "fields": ["title^2", "body", "tag_name"],
                    "type":   "best_fields",
                }
            },
            "size": self._top_k,
        }
        resp = es.search(index=index, body=body)
        hits = resp["hits"]["hits"]
        # Flatten: merge _source fields with _id and _score for normaliser
        return [
            {
                "_id":    hit["_id"],
                "_score": hit["_score"],
                **hit.get("_source", {}),
            }
            for hit in hits
        ]


# ---------------------------------------------------------------------------
# Connection registry — maps service_id to its executor
# ---------------------------------------------------------------------------

class ConnectionRegistry:
    """
    Holds ConnectionConfig per service_id so the executor can open connections.
    Populated from ServiceDescriptors before the benchmark run.
    Thread-safe for read access (immutable after construction).
    """

    def __init__(self):
        self._configs: dict[str, ConnectionConfig] = {}

    def register(self, service_id: str, cfg: ConnectionConfig) -> None:
        self._configs[service_id] = cfg

    def get(self, service_id: str) -> ConnectionConfig | None:
        return self._configs.get(service_id)


# ---------------------------------------------------------------------------
# Parallel Retrieval Executor
# ---------------------------------------------------------------------------

class ParallelRetrievalExecutor:
    """
    Step 6: fires all service queries in I₂ concurrently and returns a RetrievalBatch.

    Usage:
        executor = ParallelRetrievalExecutor(connection_registry)
        batch = executor.execute(i2)
        # batch.total_wall_ms is the RL metric for this query
        # batch.results contains one RawServiceResult per service in I₂
    """

    def __init__(
        self,
        connection_registry:      ConnectionRegistry,
        per_service_timeout_ms:   int = _DEFAULT_TIMEOUT_MS,
        max_rows:                 int = _DEFAULT_MAX_ROWS,
        es_top_k:                 int = _DEFAULT_ES_TOP_K,
    ):
        self._registry   = connection_registry
        self._timeout_s  = per_service_timeout_ms / 1000.0
        self._max_rows   = max_rows
        self._es_top_k   = es_top_k

    # ------------------------------------------------------------------

    def execute(self, i2: I2_QueryTranslationOutput) -> RetrievalBatch:
        """
        Execute all queries in I₂ in parallel.

        Each query runs in its own thread. Threads are started simultaneously;
        total_wall_ms reflects the slowest service (the parallel bottleneck),
        not the sum of all service times.
        """
        if not i2.queries:
            log.warning("ParallelRetrievalExecutor: I₂ has no queries — returning empty batch")
            return RetrievalBatch(
                query_id      = i2.query_id,
                natural_query = i2.natural_query,
                results       = [],
            )

        n_workers  = len(i2.queries)
        wall_start = time.perf_counter()

        raw_results: list[RawServiceResult] = []

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_sq = {
                pool.submit(self._execute_one, sq): sq
                for sq in i2.queries
            }

            for future in as_completed(future_to_sq, timeout=self._timeout_s + 1.0):
                sq = future_to_sq[future]
                try:
                    result = future.result(timeout=0)   # already done — no wait
                    raw_results.append(result)
                except FutureTimeoutError:
                    log.error("ParallelRetrievalExecutor: timeout for '%s'", sq.service_id)
                    raw_results.append(self._timeout_result(sq))
                except Exception as exc:
                    log.error(
                        "ParallelRetrievalExecutor: unexpected error for '%s': %s",
                        sq.service_id, exc,
                    )
                    raw_results.append(self._error_result(sq, "UnexpectedError", str(exc)))

        total_wall_ms = (time.perf_counter() - wall_start) * 1000.0

        # Restore I₂ ordering for deterministic downstream processing
        sq_order = {sq.service_id: i for i, sq in enumerate(i2.queries)}
        raw_results.sort(key=lambda r: sq_order.get(r.service_id, 999))

        batch = RetrievalBatch(
            query_id      = i2.query_id,
            natural_query = i2.natural_query,
            results       = raw_results,
            total_wall_ms = total_wall_ms,
        )

        log.info(
            "ParallelRetrievalExecutor: query_id=%s  %d services  wall=%.1fms  "
            "succeeded=%d  failed=%d",
            i2.query_id, len(i2.queries), total_wall_ms,
            len(batch.successful()), len(batch.failed()),
        )
        return batch

    # ------------------------------------------------------------------
    # Private: single-service execution (runs in a thread)
    # ------------------------------------------------------------------

    def _execute_one(self, sq: ServiceQuery) -> RawServiceResult:
        """
        Execute one service query.  Runs inside a ThreadPoolExecutor worker.
        Opens, uses, and closes the connection within this call.
        """
        cfg = self._registry.get(sq.service_id)
        if cfg is None:
            return self._error_result(
                sq, "ConfigError",
                f"No ConnectionConfig registered for service_id='{sq.service_id}'"
            )

        source_type = {
            QueryLanguage.SQL:    SourceType.SQL,
            QueryLanguage.CYPHER: SourceType.GRAPH,
            QueryLanguage.BM25:   SourceType.DOCUMENT,
        }[sq.query_language]

        t0 = time.perf_counter()
        try:
            rows = self._dispatch(sq, cfg, source_type)
            retrieval_ms = (time.perf_counter() - t0) * 1000.0
            log.debug(
                "  [%s] %s: %d rows in %.1fms",
                sq.query_language.value, sq.service_id, len(rows), retrieval_ms,
            )
            return RawServiceResult(
                query_id         = sq.query_id,
                service_id       = sq.service_id,
                display_name     = sq.display_name,
                source_type      = source_type,
                native_query     = sq.native_query,
                retrieval_weight = sq.retrieval_weight,
                rows             = rows,
                retrieval_ms     = retrieval_ms,
                succeeded        = True,
            )
        except Exception as exc:
            retrieval_ms = (time.perf_counter() - t0) * 1000.0
            log.error("  [%s] %s failed: %s", sq.query_language.value, sq.service_id, exc)
            return RawServiceResult(
                query_id         = sq.query_id,
                service_id       = sq.service_id,
                display_name     = sq.display_name,
                source_type      = source_type,
                native_query     = sq.native_query,
                retrieval_weight = sq.retrieval_weight,
                rows             = [],
                retrieval_ms     = retrieval_ms,
                succeeded        = False,
                error            = RetrievalError(
                    service_id    = sq.service_id,
                    error_type    = type(exc).__name__,
                    error_message = str(exc),
                    native_query  = sq.native_query,
                ),
            )

    def _dispatch(
        self,
        sq:          ServiceQuery,
        cfg:         ConnectionConfig,
        source_type: SourceType,
    ) -> list[dict[str, Any]]:
        if source_type == SourceType.SQL:
            return _SQLExecutor(cfg, self._max_rows).execute(sq.native_query)
        elif source_type == SourceType.GRAPH:
            return _GraphExecutor(cfg, self._max_rows).execute(sq.native_query)
        elif source_type == SourceType.DOCUMENT:
            return _DocumentExecutor(cfg, self._es_top_k).execute(sq.native_query)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def _timeout_result(self, sq: ServiceQuery) -> RawServiceResult:
        source_type = {
            QueryLanguage.SQL:    SourceType.SQL,
            QueryLanguage.CYPHER: SourceType.GRAPH,
            QueryLanguage.BM25:   SourceType.DOCUMENT,
        }.get(sq.query_language, SourceType.SQL)
        return RawServiceResult(
            query_id         = sq.query_id,
            service_id       = sq.service_id,
            display_name     = sq.display_name,
            source_type      = source_type,
            native_query     = sq.native_query,
            retrieval_weight = sq.retrieval_weight,
            rows             = [],
            retrieval_ms     = self._timeout_s * 1000.0,
            succeeded        = False,
            error            = RetrievalError(
                service_id    = sq.service_id,
                error_type    = "Timeout",
                error_message = f"Query exceeded {self._timeout_s:.1f}s timeout",
                native_query  = sq.native_query,
            ),
        )

    def _error_result(self, sq: ServiceQuery, error_type: str, msg: str) -> RawServiceResult:
        source_type = {
            QueryLanguage.SQL:    SourceType.SQL,
            QueryLanguage.CYPHER: SourceType.GRAPH,
            QueryLanguage.BM25:   SourceType.DOCUMENT,
        }.get(sq.query_language, SourceType.SQL)
        return RawServiceResult(
            query_id         = sq.query_id,
            service_id       = sq.service_id,
            display_name     = sq.display_name,
            source_type      = source_type,
            native_query     = sq.native_query,
            retrieval_weight = sq.retrieval_weight,
            rows             = [],
            retrieval_ms     = 0.0,
            succeeded        = False,
            error            = RetrievalError(
                service_id    = sq.service_id,
                error_type    = error_type,
                error_message = msg,
                native_query  = sq.native_query,
            ),
        )
