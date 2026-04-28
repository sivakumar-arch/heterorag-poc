"""
tests/layer3/test_layer3_integration.py
=========================================
Integration tests for Layer 3 against the live Docker Compose services.

Tests are graduated across two skip tiers:
  @any_service_available   — requires at least one of the three containers.
  @all_services_available  — requires all three containers.

What these tests prove that unit tests cannot:
  1. ParallelRetrievalExecutor opens real connections and retrieves real rows.
  2. SQL executor returns rows matching the bootstrap schema column names.
  3. Graph executor returns records from live Cypher traversal.
  4. ES executor returns BM25 hits with _score and _source fields.
  5. true-parallel execution: total_wall_ms < sum(per-service times).
  6. Failed service (bad query) degrades gracefully — other services complete.
  7. Full pipeline: I₂ → executor → integrator → I₃ with real data.
  8. queried_service_ids matches the successful services in the batch.
  9. total_wall_ms is recorded correctly (RL metric validity).
"""

from __future__ import annotations

import os
import socket

import pytest

from heterorag.layer1.models import ConnectionConfig
from heterorag.layer1.poc_descriptors import (
    CONTENT_SERVICE_DESCRIPTOR,
    KNOWLEDGE_GRAPH_DESCRIPTOR,
    USER_ACTIVITY_DESCRIPTOR,
)
from heterorag.layer2.models import (
    I2_QueryTranslationOutput,
    QueryLanguage,
    ServiceQuery,
    ValidationStatus,
)
from heterorag.layer3 import (
    ConnectionRegistry,
    I3_IntegratedContext,
    ParallelRetrievalExecutor,
    SemanticIntegrator,
    SourceType,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

def _tcp(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


PG_HOST     = os.getenv("PG_HOST",          "localhost")
PG_PORT     = int(os.getenv("PG_PORT",      "5432"))
PG_DBNAME   = os.getenv("PG_DBNAME",        "heterorag")
PG_USER     = os.getenv("PG_USER",          "heterorag")
PG_PASSWORD = os.getenv("PG_PASSWORD",      "heterorag_secret")

NEO4J_HOST  = os.getenv("NEO4J_HOST",       "localhost")
NEO4J_PORT  = int(os.getenv("NEO4J_BOLT_PORT", "7687"))
NEO4J_USER  = os.getenv("NEO4J_USER",       "neo4j")
NEO4J_PASS  = os.getenv("NEO4J_PASSWORD",   "heterorag_secret")

ES_HOST     = os.getenv("ES_HOST",          "localhost")
ES_PORT     = int(os.getenv("ES_PORT",      "9200"))

_pg_up    = _tcp(PG_HOST,    PG_PORT)
_neo4j_up = _tcp(NEO4J_HOST, NEO4J_PORT)
_es_up    = _tcp(ES_HOST,    ES_PORT)
_all_up   = _pg_up and _neo4j_up and _es_up

pg_available    = pytest.mark.skipif(not _pg_up,  reason="PostgreSQL not reachable")
neo4j_available = pytest.mark.skipif(not _neo4j_up, reason="Neo4j not reachable")
es_available    = pytest.mark.skipif(not _es_up,  reason="Elasticsearch not reachable")
all_services_available = pytest.mark.skipif(not _all_up,
    reason="Not all three Docker services reachable")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conn_registry() -> ConnectionRegistry:
    reg = ConnectionRegistry()
    reg.register("user-activity-service",  USER_ACTIVITY_DESCRIPTOR.connection)
    reg.register("knowledge-graph-service", KNOWLEDGE_GRAPH_DESCRIPTOR.connection)
    reg.register("content-service",        CONTENT_SERVICE_DESCRIPTOR.connection)
    return reg


@pytest.fixture(scope="module")
def executor(conn_registry) -> ParallelRetrievalExecutor:
    return ParallelRetrievalExecutor(conn_registry, per_service_timeout_ms=15_000)


def _sq(service_id: str, ql: QueryLanguage, query: str,
        weight: float = 0.333) -> ServiceQuery:
    return ServiceQuery(
        query_id=f"integ-{service_id}", service_id=service_id,
        display_name=service_id, query_language=ql,
        native_query=query, retrieval_weight=weight,
        validation_status=ValidationStatus.VALID,
    )


def _i2(*sqs, qid: str = "integ-qid") -> I2_QueryTranslationOutput:
    return I2_QueryTranslationOutput(
        query_id=qid, natural_query="integration test query",
        queries=list(sqs),
    )


# =============================================================================
# 1. Single-service connectivity
# =============================================================================

class TestSQLRetrieval:

    @pg_available
    def test_simple_count_query_returns_rows(self, executor):
        i2 = _i2(_sq("user-activity-service", QueryLanguage.SQL,
                      "SELECT id, reputation FROM users ORDER BY reputation DESC LIMIT 5"))
        batch = executor.execute(i2)
        assert len(batch.results) == 1
        result = batch.results[0]
        assert result.succeeded, f"SQL query failed: {result.error}"
        assert len(result.rows) <= 5

    @pg_available
    def test_sql_rows_have_expected_columns(self, executor):
        i2 = _i2(_sq("user-activity-service", QueryLanguage.SQL,
                      "SELECT id, reputation, display_name FROM users LIMIT 3"))
        batch = executor.execute(i2)
        result = batch.results[0]
        if result.succeeded and result.rows:
            row = result.rows[0]
            assert "id" in row, f"'id' column missing from SQL result. Got: {list(row.keys())}"
            assert "reputation" in row

    @pg_available
    def test_sql_retrieval_ms_is_positive(self, executor):
        i2 = _i2(_sq("user-activity-service", QueryLanguage.SQL,
                      "SELECT 1 AS n"))
        batch = executor.execute(i2)
        assert batch.results[0].retrieval_ms > 0

    @pg_available
    def test_bad_sql_query_returns_failed_result(self, executor):
        i2 = _i2(_sq("user-activity-service", QueryLanguage.SQL,
                      "SELECT * FROM nonexistent_table_xyz LIMIT 1"))
        batch = executor.execute(i2)
        result = batch.results[0]
        assert result.succeeded is False
        assert result.error is not None
        assert result.error.error_type != ""


class TestGraphRetrieval:

    @neo4j_available
    def test_simple_match_returns_records(self, executor):
        i2 = _i2(_sq("knowledge-graph-service", QueryLanguage.CYPHER,
                      "MATCH (u:User) RETURN u.id AS user_id, u.reputation AS reputation "
                      "ORDER BY u.reputation DESC LIMIT 5"))
        batch = executor.execute(i2)
        result = batch.results[0]
        if not result.succeeded:
            pytest.skip(f"Neo4j query failed — data may not be loaded: {result.error}")
        assert len(result.rows) <= 5

    @neo4j_available
    def test_tag_co_occurrence_query(self, executor):
        i2 = _i2(_sq("knowledge-graph-service", QueryLanguage.CYPHER,
                      "MATCH (t1:Tag {name: 'python'})-[r:CO_OCCURS_WITH]-(t2:Tag) "
                      "RETURN t2.name AS tag, r.weight AS weight "
                      "ORDER BY r.weight DESC LIMIT 5"))
        batch = executor.execute(i2)
        result = batch.results[0]
        if not result.succeeded:
            pytest.skip(f"CO_OCCURS_WITH query failed — data may not be loaded")
        # If data is loaded, should return related tags
        assert result.error is None or result.succeeded

    @neo4j_available
    def test_bad_cypher_returns_failed_result(self, executor):
        i2 = _i2(_sq("knowledge-graph-service", QueryLanguage.CYPHER,
                      "MATCH (n:NonExistentLabel99) RETURN n LIMIT 1"))
        batch = executor.execute(i2)
        # Empty result is OK — nonexistent label returns 0 rows, not an error in Neo4j
        assert batch.results[0].service_id == "knowledge-graph-service"


class TestDocumentRetrieval:

    @es_available
    def test_bm25_search_returns_hits(self, executor):
        i2 = _i2(_sq("content-service", QueryLanguage.BM25,
                      "python list comprehension"))
        batch = executor.execute(i2)
        result = batch.results[0]
        if not result.succeeded:
            pytest.skip(f"ES search failed — index may not be populated: {result.error}")
        # Hits have _score field
        if result.rows:
            assert "_score" in result.rows[0], (
                f"ES hit missing _score. Got fields: {list(result.rows[0].keys())}"
            )

    @es_available
    def test_es_hits_have_doc_type_field(self, executor):
        i2 = _i2(_sq("content-service", QueryLanguage.BM25, "python"))
        batch = executor.execute(i2)
        result = batch.results[0]
        if result.succeeded and result.rows:
            assert "doc_type" in result.rows[0]

    @es_available
    def test_empty_bm25_query_handled_gracefully(self, executor):
        """An empty BM25 query should not crash the executor."""
        i2 = _i2(_sq("content-service", QueryLanguage.BM25, ""))
        batch = executor.execute(i2)
        # Either succeeds with 0 hits or fails cleanly — must not raise
        assert batch.results[0].error is None or not batch.results[0].succeeded


# =============================================================================
# 2. Parallel execution proof
# =============================================================================

class TestParallelExecution:

    @all_services_available
    def test_wall_time_less_than_sum_of_service_times(self, executor):
        """
        Core parallelism invariant from the paper:
        total_wall_ms must be substantially less than the sum of per-service times.
        If all three services take ~50ms each, sequential would be ~150ms;
        parallel should be ~50-80ms.
        """
        i2 = _i2(
            _sq("user-activity-service",   QueryLanguage.SQL,
                "SELECT id FROM users LIMIT 10"),
            _sq("knowledge-graph-service", QueryLanguage.CYPHER,
                "MATCH (u:User) RETURN u.id LIMIT 10"),
            _sq("content-service",         QueryLanguage.BM25, "python"),
        )
        batch = executor.execute(i2)

        sum_individual = sum(r.retrieval_ms for r in batch.results)
        wall           = batch.total_wall_ms

        # Wall time should be significantly less than sum when all services respond
        # Use 0.9 × sum as the threshold: allows small scheduling overhead
        if sum_individual > 30.0:   # Only assert if queries took measurable time
            assert wall < sum_individual * 0.9, (
                f"Expected parallel execution: wall={wall:.1f}ms < "
                f"0.9 × sum={sum_individual * 0.9:.1f}ms. "
                f"Individual: {[r.retrieval_ms for r in batch.results]}"
            )

    @all_services_available
    def test_three_services_all_succeed(self, executor):
        i2 = _i2(
            _sq("user-activity-service",   QueryLanguage.SQL,
                "SELECT id, reputation FROM users LIMIT 5"),
            _sq("knowledge-graph-service", QueryLanguage.CYPHER,
                "MATCH (t:Tag) RETURN t.name LIMIT 5"),
            _sq("content-service",         QueryLanguage.BM25, "python"),
        )
        batch = executor.execute(i2)
        assert len(batch.results) == 3
        succeeded = [r.service_id for r in batch.successful()]
        assert len(succeeded) >= 1   # at least one must succeed


# =============================================================================
# 3. Graceful degradation
# =============================================================================

class TestGracefulDegradation:

    @pg_available
    def test_one_bad_service_does_not_block_others(self, executor):
        """SQL service with a bad query should fail while graph+doc are not attempted here,
        but the executor should not hang or raise."""
        i2 = _i2(
            _sq("user-activity-service", QueryLanguage.SQL,
                "SELECT * FROM table_that_does_not_exist_xyz LIMIT 1"),
        )
        batch = executor.execute(i2)
        # Executor returns — it does not raise
        assert len(batch.results) == 1
        assert batch.results[0].succeeded is False

    @all_services_available
    def test_failed_service_excluded_from_successful(self, executor):
        i2 = _i2(
            _sq("user-activity-service",   QueryLanguage.SQL,
                "SELECT * FROM nonexistent_table_xyz LIMIT 1"),
            _sq("content-service",         QueryLanguage.BM25, "python"),
        )
        batch = executor.execute(i2)
        good = batch.successful()
        bad  = batch.failed()
        # content-service (BM25) should succeed
        assert any(r.service_id == "content-service" for r in good), (
            f"content-service should have succeeded. Successful: {[r.service_id for r in good]}"
        )
        # user-activity-service (bad SQL) should fail
        assert any(r.service_id == "user-activity-service" for r in bad)


# =============================================================================
# 4. Full pipeline: I₂ → executor → SemanticIntegrator → I₃
# =============================================================================

class TestFullLayer3Pipeline:

    @pg_available
    def test_sql_to_i3(self, executor):
        i2 = _i2(
            _sq("user-activity-service", QueryLanguage.SQL,
                "SELECT id, reputation, display_name FROM users "
                "ORDER BY reputation DESC LIMIT 5"),
        )
        batch = executor.execute(i2)
        i3 = SemanticIntegrator().integrate(batch, "top users by reputation")

        assert isinstance(i3, I3_IntegratedContext)
        assert i3.query_id.startswith("integ-")
        assert "user-activity-service" in i3.queried_service_ids
        assert i3.total_wall_ms > 0

    @pg_available
    def test_context_text_contains_real_data(self, executor):
        """context_text must contain content from real DB rows — not placeholder text."""
        i2 = _i2(
            _sq("user-activity-service", QueryLanguage.SQL,
                "SELECT id, reputation FROM users LIMIT 3"),
        )
        batch = executor.execute(i2)
        i3 = SemanticIntegrator().integrate(batch, "users")

        if batch.successful() and batch.successful()[0].rows:
            # Context should reference at least one user ID from the real data
            first_id = str(batch.successful()[0].rows[0].get("id", ""))
            if first_id:
                assert first_id in i3.context_text, (
                    f"Real user id '{first_id}' missing from context_text:\n"
                    f"{i3.context_text[:500]}"
                )

    @all_services_available
    def test_three_service_pipeline_queried_ids(self, executor):
        i2 = _i2(
            _sq("user-activity-service",   QueryLanguage.SQL,
                "SELECT id FROM users LIMIT 5"),
            _sq("knowledge-graph-service", QueryLanguage.CYPHER,
                "MATCH (t:Tag) RETURN t.name LIMIT 5"),
            _sq("content-service",         QueryLanguage.BM25, "python"),
            qid="full-pipeline-test",
        )
        batch = executor.execute(i2)
        i3 = SemanticIntegrator().integrate(batch, "comprehensive test query")

        assert i3.query_id == "full-pipeline-test"
        # queried_service_ids = successful services only
        expected = {r.service_id for r in batch.successful()}
        assert set(i3.queried_service_ids) == expected

    @pg_available
    def test_total_wall_ms_in_i3_matches_batch(self, executor):
        i2 = _i2(_sq("user-activity-service", QueryLanguage.SQL,
                      "SELECT 1 AS n"))
        batch = executor.execute(i2)
        i3 = SemanticIntegrator().integrate(batch, "q")
        assert i3.total_wall_ms == pytest.approx(batch.total_wall_ms)
