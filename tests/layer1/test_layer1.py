"""
tests/layer1/test_layer1.py
============================
Comprehensive tests for Layer 1 — Service Discovery.

Covers:
  - ServiceDescriptor validation (all schema types, cross-field invariants)
  - ServiceRegistry lifecycle (register, re-register, heartbeat, expiry)
  - Schema drift detection and partial cold-start policy
  - Cold-start uniform weight initialisation
  - RelevanceFilter Stage 1 (embedding similarity) and Stage 2 (LLM confirmation)
  - I₁ producer guarantees
  - POC descriptors correctness
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from heterorag.layer1.models import (
    ColumnSpec,
    ConnectionConfig,
    DocumentFieldSpec,
    DocumentSchemaSpec,
    DocumentTypeSpec,
    EdgeTypeSpec,
    GraphSchemaSpec,
    I1_ServiceDiscoveryOutput,
    NodeTypeSpec,
    PersistenceType,
    SQLSchemaSpec,
    ServiceDescriptor,
    TableSpec,
)
from heterorag.layer1.poc_descriptors import (
    CONTENT_SERVICE_DESCRIPTOR,
    KNOWLEDGE_GRAPH_DESCRIPTOR,
    USER_ACTIVITY_DESCRIPTOR,
    build_poc_registry,
)
from heterorag.layer1.registry import RelevanceFilter, ServiceRegistry, _unit_norm


# =============================================================================
# Fixtures — minimal valid descriptors for each persistence type
# =============================================================================

def _make_connection() -> ConnectionConfig:
    return ConnectionConfig(host="localhost", port=5432, database="test")


def _make_sql_descriptor(service_id: str = "svc-sql", schema_version: str = "1.0") -> ServiceDescriptor:
    return ServiceDescriptor(
        service_id       = service_id,
        display_name     = "Test SQL Service",
        persistence_type = PersistenceType.SQL,
        schema_spec      = SQLSchemaSpec(
            tables=[TableSpec(
                table_name="users",
                columns=[ColumnSpec(name="id", data_type="integer")],
            )]
        ),
        schema_version   = schema_version,
        connection       = _make_connection(),
    )


def _make_graph_descriptor(service_id: str = "svc-graph") -> ServiceDescriptor:
    return ServiceDescriptor(
        service_id       = service_id,
        display_name     = "Test Graph Service",
        persistence_type = PersistenceType.GRAPH,
        schema_spec      = GraphSchemaSpec(
            node_types=[NodeTypeSpec(label="User", properties=["id"])],
            edge_types=[EdgeTypeSpec(edge_type="KNOWS", from_label="User", to_label="User")],
        ),
        schema_version   = "1.0",
        connection       = _make_connection(),
    )


def _make_document_descriptor(service_id: str = "svc-doc") -> ServiceDescriptor:
    return ServiceDescriptor(
        service_id       = service_id,
        display_name     = "Test Document Service",
        persistence_type = PersistenceType.DOCUMENT,
        schema_spec      = DocumentSchemaSpec(
            index_name = "test_index",
            doc_types  = [DocumentTypeSpec(
                doc_type = "article",
                fields   = [DocumentFieldSpec(name="body", field_type="text")],
            )],
        ),
        schema_version   = "1.0",
        connection       = _make_connection(),
    )


def _make_unit_vec(dim: int = 4, direction: int = 0) -> list[float]:
    """Returns a unit vector with 1.0 at the given dimension index."""
    v = [0.0] * dim
    v[direction] = 1.0
    return v


# =============================================================================
# Model validation tests
# =============================================================================

class TestServiceDescriptorValidation:

    def test_valid_sql_descriptor(self):
        d = _make_sql_descriptor()
        assert d.service_id == "svc-sql"
        assert d.persistence_type == PersistenceType.SQL

    def test_valid_graph_descriptor(self):
        d = _make_graph_descriptor()
        assert d.persistence_type == PersistenceType.GRAPH

    def test_valid_document_descriptor(self):
        d = _make_document_descriptor()
        assert d.persistence_type == PersistenceType.DOCUMENT

    def test_persistence_type_schema_mismatch_raises(self):
        """persistence_type=GRAPH but schema_spec is SQLSchemaSpec → error."""
        with pytest.raises(Exception, match="persistence_type"):
            ServiceDescriptor(
                service_id       = "bad",
                display_name     = "Bad",
                persistence_type = PersistenceType.GRAPH,   # wrong
                schema_spec      = SQLSchemaSpec(
                    tables=[TableSpec(
                        table_name="t",
                        columns=[ColumnSpec(name="id", data_type="integer")],
                    )]
                ),
                schema_version   = "1.0",
                connection       = _make_connection(),
            )

    def test_capability_summary_token_limit_enforced(self):
        """A summary with > 200 words should be rejected."""
        too_long = " ".join(["word"] * 250)
        with pytest.raises(Exception, match="token limit"):
            _make_sql_descriptor().__class__(
                **_make_sql_descriptor().model_dump() | {"capability_summary": too_long}
            )

    def test_capability_summary_within_limit_accepted(self):
        short = " ".join(["word"] * 50)
        d = ServiceDescriptor(
            **_make_sql_descriptor().model_dump() | {"capability_summary": short}
        )
        assert d.capability_summary == short

    def test_embedding_without_summary_raises(self):
        with pytest.raises(Exception, match="capability_summary"):
            ServiceDescriptor(
                **_make_sql_descriptor().model_dump() | {
                    "capability_embedding": _make_unit_vec(),
                    "capability_summary":   None,
                }
            )

    def test_non_unit_norm_embedding_raises(self):
        not_unit = [0.5, 0.5, 0.5, 0.5]  # L2 = 1.0 / sqrt(1) = 1, actually this IS unit
        # build one that's clearly not unit
        bad = [2.0, 0.0, 0.0, 0.0]       # L2 = 2.0
        with pytest.raises(Exception, match="unit-norm"):
            ServiceDescriptor(
                **_make_sql_descriptor().model_dump() | {
                    "capability_summary":   "some summary",
                    "capability_embedding": bad,
                }
            )

    def test_unit_norm_embedding_accepted(self):
        emb = _make_unit_vec(4, 0)
        d = ServiceDescriptor(
            **_make_sql_descriptor().model_dump() | {
                "capability_summary":   "some summary",
                "capability_embedding": emb,
            }
        )
        assert d.capability_embedding == emb

    def test_empty_service_id_raises(self):
        with pytest.raises(Exception):
            _make_sql_descriptor(service_id="")

    def test_empty_tables_raises(self):
        with pytest.raises(Exception, match="at least one table"):
            SQLSchemaSpec(tables=[])

    def test_empty_node_types_raises(self):
        with pytest.raises(Exception, match="at least one node"):
            GraphSchemaSpec(node_types=[], edge_types=[])

    def test_schema_version_changed(self):
        d = _make_sql_descriptor(schema_version="1.0")
        assert d.schema_version_changed("2.0") is True
        assert d.schema_version_changed("1.0") is False


# =============================================================================
# I₁ interface guarantee tests
# =============================================================================

class TestI1Interface:

    def test_valid_i1(self):
        out = I1_ServiceDiscoveryOutput(
            query_id    = "q1",
            descriptors = [_make_sql_descriptor(), _make_graph_descriptor()],
        )
        assert len(out) == 2

    def test_duplicate_service_ids_rejected(self):
        with pytest.raises(Exception, match="duplicate service_ids"):
            I1_ServiceDiscoveryOutput(
                query_id    = "q1",
                descriptors = [_make_sql_descriptor("same"), _make_sql_descriptor("same")],
            )

    def test_inactive_service_rejected(self):
        d = _make_sql_descriptor()
        d = d.model_copy(update={"is_active": False})
        with pytest.raises(Exception, match="inactive"):
            I1_ServiceDiscoveryOutput(query_id="q1", descriptors=[d])

    def test_service_ids_method(self):
        d1 = _make_sql_descriptor("a")
        d2 = _make_graph_descriptor("b")
        out = I1_ServiceDiscoveryOutput(query_id="q1", descriptors=[d1, d2])
        assert out.service_ids() == ["a", "b"]

    def test_produced_at_is_utc(self):
        out = I1_ServiceDiscoveryOutput(query_id="q1", descriptors=[])
        assert out.produced_at.tzinfo is not None


# =============================================================================
# ServiceRegistry lifecycle tests
# =============================================================================

class TestServiceRegistryLifecycle:

    def _registry(self) -> ServiceRegistry:
        return ServiceRegistry(heartbeat_timeout_seconds=2)

    def test_register_and_retrieve(self):
        r = self._registry()
        d = _make_sql_descriptor()
        r.register(d)
        retrieved = r.get("svc-sql")
        assert retrieved is not None
        assert retrieved.service_id == "svc-sql"
        assert retrieved.is_active is True

    def test_registered_at_is_stamped(self):
        r = self._registry()
        r.register(_make_sql_descriptor())
        d = r.get("svc-sql")
        assert d.registered_at is not None

    def test_deregister_marks_inactive(self):
        r = self._registry()
        r.register(_make_sql_descriptor())
        r.deregister("svc-sql")
        assert r.get("svc-sql").is_active is False

    def test_len_counts_only_active(self):
        r = self._registry()
        r.register(_make_sql_descriptor("a"))
        r.register(_make_sql_descriptor("b"))
        assert len(r) == 2
        r.deregister("a")
        assert len(r) == 1

    def test_heartbeat_updates_timestamp(self):
        r = self._registry()
        r.register(_make_sql_descriptor())
        before = r._entries["svc-sql"].last_heartbeat
        time.sleep(0.01)
        r.heartbeat("svc-sql")
        after = r._entries["svc-sql"].last_heartbeat
        assert after > before

    def test_stale_service_expires(self):
        r = ServiceRegistry(heartbeat_timeout_seconds=1)
        r.register(_make_sql_descriptor())
        time.sleep(1.1)
        expired = r.expire_stale_services()
        assert "svc-sql" in expired
        assert r.get("svc-sql").is_active is False

    def test_active_service_not_expired(self):
        r = ServiceRegistry(heartbeat_timeout_seconds=5)
        r.register(_make_sql_descriptor())
        expired = r.expire_stale_services()
        assert expired == []
        assert r.get("svc-sql").is_active is True

    def test_unknown_heartbeat_logged_not_raised(self):
        r = self._registry()
        r.heartbeat("does-not-exist")   # should not raise


# =============================================================================
# Cold-start policy tests
# =============================================================================

class TestColdStartPolicy:

    def test_initial_weight_is_zero(self):
        r = ServiceRegistry()
        r.register(_make_sql_descriptor())
        assert r.get("svc-sql").retrieval_weight == 0.0

    def test_first_discover_triggers_cold_start(self):
        r = ServiceRegistry()
        r.register(_make_sql_descriptor("a"))
        r.register(_make_graph_descriptor("b"))
        r.discover("q1", "some query")
        assert r.get("a").retrieval_weight == pytest.approx(0.5)
        assert r.get("b").retrieval_weight == pytest.approx(0.5)

    def test_cold_start_only_triggers_once(self):
        r = ServiceRegistry()
        r.register(_make_sql_descriptor())
        r.discover("q1", "query one")
        weight_after_first = r.get("svc-sql").retrieval_weight
        # Simulate planner updating weight
        r.update_weight("svc-sql", 0.9)
        r.discover("q2", "query two")
        # Weight should NOT revert to uniform on second query
        assert r.get("svc-sql").retrieval_weight == pytest.approx(0.9)

    def test_update_weight_out_of_range_raises(self):
        r = ServiceRegistry()
        r.register(_make_sql_descriptor())
        with pytest.raises(ValueError):
            r.update_weight("svc-sql", 1.5)

    def test_update_weight_unknown_service_raises(self):
        r = ServiceRegistry()
        with pytest.raises(KeyError):
            r.update_weight("nonexistent", 0.5)


# =============================================================================
# Schema drift and partial cold-start tests
# =============================================================================

class TestSchemaDrift:

    def test_reregister_same_version_preserves_weight(self):
        r = ServiceRegistry()
        r.register(_make_sql_descriptor(schema_version="1.0"))
        r.update_weight("svc-sql", 0.8)
        # Re-register with the same version
        r.register(_make_sql_descriptor(schema_version="1.0"))
        assert r.get("svc-sql").retrieval_weight == pytest.approx(0.8)

    def test_reregister_new_version_triggers_partial_cold_start(self):
        r = ServiceRegistry()
        r.register(_make_sql_descriptor("svc-sql", schema_version="1.0"))
        r.register(_make_graph_descriptor("svc-graph"))
        # Simulate planner learning weights
        r.update_weight("svc-sql", 0.8)
        r.update_weight("svc-graph", 0.2)
        # SQL service schema changes
        r.register(_make_sql_descriptor("svc-sql", schema_version="2.0"))
        # svc-sql should revert to uniform (1/2 = 0.5)
        assert r.get("svc-sql").retrieval_weight == pytest.approx(0.5)
        # svc-graph should keep its learned weight
        assert r.get("svc-graph").retrieval_weight == pytest.approx(0.2)

    def test_reregister_new_version_preserves_registered_at(self):
        r = ServiceRegistry()
        r.register(_make_sql_descriptor(schema_version="1.0"))
        original_time = r.get("svc-sql").registered_at
        time.sleep(0.01)
        r.register(_make_sql_descriptor(schema_version="2.0"))
        assert r.get("svc-sql").registered_at == original_time


# =============================================================================
# RelevanceFilter tests
# =============================================================================

class TestRelevanceFilter:

    def _descriptors_with_embeddings(self) -> list[ServiceDescriptor]:
        """Three descriptors with orthogonal unit-vector embeddings."""
        descs = []
        for i, (sid, direction) in enumerate([("sql", 0), ("graph", 1), ("doc", 2)]):
            emb = _make_unit_vec(3, direction)
            raw = _make_sql_descriptor(sid)
            d = ServiceDescriptor(
                **raw.model_dump() | {
                    "capability_summary":   f"Summary for {sid}",
                    "capability_embedding": emb,
                }
            )
            descs.append(d)
        return descs

    def test_no_embed_fn_passes_all_through(self):
        f = RelevanceFilter(embed_fn=None, llm_confirm_fn=None)
        descriptors = [_make_sql_descriptor(), _make_graph_descriptor()]
        result = f.filter("any query", descriptors)
        assert len(result) == 2

    def test_stage1_selects_most_similar(self):
        descs = self._descriptors_with_embeddings()

        # Query embedding points toward "sql" (direction 0)
        def embed_fn(text: str) -> list[float]:
            return [1.0, 0.0, 0.0]   # already unit norm

        f = RelevanceFilter(embed_fn=embed_fn, llm_confirm_fn=None, stage1_top_k=1)
        result = f.filter("query", descs)
        assert len(result) == 1
        assert result[0].service_id == "sql"

    def test_stage1_top_k_respected(self):
        descs = self._descriptors_with_embeddings()

        def embed_fn(text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

        f = RelevanceFilter(embed_fn=embed_fn, llm_confirm_fn=None, stage1_top_k=2)
        result = f.filter("query", descs)
        assert len(result) == 2

    def test_stage2_llm_confirm_filters_further(self):
        descs = [_make_sql_descriptor("a"), _make_graph_descriptor("b")]

        def embed_fn(text: str) -> list[float]:
            return [1.0]

        # LLM confirms only "a"
        def llm_confirm(query: str, descriptors: list) -> list[str]:
            return ["a"]

        f = RelevanceFilter(embed_fn=embed_fn, llm_confirm_fn=llm_confirm, stage1_top_k=10)
        result = f.filter("query", descs)
        assert len(result) == 1
        assert result[0].service_id == "a"

    def test_embed_fn_failure_falls_back_to_all(self):
        def bad_embed(text: str) -> list[float]:
            raise RuntimeError("embedding service down")

        f = RelevanceFilter(embed_fn=bad_embed, llm_confirm_fn=None)
        descs = [_make_sql_descriptor(), _make_graph_descriptor()]
        result = f.filter("query", descs)
        assert len(result) == 2     # graceful fallback — all pass through

    def test_descriptors_without_embedding_pass_stage1(self):
        """Services without an embedding are treated as similarity=1.0 in Stage 1."""
        no_emb = _make_sql_descriptor("no-emb")
        has_emb_raw = _make_graph_descriptor("has-emb")
        has_emb = ServiceDescriptor(
            **has_emb_raw.model_dump() | {
                "capability_summary":   "graph service",
                "capability_embedding": [0.0, 1.0],   # unit norm, dim 2
            }
        )

        def embed_fn(text: str) -> list[float]:
            return [1.0, 0.0]   # points away from has_emb

        f = RelevanceFilter(embed_fn=embed_fn, llm_confirm_fn=None, stage1_top_k=2)
        result = f.filter("query", [no_emb, has_emb])
        ids = [d.service_id for d in result]
        assert "no-emb" in ids   # passes through conservatively


# =============================================================================
# discover() integration tests
# =============================================================================

class TestDiscover:

    def test_discover_returns_i1(self):
        r = ServiceRegistry()
        r.register(_make_sql_descriptor())
        i1 = r.discover("qid-001", "how many users have reputation > 1000?")
        assert isinstance(i1, I1_ServiceDiscoveryOutput)
        assert i1.query_id == "qid-001"

    def test_discover_empty_registry(self):
        r = ServiceRegistry()
        i1 = r.discover("qid-000", "anything")
        assert i1.descriptors == []

    def test_discover_excludes_inactive(self):
        r = ServiceRegistry()
        r.register(_make_sql_descriptor("active"))
        r.register(_make_graph_descriptor("inactive"))
        r.deregister("inactive")
        i1 = r.discover("q1", "query")
        assert all(d.service_id != "inactive" for d in i1.descriptors)

    def test_discover_with_llm_filter(self):
        r = ServiceRegistry()
        r.register(_make_sql_descriptor("sql"))
        r.register(_make_graph_descriptor("graph"))
        r.register(_make_document_descriptor("doc"))

        # LLM only confirms sql
        r._filter._llm_confirm_fn = lambda q, ds: ["sql"]

        i1 = r.discover("q1", "top users by reputation")
        assert i1.service_ids() == ["sql"]


# =============================================================================
# POC descriptor correctness tests
# =============================================================================

class TestPOCDescriptors:

    def test_user_activity_descriptor_valid(self):
        d = USER_ACTIVITY_DESCRIPTOR
        assert d.service_id == "user-activity-service"
        assert d.persistence_type == PersistenceType.SQL
        assert d.schema_spec.persistence_type == PersistenceType.SQL
        assert len(d.schema_spec.tables) == 7   # users, posts, votes, badges, tags, comments, post_links

    def test_knowledge_graph_descriptor_valid(self):
        d = KNOWLEDGE_GRAPH_DESCRIPTOR
        assert d.service_id == "knowledge-graph-service"
        assert d.persistence_type == PersistenceType.GRAPH
        node_labels = {n.label for n in d.schema_spec.node_types}
        assert node_labels == {"User", "Question", "Answer", "Tag"}
        edge_types = {e.edge_type for e in d.schema_spec.edge_types}
        assert "CO_OCCURS_WITH" in edge_types
        assert "DUPLICATE_OF" in edge_types
        assert "TAGGED_WITH" in edge_types

    def test_content_service_descriptor_valid(self):
        d = CONTENT_SERVICE_DESCRIPTOR
        assert d.service_id == "content-service"
        assert d.persistence_type == PersistenceType.DOCUMENT
        doc_types = {dt.doc_type for dt in d.schema_spec.doc_types}
        assert doc_types == {"question", "answer", "comment", "user_about", "tag_wiki"}

    def test_capability_summaries_within_token_limit(self):
        for d in [USER_ACTIVITY_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR, CONTENT_SERVICE_DESCRIPTOR]:
            if d.capability_summary:
                tokens = len(d.capability_summary.split())
                assert tokens <= 200, (
                    f"{d.service_id} capability_summary has {tokens} tokens (limit 200)"
                )

    def test_persistence_type_schema_consistency_all_three(self):
        for d in [USER_ACTIVITY_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR, CONTENT_SERVICE_DESCRIPTOR]:
            assert d.persistence_type.value == d.schema_spec.persistence_type.value

    def test_build_poc_registry(self):
        r = build_poc_registry()
        assert len(r) == 3
        ids = {d.service_id for d in r.all_descriptors()}
        assert ids == {"user-activity-service", "knowledge-graph-service", "content-service"}

    def test_build_poc_registry_with_confirm_fn(self):
        # LLM only selects SQL for a structured query
        def confirm(query: str, descriptors: list) -> list[str]:
            return ["user-activity-service"]

        r = build_poc_registry(llm_confirm_fn=confirm)
        i1 = r.discover("q1", "what is the highest reputation user?")
        assert i1.service_ids() == ["user-activity-service"]

    def test_all_poc_descriptors_have_connection_config(self):
        for d in [USER_ACTIVITY_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR, CONTENT_SERVICE_DESCRIPTOR]:
            assert isinstance(d.connection, ConnectionConfig)
            assert d.connection.host != ""
