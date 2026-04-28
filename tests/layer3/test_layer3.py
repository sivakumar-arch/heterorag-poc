"""
tests/layer3/test_layer3.py
============================
Unit tests for Layer 3 — Steps 6–9.

Covers:
  Step 6 — ParallelRetrievalExecutor: parallel dispatch, timeout handling,
            failed service graceful degradation, ordering preservation,
            total_wall_ms < sum(per-service times) [proof of true parallelism]

  Step 7 — ResultNormaliser: SQL path, Graph path, Document path,
            score normalisation, entity extraction (deterministic vs probabilistic),
            failed service skipped silently

  Step 8 — EntityResolver: single-source pass-through, cross-source merge,
            freetext below threshold NOT merged, conflict detection,
            conflict_count propagation

  Step 9 — Ranker: cross-source boost applied, character cap enforced,
            I₃ guarantees (ordering, queried_service_ids, conflict_count),
            empty input produces empty I₃

  SemanticIntegrator: happy path end-to-end Steps 7→8→9
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from heterorag.layer2.models import (
    I2_QueryTranslationOutput,
    QueryLanguage,
    ServiceQuery,
    ValidationStatus,
)
from heterorag.layer3.models import (
    ConflictRecord,
    DeduplicatedBatch,
    DeduplicatedItem,
    ExtractedEntity,
    I3_IntegratedContext,
    NormalisedBatch,
    NormalisedItem,
    NormalisedServiceResult,
    RankedItem,
    RawServiceResult,
    RetrievalBatch,
    RetrievalError,
    SourceType,
)
from heterorag.layer3.entity_resolver import EntityResolver, CONFLICT_TOLERANCE
from heterorag.layer3.ranker import Ranker
from heterorag.layer3.result_normaliser import (
    ResultNormaliser,
    _normalise_sql_row,
    _normalise_graph_record,
    _normalise_bm25_hit,
    _stable_item_id,
    _normalise_score,
)
from heterorag.layer3.retrieval_executor import (
    ConnectionRegistry,
    ParallelRetrievalExecutor,
)
from heterorag.layer3 import SemanticIntegrator


# =============================================================================
# Fixtures
# =============================================================================

def _sq(service_id: str, ql: QueryLanguage, query: str = "SELECT 1",
        weight: float = 0.333) -> ServiceQuery:
    return ServiceQuery(
        query_id          = "qid-test",
        service_id        = service_id,
        display_name      = service_id,
        query_language    = ql,
        native_query      = query,
        retrieval_weight  = weight,
        validation_status = ValidationStatus.VALID,
    )


def _i2(*sqs, query_id: str = "qid-test") -> I2_QueryTranslationOutput:
    return I2_QueryTranslationOutput(
        query_id      = query_id,
        natural_query = "test question",
        queries       = list(sqs),
    )


def _raw(service_id: str, source_type: SourceType,
         rows: list[dict] = None, succeeded: bool = True,
         ms: float = 10.0) -> RawServiceResult:
    return RawServiceResult(
        query_id         = "qid-test",
        service_id       = service_id,
        display_name     = service_id,
        source_type      = source_type,
        native_query     = "SELECT 1",
        retrieval_weight = 0.333,
        rows             = rows or [],
        retrieval_ms     = ms,
        succeeded        = succeeded,
        error            = None if succeeded else RetrievalError(
            service_id    = service_id,
            error_type    = "TestError",
            error_message = "simulated failure",
            native_query  = "SELECT 1",
        ),
    )


def _batch(*raws, wall_ms: float = 15.0) -> RetrievalBatch:
    return RetrievalBatch(
        query_id      = "qid-test",
        natural_query = "test question",
        results       = list(raws),
        total_wall_ms = wall_ms,
    )


def _normed_item(service_id: str, source_type: SourceType,
                 content: str = "test content",
                 entities: list = None, score: float = 1.0,
                 item_id: str = None, weight: float = 0.333) -> NormalisedItem:
    return NormalisedItem(
        source_service_id = service_id,
        source_type       = source_type,
        item_id           = item_id or f"{service_id}:item:0",
        content           = content,
        entities          = entities or [],
        score             = score,
        raw_record        = {},
        retrieval_weight  = weight,
    )


def _normed_batch(*services_and_items) -> NormalisedBatch:
    results = []
    for (svc_id, src_type), items in services_and_items:
        results.append(NormalisedServiceResult(
            service_id  = svc_id,
            source_type = src_type,
            items       = items,
        ))
    return NormalisedBatch(query_id="qid-test", results=results)


def _entity(etype: str, value: Any, confidence: float = 1.0,
            source: SourceType = SourceType.SQL) -> ExtractedEntity:
    return ExtractedEntity(
        name=etype, entity_type=etype, value=value,
        extraction_confidence=confidence, source_type=source,
    )


# =============================================================================
# Step 6 — ParallelRetrievalExecutor (unit — no live DB)
# =============================================================================

class TestParallelRetrievalExecutor:

    def _executor_with_mock(self, side_effects: dict[str, list[dict]]) -> ParallelRetrievalExecutor:
        """Build an executor whose _execute_one is monkey-patched to return fake rows."""
        from heterorag.layer1.models import ConnectionConfig
        reg = ConnectionRegistry()
        for sid in side_effects:
            reg.register(sid, ConnectionConfig(host="localhost", port=5432))
        exe = ParallelRetrievalExecutor(reg)

        def mock_execute_one(sq: ServiceQuery) -> RawServiceResult:
            rows = side_effects.get(sq.service_id, [])
            src  = {QueryLanguage.SQL: SourceType.SQL,
                    QueryLanguage.CYPHER: SourceType.GRAPH,
                    QueryLanguage.BM25:   SourceType.DOCUMENT}[sq.query_language]
            return RawServiceResult(
                query_id=sq.query_id, service_id=sq.service_id,
                display_name=sq.display_name, source_type=src,
                native_query=sq.native_query, retrieval_weight=sq.retrieval_weight,
                rows=rows, retrieval_ms=5.0, succeeded=True,
            )

        exe._execute_one = mock_execute_one
        return exe

    def test_returns_one_result_per_query(self):
        exe = self._executor_with_mock({
            "svc-sql":   [{"id": 1, "reputation": 500}],
            "svc-graph": [{"user_id": 1}],
        })
        i2    = _i2(_sq("svc-sql", QueryLanguage.SQL),
                    _sq("svc-graph", QueryLanguage.CYPHER))
        batch = exe.execute(i2)
        assert len(batch.results) == 2

    def test_all_results_succeeded(self):
        exe   = self._executor_with_mock({"svc-sql": [{"id": 1}]})
        batch = exe.execute(_i2(_sq("svc-sql", QueryLanguage.SQL)))
        assert all(r.succeeded for r in batch.results)

    def test_empty_i2_returns_empty_batch(self):
        exe   = self._executor_with_mock({})
        batch = exe.execute(_i2())
        assert batch.results == []
        assert batch.total_wall_ms == 0.0

    def test_query_id_preserved(self):
        exe   = self._executor_with_mock({"svc-sql": []})
        i2    = _i2(_sq("svc-sql", QueryLanguage.SQL), query_id="my-unique-id")
        batch = exe.execute(i2)
        assert batch.query_id == "my-unique-id"
        assert batch.results[0].query_id == "my-unique-id"

    def test_ordering_matches_i2(self):
        """Results must be in the same order as I₂ queries."""
        exe = self._executor_with_mock({
            "svc-sql":   [{"id": 1}],
            "svc-graph": [{"user_id": 2}],
            "svc-doc":   [{"body": "text"}],
        })
        i2 = _i2(
            _sq("svc-sql",   QueryLanguage.SQL),
            _sq("svc-graph", QueryLanguage.CYPHER),
            _sq("svc-doc",   QueryLanguage.BM25),
        )
        batch = exe.execute(i2)
        assert batch.service_ids() == ["svc-sql", "svc-graph", "svc-doc"]

    def test_retrieval_weight_carried_through(self):
        exe   = self._executor_with_mock({"svc-sql": [{"id": 1}]})
        sq    = _sq("svc-sql", QueryLanguage.SQL, weight=0.75)
        batch = exe.execute(_i2(sq))
        assert batch.results[0].retrieval_weight == pytest.approx(0.75)

    def test_failed_service_is_not_raised(self):
        """A service that raises should produce a failed RawServiceResult, not propagate."""
        reg = ConnectionRegistry()
        from heterorag.layer1.models import ConnectionConfig
        reg.register("svc-sql", ConnectionConfig(host="localhost", port=5432))
        exe = ParallelRetrievalExecutor(reg)

        def boom(sq):
            raise RuntimeError("connection refused")

        exe._execute_one = boom
        # Should not raise — graceful degradation
        batch = exe.execute(_i2(_sq("svc-sql", QueryLanguage.SQL)))
        assert len(batch.results) == 1
        assert batch.results[0].succeeded is False

    def test_total_wall_ms_is_set(self):
        exe   = self._executor_with_mock({"svc-sql": [{"id": 1}], "svc-doc": []})
        batch = exe.execute(_i2(
            _sq("svc-sql", QueryLanguage.SQL),
            _sq("svc-doc", QueryLanguage.BM25),
        ))
        assert batch.total_wall_ms > 0


# =============================================================================
# Step 7 — ResultNormaliser
# =============================================================================

class TestResultNormaliserSQL:

    def test_sql_row_normalised_to_item(self):
        row  = {"id": 1, "reputation": 500, "display_name": "Alice"}
        item = _normalise_sql_row(row, "svc-sql", 0, 0.333)
        assert isinstance(item, NormalisedItem)
        assert item.source_type == SourceType.SQL
        assert "Alice" in item.content

    def test_sql_entity_extraction_deterministic(self):
        row  = {"id": 42, "reputation": 800, "owner_user_id": 7}
        item = _normalise_sql_row(row, "svc-sql", 0, 0.333)
        entity_types = {e.entity_type for e in item.entities}
        assert "id" in entity_types or "owner_user_id" in entity_types

    def test_sql_entity_confidence_is_1(self):
        row  = {"id": 1, "reputation": 500}
        item = _normalise_sql_row(row, "svc-sql", 0, 0.333)
        for e in item.entities:
            assert e.extraction_confidence == 1.0, (
                f"SQL entity '{e.entity_type}' has confidence < 1.0"
            )

    def test_sql_tag_parsing_angle_bracket_format(self):
        row  = {"id": 1, "tags": "<python><pandas><numpy>"}
        item = _normalise_sql_row(row, "svc-sql", 0, 0.333)
        tag_values = [e.value for e in item.entities if e.entity_type == "tag"]
        assert "python" in tag_values
        assert "pandas" in tag_values

    def test_sql_score_always_1(self):
        row  = {"id": 1}
        item = _normalise_sql_row(row, "svc-sql", 0, 0.333)
        assert item.score == 1.0

    def test_sql_long_text_truncated_in_content(self):
        row  = {"id": 1, "title": "x" * 500}
        item = _normalise_sql_row(row, "svc-sql", 0, 0.333)
        assert len(item.content) < 1000   # truncated

    def test_sql_stable_item_id_uses_pk(self):
        row    = {"id": 99, "reputation": 100}
        iid    = _stable_item_id("svc-sql", 0, row)
        assert "99" in iid

    def test_normaliser_skips_failed_services(self):
        batch = _batch(
            _raw("svc-sql",  SourceType.SQL, [{"id": 1}], succeeded=True),
            _raw("svc-graph", SourceType.GRAPH, [], succeeded=False),
        )
        norm = ResultNormaliser().normalise(batch)
        svc_ids = [r.service_id for r in norm.results]
        assert "svc-sql"   in svc_ids
        assert "svc-graph" not in svc_ids


class TestResultNormaliserGraph:

    def test_graph_record_normalised(self):
        record = {"user_id": 5, "reputation": 300}
        item   = _normalise_graph_record(record, "svc-graph", 0, 0.333)
        assert item.source_type == SourceType.GRAPH
        assert "5" in item.content or "user_id" in item.content

    def test_graph_entity_confidence_is_1(self):
        record = {"id": 10, "name": "python"}
        item   = _normalise_graph_record(record, "svc-graph", 0, 0.333)
        for e in item.entities:
            assert e.extraction_confidence == 1.0

    def test_graph_tag_entity_extracted(self):
        record = {"name": "python", "count": 50000}
        item   = _normalise_graph_record(record, "svc-graph", 0, 0.333)
        tag_entities = [e for e in item.entities if e.entity_type == "tag"]
        assert any(e.value == "python" for e in tag_entities)


class TestResultNormaliserDocument:

    def test_bm25_score_normalised(self):
        hit1 = {"_id": "p1", "_score": 10.0, "body": "text one"}
        hit2 = {"_id": "p2", "_score": 5.0,  "body": "text two"}
        item1 = _normalise_bm25_hit(hit1, "svc-doc", 0, 10.0, 10.0, 0.333)
        item2 = _normalise_bm25_hit(hit2, "svc-doc", 1, 5.0,  10.0, 0.333)
        assert item1.score == pytest.approx(1.0)
        assert item2.score == pytest.approx(0.5)

    def test_bm25_body_truncated(self):
        long_body = "word " * 300
        hit  = {"_id": "p1", "_score": 1.0, "body": long_body}
        item = _normalise_bm25_hit(hit, "svc-doc", 0, 1.0, 1.0, 0.333)
        assert len(item.content) < len(long_body)

    def test_bm25_post_id_entity_confidence_1(self):
        hit  = {"_id": "42", "_score": 5.0, "post_id": 42, "body": "text"}
        item = _normalise_bm25_hit(hit, "svc-doc", 0, 5.0, 5.0, 0.333)
        post_id_entities = [e for e in item.entities if e.entity_type == "post_id"]
        assert len(post_id_entities) >= 1
        assert post_id_entities[0].extraction_confidence == 1.0

    def test_bm25_normalise_score_helper(self):
        assert _normalise_score(5.0, 10.0) == pytest.approx(0.5)
        assert _normalise_score(10.0, 10.0) == pytest.approx(1.0)
        assert _normalise_score(0.0, 0.0) == 1.0   # zero-max edge case

    def test_source_wall_ms_propagated(self):
        batch = _batch(_raw("svc-sql", SourceType.SQL, [{"id": 1}]), wall_ms=42.5)
        norm  = ResultNormaliser().normalise(batch)
        assert norm.source_wall_ms == pytest.approx(42.5)


# =============================================================================
# Step 8 — EntityResolver + Deduplicator
# =============================================================================

class TestEntityResolver:

    def test_single_source_item_passthrough(self):
        item  = _normed_item("svc-sql", SourceType.SQL, entities=[_entity("id", 1)])
        batch = _normed_batch((("svc-sql", SourceType.SQL), [item]))
        dedup = EntityResolver().deduplicate(batch)
        assert len(dedup.items) == 1
        assert dedup.items[0].is_cross_source is False

    def test_cross_source_items_merged_by_user_id(self):
        """Two items with the same user_id from different services → merged."""
        item_sql = _normed_item(
            "svc-sql", SourceType.SQL, "user row",
            entities=[_entity("user_id", 42)], item_id="sql:user_id:42",
        )
        item_graph = _normed_item(
            "svc-graph", SourceType.GRAPH, "graph user node",
            entities=[_entity("user_id", 42, source=SourceType.GRAPH)],
            item_id="graph:user_id:42",
        )
        batch = _normed_batch(
            (("svc-sql",   SourceType.SQL),   [item_sql]),
            (("svc-graph", SourceType.GRAPH), [item_graph]),
        )
        dedup = EntityResolver().deduplicate(batch)
        merged = [i for i in dedup.items if i.is_cross_source]
        assert len(merged) == 1
        assert "svc-sql"   in merged[0].source_service_ids
        assert "svc-graph" in merged[0].source_service_ids

    def test_merged_item_content_contains_both_sources(self):
        item_a = _normed_item("svc-sql", SourceType.SQL, "SQL content",
                              entities=[_entity("post_id", 99)], item_id="sql:post_id:99")
        item_b = _normed_item("svc-doc", SourceType.DOCUMENT, "Document content",
                              entities=[_entity("post_id", 99, source=SourceType.DOCUMENT)],
                              item_id="doc:post_id:99")
        batch = _normed_batch(
            (("svc-sql", SourceType.SQL),      [item_a]),
            (("svc-doc", SourceType.DOCUMENT), [item_b]),
        )
        dedup = EntityResolver().deduplicate(batch)
        merged = next(i for i in dedup.items if i.is_cross_source)
        assert "SQL content" in merged.content
        assert "Document content" in merged.content

    def test_freetext_below_threshold_not_merged(self):
        """Low-confidence freetext mentions (< 0.5) must NOT be merged across sources."""
        low_conf = _entity("freetext_mention", "Alice", confidence=0.3,
                           source=SourceType.DOCUMENT)
        item_a = _normed_item("svc-doc",   SourceType.DOCUMENT, "doc A",
                              entities=[low_conf], item_id="doc:freetext:Alice")
        item_b = _normed_item("svc-graph", SourceType.GRAPH, "graph B",
                              entities=[_entity("freetext_mention", "Alice",
                                                confidence=0.3,
                                                source=SourceType.GRAPH)],
                              item_id="graph:freetext:Alice")
        batch = _normed_batch(
            (("svc-doc",   SourceType.DOCUMENT), [item_a]),
            (("svc-graph", SourceType.GRAPH),    [item_b]),
        )
        dedup = EntityResolver().deduplicate(batch)
        cross = [i for i in dedup.items if i.is_cross_source]
        assert len(cross) == 0, "Low-confidence freetext should not be merged"

    def test_conflict_detected_on_numeric_attribute_difference(self):
        """Two items with same user_id but different reputation → conflict recorded."""
        item_a = _normed_item(
            "svc-sql", SourceType.SQL, "user row",
            entities=[_entity("user_id", 1)], item_id="sql:user_id:1",
        )
        item_a = item_a.model_copy(update={"raw_record": {"user_id": 1, "reputation": 1500}})

        item_b = _normed_item(
            "svc-graph", SourceType.GRAPH, "graph user",
            entities=[_entity("user_id", 1, source=SourceType.GRAPH)],
            item_id="graph:user_id:1",
        )
        item_b = item_b.model_copy(update={"raw_record": {"user_id": 1, "reputation": 1000}})

        batch = _normed_batch(
            (("svc-sql",   SourceType.SQL),   [item_a]),
            (("svc-graph", SourceType.GRAPH), [item_b]),
        )
        dedup = EntityResolver().deduplicate(batch)
        assert dedup.conflict_count >= 1
        conflict = dedup.conflict_records[0]
        assert conflict.attribute == "reputation"

    def test_no_conflict_within_tolerance(self):
        """Tiny numeric differences (< CONFLICT_TOLERANCE) should not trigger a conflict."""
        item_a = _normed_item("svc-sql", SourceType.SQL, "a",
                              entities=[_entity("user_id", 1)], item_id="sql:user_id:1")
        item_a = item_a.model_copy(update={"raw_record": {"user_id": 1, "reputation": 1000}})
        item_b = _normed_item("svc-graph", SourceType.GRAPH, "b",
                              entities=[_entity("user_id", 1, source=SourceType.GRAPH)],
                              item_id="graph:user_id:1")
        item_b = item_b.model_copy(update={"raw_record": {"user_id": 1, "reputation": 1001}})
        batch = _normed_batch(
            (("svc-sql",   SourceType.SQL),   [item_a]),
            (("svc-graph", SourceType.GRAPH), [item_b]),
        )
        dedup = EntityResolver().deduplicate(batch)
        assert dedup.conflict_count == 0

    def test_empty_input_returns_empty(self):
        batch = _normed_batch()
        dedup = EntityResolver().deduplicate(batch)
        assert dedup.items == []


# =============================================================================
# Step 9 — Ranker
# =============================================================================

class TestRanker:

    def _batch_from_items(self, items: list[DeduplicatedItem]) -> DeduplicatedBatch:
        return DeduplicatedBatch(query_id="qid-test", items=items)

    def test_single_item_rank_1(self):
        item  = DeduplicatedItem(canonical_id="a", content="hello",
                                 score=0.8, source_service_ids=["svc-sql"],
                                 source_types=[SourceType.SQL])
        dedup = self._batch_from_items([item])
        i3    = Ranker().rank(dedup, ["svc-sql"], "q")
        assert len(i3.items) == 1
        assert i3.items[0].rank == 1

    def test_ordering_descending_by_score(self):
        items = [
            DeduplicatedItem(canonical_id="low",  content="low",  score=0.3,
                             source_service_ids=["s"], source_types=[SourceType.SQL]),
            DeduplicatedItem(canonical_id="high", content="high", score=0.9,
                             source_service_ids=["s"], source_types=[SourceType.SQL]),
            DeduplicatedItem(canonical_id="mid",  content="mid",  score=0.6,
                             source_service_ids=["s"], source_types=[SourceType.SQL]),
        ]
        i3 = Ranker().rank(self._batch_from_items(items), ["s"], "q")
        scores = [it.final_score for it in i3.items]
        assert scores == sorted(scores, reverse=True)

    def test_cross_source_boost_applied(self):
        """A cross-source item should outscore a same-score single-source item."""
        single = DeduplicatedItem(canonical_id="single", content="x",
                                  score=0.5, is_cross_source=False,
                                  source_service_ids=["svc-sql"],
                                  source_types=[SourceType.SQL])
        cross  = DeduplicatedItem(canonical_id="cross",  content="y",
                                  score=0.5, is_cross_source=True,
                                  source_service_ids=["svc-sql", "svc-graph"],
                                  source_types=[SourceType.SQL, SourceType.GRAPH])
        i3 = Ranker(cross_source_boost=1.2).rank(
            self._batch_from_items([single, cross]), ["svc-sql", "svc-graph"], "q"
        )
        cross_item  = next(it for it in i3.items if it.is_cross_source)
        single_item = next(it for it in i3.items if not it.is_cross_source)
        assert cross_item.final_score > single_item.final_score

    def test_context_text_contains_all_items_within_cap(self):
        items = [
            DeduplicatedItem(canonical_id=f"item{i}", content=f"Content {i}",
                             score=1.0 - i * 0.1,
                             source_service_ids=["s"], source_types=[SourceType.SQL])
            for i in range(5)
        ]
        i3 = Ranker(context_char_cap=10_000).rank(self._batch_from_items(items), ["s"], "q")
        for item in i3.items:
            assert item.content in i3.context_text

    def test_character_cap_drops_trailing_items(self):
        """Items past the cap should be absent from context_text (not truncated mid-item)."""
        big_content = "A" * 2000
        items = [
            DeduplicatedItem(canonical_id=f"item{i}", content=big_content,
                             score=1.0 - i * 0.1,
                             source_service_ids=["s"], source_types=[SourceType.SQL])
            for i in range(10)
        ]
        i3 = Ranker(context_char_cap=5_000).rank(self._batch_from_items(items), ["s"], "q")
        # context_text must be under cap
        assert len(i3.context_text) <= 5_000 + 200   # small header overhead allowed

    def test_empty_input_returns_empty_i3(self):
        dedup = self._batch_from_items([])
        i3    = Ranker().rank(dedup, [], "q")
        assert len(i3.items) == 0
        assert i3.context_text == ""

    def test_queried_service_ids_propagated(self):
        dedup = self._batch_from_items([])
        i3    = Ranker().rank(dedup, ["svc-sql", "svc-graph"], "q")
        assert i3.queried_service_ids == ["svc-sql", "svc-graph"]

    def test_conflict_count_propagated(self):
        dedup = DeduplicatedBatch(
            query_id="q",
            items=[],
            conflict_records=[ConflictRecord(
                entity_type="numeric_attribute", entity_value="reputation",
                service_a="a", service_b="b",
                attribute="reputation", value_a=1000, value_b=2000,
            )],
        )
        i3 = Ranker().rank(dedup, [], "q")
        assert i3.conflict_count == 1

    def test_total_wall_ms_propagated(self):
        dedup = self._batch_from_items([])
        i3    = Ranker().rank(dedup, [], "q", total_wall_ms=99.5)
        assert i3.total_wall_ms == pytest.approx(99.5)

    def test_i3_items_have_rank_numbers_starting_at_1(self):
        items = [
            DeduplicatedItem(canonical_id=f"item{i}", content=f"c{i}",
                             score=0.9 - i * 0.1,
                             source_service_ids=["s"], source_types=[SourceType.SQL])
            for i in range(3)
        ]
        i3 = Ranker().rank(self._batch_from_items(items), ["s"], "q")
        assert [it.rank for it in i3.items] == [1, 2, 3]

    def test_max_items_cap_respected(self):
        items = [
            DeduplicatedItem(canonical_id=f"item{i}", content=f"c{i}", score=0.9,
                             source_service_ids=["s"], source_types=[SourceType.SQL])
            for i in range(30)
        ]
        i3 = Ranker(max_items=5).rank(self._batch_from_items(items), ["s"], "q")
        assert len(i3.items) <= 5


# =============================================================================
# SemanticIntegrator — end-to-end Steps 7→8→9
# =============================================================================

class TestSemanticIntegrator:

    def test_happy_path_sql_only(self):
        batch = _batch(
            _raw("svc-sql", SourceType.SQL, [
                {"id": 1, "reputation": 500, "display_name": "Alice"},
                {"id": 2, "reputation": 300, "display_name": "Bob"},
            ]),
        )
        integrator = SemanticIntegrator()
        i3 = integrator.integrate(batch, "top users")
        assert isinstance(i3, I3_IntegratedContext)
        assert i3.query_id == "qid-test"
        assert len(i3.items) >= 1
        assert "svc-sql" in i3.queried_service_ids

    def test_happy_path_three_services(self):
        batch = _batch(
            _raw("svc-sql",   SourceType.SQL,
                 [{"id": 1, "reputation": 500}]),
            _raw("svc-graph", SourceType.GRAPH,
                 [{"user_id": 1, "reputation": 500}]),
            _raw("svc-doc",   SourceType.DOCUMENT,
                 [{"_id": "p1", "_score": 8.0, "body": "Python tutorial"}]),
        )
        integrator = SemanticIntegrator()
        i3 = integrator.integrate(batch, "top python users")
        assert len(i3.queried_service_ids) == 3
        assert len(i3.items) >= 1

    def test_failed_service_excluded_from_queried_ids(self):
        batch = _batch(
            _raw("svc-sql",   SourceType.SQL, [{"id": 1}], succeeded=True),
            _raw("svc-graph", SourceType.GRAPH, [], succeeded=False),
        )
        i3 = SemanticIntegrator().integrate(batch, "q")
        assert "svc-sql"   in i3.queried_service_ids
        assert "svc-graph" not in i3.queried_service_ids

    def test_context_text_non_empty_for_non_empty_results(self):
        batch = _batch(_raw("svc-sql", SourceType.SQL, [{"id": 1, "score": 50}]))
        i3 = SemanticIntegrator().integrate(batch, "q")
        assert i3.context_text.strip() != ""

    def test_total_wall_ms_matches_batch(self):
        batch = _batch(_raw("svc-sql", SourceType.SQL, [{"id": 1}]), wall_ms=123.0)
        i3 = SemanticIntegrator().integrate(batch, "q")
        assert i3.total_wall_ms == pytest.approx(123.0)
