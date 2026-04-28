"""
heterorag/layer3/models.py
==========================
Data models for Layer 3 — Semantic Integration.

Defines every intermediate and final type that flows through Steps 6-9:
  Step 6 — RawServiceResult      (Parallel Retrieval Executor output)
  Step 7 — NormalisedResult      (ResultNormaliser output)
  Step 8 — DeduplicatedResult    (EntityResolver + Deduplicator output)
  Step 9 — RankedContext         (Ranker output → feeds Layer 4)
  Final  — I3_IntegratedContext  (formal I₃ interface, consumed by Layer 4)

Design decisions from Foundation Doc v1.6 §6:
  - NormalisedContent asymmetry: SQL + Graph entity extraction is deterministic;
    Document entity extraction is probabilistic (NER). Deduplication confidence
    weights cross-source entity matches by source type.
  - Retrieval Latency (RL) is wall-clock from Layer 1 entry to I₃ delivery.
    Each RawServiceResult carries its own retrieval_ms for per-service breakdown.
  - ICR (Integration Conflict Rate) is recorded at the deduplication step.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Source type — mirrors PersistenceType but lives in Layer 3 for independence
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    SQL      = "sql"
    GRAPH    = "graph"
    DOCUMENT = "document"


# =============================================================================
# Step 6 — Raw retrieval results (one per service, parallel)
# =============================================================================

class RetrievalError(BaseModel):
    """Records a retrieval failure without aborting the whole pipeline."""
    service_id:    str
    error_type:    str       # e.g. "ExecutionError", "Timeout", "ConnectionError"
    error_message: str
    native_query:  str       # the query that failed — for debugging


class RawServiceResult(BaseModel):
    """
    Raw, unprocessed results from one service execution.

    SQL    → rows: list of dicts (column→value)
    Graph  → rows: list of dicts (variable→value, may include node/rel maps)
    BM25   → rows: list of dicts (hit metadata + _source fields)

    retrieval_ms is the wall-clock time for just this service's query execution,
    measured inside the executor thread. Excludes connection setup overhead so
    results are comparable across runs.
    """
    query_id:         str
    service_id:       str
    display_name:     str
    source_type:      SourceType
    native_query:     str
    retrieval_weight: float
    rows:             list[dict[str, Any]] = Field(default_factory=list)
    retrieval_ms:     float = 0.0          # per-service wall-clock execution time
    error:            RetrievalError | None = None
    succeeded:        bool = True


class RetrievalBatch(BaseModel):
    """
    The complete output of the Parallel Retrieval Executor (Step 6).

    Contains one RawServiceResult per service in I₂.
    Failed services have succeeded=False and a populated error field —
    they are not removed here; ResultNormaliser decides what to do with them.

    total_wall_ms is the wall-clock time for the entire parallel batch
    (≈ max(per-service times) since execution is truly parallel).
    This is the primary latency number reported in the paper.
    """
    query_id:       str
    natural_query:  str
    results:        list[RawServiceResult]
    total_wall_ms:  float = 0.0            # wall-clock for full parallel batch
    started_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def successful(self) -> list[RawServiceResult]:
        return [r for r in self.results if r.succeeded]

    def failed(self) -> list[RawServiceResult]:
        return [r for r in self.results if not r.succeeded]

    def service_ids(self) -> list[str]:
        return [r.service_id for r in self.results]


# =============================================================================
# Step 7 — Normalised results (one per service, after ResultNormaliser)
# =============================================================================

class ExtractedEntity(BaseModel):
    """
    A named entity extracted from a result item.

    extraction_confidence reflects the asymmetry from Foundation Doc §6:
      - SQL/Graph: 1.0 (deterministic — comes directly from structured fields)
      - Document:  0.0–1.0 (probabilistic NER — spaCy confidence score)

    This confidence is propagated to the EntityResolver so cross-source
    deduplication can downweight uncertain Document entities.
    """
    name:                   str
    entity_type:            str    # "user_id", "post_id", "tag", "freetext_mention"
    value:                  Any    # the canonical value (int id, str name, etc.)
    extraction_confidence:  float = Field(ge=0.0, le=1.0, default=1.0)
    source_type:            SourceType


class NormalisedItem(BaseModel):
    """
    One normalised result item from any persistence type.

    content is a flat string representation suitable for LLM context injection.
    entities is the list of extracted entities for cross-source deduplication.
    score is a normalised [0,1] relevance score:
      - SQL:    1.0 (all rows equally relevant — no ranking from SQL)
      - Graph:  1.0 for traversal results; NDCG-rank for algorithm results
      - BM25:   Elasticsearch _score normalised to [0,1] within this result set
    """
    source_service_id:  str
    source_type:        SourceType
    item_id:            str           # unique within this result set
    content:            str           # flat text representation
    entities:           list[ExtractedEntity] = Field(default_factory=list)
    score:              float = Field(ge=0.0, le=1.0, default=1.0)
    raw_record:         dict[str, Any] = Field(default_factory=dict)
    retrieval_weight:   float = Field(ge=0.0, le=1.0, default=1.0)


class NormalisedServiceResult(BaseModel):
    """Normalised results from one service."""
    service_id:   str
    source_type:  SourceType
    items:        list[NormalisedItem]
    item_count:   int = 0

    def model_post_init(self, __context: Any) -> None:
        self.item_count = len(self.items)


class NormalisedBatch(BaseModel):
    """Output of Step 7 (ResultNormaliser) — all services normalised."""
    query_id:  str
    results:   list[NormalisedServiceResult]
    source_wall_ms: float = 0.0    # propagated from RetrievalBatch

    def all_items(self) -> list[NormalisedItem]:
        return [item for r in self.results for item in r.items]


# =============================================================================
# Step 8 — Deduplicated results (after EntityResolver + Deduplicator)
# =============================================================================

class ConflictRecord(BaseModel):
    """
    Records a detected conflict between entity references across services.
    Used to compute ICR (Integration Conflict Rate) metric.

    A conflict is when two services return data about the same real-world entity
    (same user_id, same post_id) but with different attribute values
    (e.g. SQL says reputation=1500, Graph says reputation=1450).
    """
    entity_type:       str
    entity_value:      Any
    service_a:         str
    service_b:         str
    attribute:         str
    value_a:           Any
    value_b:           Any


class DeduplicatedItem(BaseModel):
    """A deduplicated result item, possibly merged from multiple sources."""
    canonical_id:      str
    content:           str
    entities:          list[ExtractedEntity] = Field(default_factory=list)
    score:             float = Field(ge=0.0, le=1.0, default=1.0)
    source_service_ids: list[str] = Field(default_factory=list)
    source_types:       list[SourceType] = Field(default_factory=list)
    is_cross_source:   bool = False    # True if merged from ≥2 services


class DeduplicatedBatch(BaseModel):
    """Output of Step 8 (EntityResolver + Deduplicator)."""
    query_id:          str
    items:             list[DeduplicatedItem]
    conflict_records:  list[ConflictRecord] = Field(default_factory=list)
    source_wall_ms:    float = 0.0

    @property
    def conflict_count(self) -> int:
        return len(self.conflict_records)


# =============================================================================
# Step 9 / I₃ — Ranked and integrated context (Ranker output → Layer 4)
# =============================================================================

class RankedItem(BaseModel):
    """One item in the final ranked context, ready for LLM injection."""
    rank:              int
    content:           str
    source_service_ids: list[str]
    source_types:       list[SourceType]
    final_score:       float
    is_cross_source:   bool = False


class I3_IntegratedContext(BaseModel):
    """
    Formal interface I₃: the output of Layer 3 (Semantic Integration).
    Consumed by Layer 4 (Generation).

    I₃ producer guarantees:
      1. items are ordered by final_score descending (rank 1 = most relevant).
      2. context_text is the concatenation of item.content strings in rank order,
         formatted for direct injection into the Layer 4 generation prompt.
      3. query_id links I₃ to the I₁ and I₂ that produced it.
      4. total_wall_ms is the full pipeline latency (Layer 1 entry → I₃ delivery).
         This is the RL metric reported in the paper.
      5. queried_service_ids records which services were actually queried
         (for Source Coverage computation in the benchmark).
      6. conflict_count is the raw ICR numerator for this query.
    """
    query_id:             str
    natural_query:        str
    items:                list[RankedItem]
    context_text:         str            # formatted for Layer 4 prompt
    queried_service_ids:  list[str]      # for Source Coverage (SC) metric
    total_wall_ms:        float = 0.0    # Retrieval Latency (RL) metric
    conflict_count:       int   = 0      # ICR numerator
    produced_at:          datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __len__(self) -> int:
        return len(self.items)
