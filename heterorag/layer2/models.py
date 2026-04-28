"""
heterorag/layer2/models.py
==========================
Formal data model for Layer 2 — Query Translation.

Defines the I₂ interface (List[ServiceQuery]) and all supporting types.

I₂ producer guarantees (enforced by QueryValidator before I₂ is constructed):
  1. Every ServiceQuery.native_query is syntactically valid for its persistence type.
     - SQL:      parseable by sqlglot (dialect=postgres)
     - Cypher:   parseable by the neo4j driver's plan endpoint
     - BM25:     always valid (freeform text) — QTSR for Document is always 1.0
  2. No ServiceQuery survives double validation failure — the service is dropped
     from I₂ entirely and the drop is recorded in validation_log.
  3. query_id matches the I₁ query_id that produced this I₂.
  4. produced_at timestamp is set by the QueryPlanner, not the translator.

Reference: Foundation Doc v1.6 §6 (QueryValidator policy, I₂ interface).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Query language taxonomy
# ---------------------------------------------------------------------------

class QueryLanguage(str, Enum):
    SQL    = "sql"      # PostgreSQL dialect
    CYPHER = "cypher"   # Neo4j Cypher
    BM25   = "bm25"     # Elasticsearch BM25 free-text query


# ---------------------------------------------------------------------------
# Validation outcome
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    VALID          = "valid"           # passed syntactic validation
    FIXED          = "fixed"           # failed first check, LLM retry succeeded
    DROPPED        = "dropped"         # failed both attempts — service excluded from I₂
    SKIPPED        = "skipped"         # BM25 — no syntactic validation needed


class ValidationRecord(BaseModel):
    """Audit trail entry for one validation attempt on one service."""
    service_id:       str
    attempt:          int              # 1 = initial, 2 = LLM retry
    status:           ValidationStatus
    query_before:     str              # query text submitted for validation
    query_after:      str | None       # corrected query (if LLM retry succeeded)
    error_message:    str | None       # parse error on failure
    duration_ms:      float = 0.0


# ---------------------------------------------------------------------------
# ServiceQuery — one translated, validated query for one service
# ---------------------------------------------------------------------------

class ServiceQuery(BaseModel):
    """
    A translated, validated native query for one microservice.

    This is the atomic unit of I₂. Each entry in I₂ corresponds to one service
    that (a) was selected by the QueryPlanner and (b) produced a syntactically
    valid query after at most one LLM retry.

    The retrieval_weight carries the planner's weight from the ServiceDescriptor
    so Layer 3 (Retrieval Executor) can weight results without calling back to Layer 1.
    """
    # --- Linkage ---
    query_id:          str    # matches I₁ query_id
    service_id:        str    # matches ServiceDescriptor.service_id
    display_name:      str    # human-readable service name for logging

    # --- Query ---
    query_language:    QueryLanguage
    native_query:      str    # the translated, validated native query string

    # --- Weighting (from QueryPlanner) ---
    retrieval_weight:  float = Field(ge=0.0, le=1.0)

    # --- Provenance ---
    validation_status: ValidationStatus
    translation_prompt_tokens: int = 0    # token count of the LLM prompt used
    translation_reply_tokens:  int = 0    # token count of the LLM reply


# ---------------------------------------------------------------------------
# I₂ — formal interface between Layer 2 and Layer 3
# ---------------------------------------------------------------------------

class I2_QueryTranslationOutput(BaseModel):
    """
    Formal interface I₂: the output of Layer 2 (Query Translation).

    Consumed by Layer 3 (Parallel Retrieval Executor).

    Producer guarantees:
      1. All ServiceQuery entries have syntactically valid native_query strings.
      2. No dropped services appear — only VALID or FIXED or SKIPPED entries.
      3. query_id matches the I₁ that seeded this translation pass.
      4. validation_log contains one record per service per attempt (including drops).
      5. weights need not sum to 1.0 — normalisation happens in Layer 3.
    """
    query_id:        str
    natural_query:   str                        # the original user NL question
    queries:         list[ServiceQuery]         # one entry per selected + valid service
    validation_log:  list[ValidationRecord] = Field(default_factory=list)
    produced_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Dropped services are not in queries[] but are recorded here for QTSR metric
    dropped_service_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_dropped_in_queries(self) -> "I2_QueryTranslationOutput":
        bad = [q.service_id for q in self.queries
               if q.validation_status == ValidationStatus.DROPPED]
        if bad:
            raise ValueError(
                f"I₂ must not contain DROPPED queries — "
                f"remove them before constructing I₂: {bad}"
            )
        return self

    def service_ids(self) -> list[str]:
        return [q.service_id for q in self.queries]

    def query_for(self, service_id: str) -> ServiceQuery | None:
        return next((q for q in self.queries if q.service_id == service_id), None)

    def __len__(self) -> int:
        return len(self.queries)
