"""
heterorag/layer1/models.py
==========================
Formal data model for Layer 1 — Service Discovery.

This module defines the typed ServiceDescriptor and its persistence-type-specific
SchemaSpec variants.  These types are the authoritative representation of what each
microservice owns and how it can be queried.

Interface contract (I₁ producer guarantee):
  - Every ServiceDescriptor returned by the registry has a non-empty service_id,
    a valid PersistenceType, a fully populated SchemaSpec of the correct subtype,
    and a schema_version string.
  - capability_summary, when present, is ≤ 200 tokens (enforced at registration).
  - capability_embedding, when present, is a unit-norm float list ready for cosine
    similarity comparison in the two-stage RelevanceFilter.

Reference: Foundation Document v1.6, §6 (Locked Design Decisions OQ1–OQ3),
           Architecture Design chat (I₁ formal interface specification).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Persistence type taxonomy
# ---------------------------------------------------------------------------

class PersistenceType(str, Enum):
    """The three structurally distinct persistence types in the POC mesh."""
    SQL      = "sql"
    GRAPH    = "graph"
    DOCUMENT = "document"


# ---------------------------------------------------------------------------
# Persistence-type-specific SchemaSpec variants
# ---------------------------------------------------------------------------

class ColumnSpec(BaseModel):
    """A single column in a SQL table."""
    name:        str
    data_type:   str                          # e.g. "integer", "text", "timestamptz"
    nullable:    bool = True
    description: str | None = None


class TableSpec(BaseModel):
    """A single SQL table with its columns."""
    table_name:  str
    columns:     list[ColumnSpec]
    description: str | None = None           # one-line semantic description for LLM prompt


class SQLSchemaSpec(BaseModel):
    """SchemaSpec for SQL persistence (Service 1 — PostgreSQL)."""
    persistence_type: Literal[PersistenceType.SQL] = PersistenceType.SQL
    tables:           list[TableSpec]

    @field_validator("tables")
    @classmethod
    def at_least_one_table(cls, v: list[TableSpec]) -> list[TableSpec]:
        if not v:
            raise ValueError("SQLSchemaSpec must contain at least one table")
        return v


class NodeTypeSpec(BaseModel):
    """A node label in the graph with its key properties."""
    label:       str
    properties:  list[str]                   # property names, not full column specs
    description: str | None = None


class EdgeTypeSpec(BaseModel):
    """A directed edge type in the graph."""
    edge_type:   str                          # e.g. "ASKED"
    from_label:  str                          # source node label
    to_label:    str                          # target node label
    properties:  list[str] = Field(default_factory=list)
    description: str | None = None


class GraphSchemaSpec(BaseModel):
    """SchemaSpec for Graph persistence (Service 2 — Neo4j)."""
    persistence_type: Literal[PersistenceType.GRAPH] = PersistenceType.GRAPH
    node_types:       list[NodeTypeSpec]
    edge_types:       list[EdgeTypeSpec]

    @field_validator("node_types")
    @classmethod
    def at_least_one_node(cls, v: list[NodeTypeSpec]) -> list[NodeTypeSpec]:
        if not v:
            raise ValueError("GraphSchemaSpec must define at least one node type")
        return v


class DocumentFieldSpec(BaseModel):
    """A searchable field in the document store.

    Accepts both field_name="x" and name="x" at construction time.
    Internal attribute is always .field_name so downstream code (prompts.py) works unchanged.
    """
    model_config = {"populate_by_name": True}

    field_name:   str  = Field(..., alias="name")
    field_type:   str                          # "text" | "keyword" | "integer" | "date"
    searchable:   bool = True                  # participates in full-text / BM25 retrieval
    description:  str | None = None


class DocumentTypeSpec(BaseModel):
    """A logical document type in the index (e.g. question body, comment)."""
    doc_type:     str                          # value of the doc_type field in ES
    fields:       list[DocumentFieldSpec]
    description:  str | None = None


class DocumentSchemaSpec(BaseModel):
    """SchemaSpec for Document persistence (Service 3 — Elasticsearch)."""
    persistence_type: Literal[PersistenceType.DOCUMENT] = PersistenceType.DOCUMENT
    index_name:       str
    doc_types:        list[DocumentTypeSpec]

    @field_validator("doc_types")
    @classmethod
    def at_least_one_doc_type(cls, v: list[DocumentTypeSpec]) -> list[DocumentTypeSpec]:
        if not v:
            raise ValueError("DocumentSchemaSpec must define at least one document type")
        return v


# Union type used in ServiceDescriptor — Pydantic resolves via the
# discriminator field `persistence_type`.
SchemaSpec = Annotated[
    Union[SQLSchemaSpec, GraphSchemaSpec, DocumentSchemaSpec],
    Field(discriminator="persistence_type"),
]


# ---------------------------------------------------------------------------
# ConnectionConfig — how Layer 2 connects to the underlying store
# ---------------------------------------------------------------------------

class ConnectionConfig(BaseModel):
    """
    Opaque connection parameters for a service's persistence layer.
    Layer 1 carries this through to Layer 2 without interpreting it.
    Credentials are passed via environment variables; this config holds only
    non-secret connection topology.
    """
    host:          str
    port:          int
    database:      str | None = None          # SQL database name / ES index name
    extra_params:  dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ServiceDescriptor — the atomic unit of I₁
# ---------------------------------------------------------------------------

_CAPABILITY_SUMMARY_TOKEN_LIMIT = 200        # hard cap from Foundation Doc §6 OQ2


class ServiceDescriptor(BaseModel):
    """
    The complete description of one microservice as registered with HeteroRAG.

    Producers (microservices) push this to the ServiceRegistry at startup.
    Consumers (Layer 2, RelevanceFilter) receive List[ServiceDescriptor] as I₁.

    I₁ producer guarantees:
      1. service_id is non-empty and unique in the registry.
      2. persistence_type matches the concrete SchemaSpec subtype.
      3. schema_spec is fully populated and valid for its type.
      4. capability_summary, if present, is ≤ 200 tokens.
      5. capability_embedding, if present, is unit-norm.
      6. schema_version is a non-empty string; incremented on schema change.
      7. registered_at is set by the registry on first registration.
    """

    # ---- Identity ----
    service_id:          str = Field(..., min_length=1,
                                     description="Stable unique identifier for this microservice")
    display_name:        str = Field(..., min_length=1)
    persistence_type:    PersistenceType

    # ---- Schema ----
    schema_spec:         SchemaSpec
    schema_version:      str = Field(..., min_length=1,
                                     description="Opaque version token; increment on schema change "
                                                 "to trigger partial cold-start on the affected service")

    # ---- Semantic capability description ----
    capability_summary:  str | None = Field(
        default=None,
        description="Free-text description of what this service owns and can answer. "
                    "Used in Stage 2 of the two-stage RelevanceFilter (LLM confirmation). "
                    f"Hard cap: {_CAPABILITY_SUMMARY_TOKEN_LIMIT} tokens.",
    )
    capability_embedding: list[float] | None = Field(
        default=None,
        description="Unit-norm embedding of capability_summary. "
                    "Pre-computed at registration time. "
                    "Used in Stage 1 of the two-stage RelevanceFilter (embedding pre-filter). "
                    "None if capability_summary is not provided.",
    )

    # ---- Connection topology (non-secret) ----
    connection:          ConnectionConfig

    # ---- Registry metadata (set by ServiceRegistry, not by the registrant) ----
    registered_at:       datetime | None = None
    last_heartbeat:      datetime | None = None

    # ---- Cold-start / planning state ----
    retrieval_weight:    float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current retrieval weight assigned by the planner. "
                    "Initialised to 0.0; set to 1/|M| on first query (cold start). "
                    "Reverts to 1/|M| on schema_version change (partial cold start).",
    )
    is_active:           bool = True

    # ---- Validators ----

    @field_validator("capability_summary")
    @classmethod
    def summary_within_token_limit(cls, v: str | None) -> str | None:
        """
        Approximate token count by whitespace splitting.
        A tighter count (e.g. tiktoken) can be plugged in at the registry layer.
        This guard prevents runaway context costs per-query.
        """
        if v is None:
            return v
        approx_tokens = len(v.split())
        if approx_tokens > _CAPABILITY_SUMMARY_TOKEN_LIMIT:
            raise ValueError(
                f"capability_summary exceeds {_CAPABILITY_SUMMARY_TOKEN_LIMIT}-token limit "
                f"(approx {approx_tokens} tokens). Shorten the description."
            )
        return v

    @field_validator("capability_embedding")
    @classmethod
    def embedding_is_unit_norm(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("capability_embedding must be non-empty if provided")
        norm_sq = sum(x * x for x in v)
        # Allow 1% tolerance for floating-point imprecision
        if abs(norm_sq - 1.0) > 0.01:
            raise ValueError(
                f"capability_embedding must be unit-norm (L2=1.0). "
                f"Got L2²={norm_sq:.6f}. Normalise the vector before registration."
            )
        return v

    @model_validator(mode="after")
    def schema_type_matches_persistence_type(self) -> "ServiceDescriptor":
        """Cross-field check: persistence_type must match the concrete SchemaSpec subtype."""
        expected = self.persistence_type.value
        actual   = self.schema_spec.persistence_type.value
        if expected != actual:
            raise ValueError(
                f"persistence_type='{expected}' does not match "
                f"schema_spec.persistence_type='{actual}'"
            )
        return self

    @model_validator(mode="after")
    def embedding_requires_summary(self) -> "ServiceDescriptor":
        if self.capability_embedding is not None and self.capability_summary is None:
            raise ValueError(
                "capability_embedding requires capability_summary to be set. "
                "The embedding is derived from the summary text."
            )
        return self

    def schema_version_changed(self, new_version: str) -> bool:
        """Returns True if new_version differs from the current schema_version."""
        return self.schema_version != new_version


# ---------------------------------------------------------------------------
# I₁ — the formal interface produced by Layer 1 and consumed by Layer 2
# ---------------------------------------------------------------------------

class I1_ServiceDiscoveryOutput(BaseModel):
    """
    Formal interface I₁: the output of Layer 1 (Service Discovery).

    Consumed by:
      - Layer 2 (Query Translation) — to build per-service query translation prompts
      - Two-stage RelevanceFilter — to shortlist services relevant to a given query

    Producer guarantees (all items in descriptors satisfy ServiceDescriptor invariants):
      1. All descriptors have is_active=True.
      2. All descriptors have a populated schema_spec.
      3. No duplicate service_ids.
      4. query_id links this output to a specific user query for traceability.
    """
    query_id:    str = Field(..., description="Opaque identifier linking I₁ to a specific user query")
    descriptors: list[ServiceDescriptor]
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def no_duplicate_service_ids(self) -> "I1_ServiceDiscoveryOutput":
        ids = [d.service_id for d in self.descriptors]
        if len(ids) != len(set(ids)):
            dupes = [sid for sid in ids if ids.count(sid) > 1]
            raise ValueError(f"I₁ contains duplicate service_ids: {set(dupes)}")
        return self

    @model_validator(mode="after")
    def all_descriptors_active(self) -> "I1_ServiceDiscoveryOutput":
        inactive = [d.service_id for d in self.descriptors if not d.is_active]
        if inactive:
            raise ValueError(
                f"I₁ must not contain inactive services: {inactive}. "
                "Filter inactive services before constructing I₁."
            )
        return self

    def __len__(self) -> int:
        return len(self.descriptors)

    def service_ids(self) -> list[str]:
        return [d.service_id for d in self.descriptors]
