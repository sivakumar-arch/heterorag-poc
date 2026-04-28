"""
heterorag/layer1/registry.py
============================
ServiceRegistry — the authoritative store of active ServiceDescriptors.

Responsibilities:
  1. Accept self-registrations from microservices (push model, OQ1 resolved).
  2. Detect schema_version changes and apply partial cold-start policy (OQ14 mitigation).
  3. Maintain heartbeat-based liveness; mark services inactive on missed heartbeats.
  4. Produce I₁ (List[ServiceDescriptor]) for a given query via the two-stage
     RelevanceFilter (embedding pre-filter → LLM confirmation).
  5. Expose cold-start initialisation: set weight = 1/|M| when the first query arrives
     and the planner has no learned weights yet.

Design decisions locked in Foundation Doc v1.6 §6:
  - Self-registration: services push ServiceDescriptor at startup.
  - Two-stage RelevanceFilter: Stage 1 = embedding cosine similarity (O(n), no LLM);
    Stage 2 = LLM confirmation on top-k descriptors only.
  - Partial cold start: on schema_version change, ONLY the affected service reverts
    to uniform weight; other services retain learned weights.
  - capability_summary embeddings pre-computed at registration time, not per-query.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

from .models import (
    I1_ServiceDiscoveryOutput,
    PersistenceType,
    ServiceDescriptor,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry configuration
# ---------------------------------------------------------------------------

_DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 60
_DEFAULT_STAGE1_TOP_K              = 10     # max candidates passed to Stage 2 LLM filter
_DEFAULT_STAGE1_MIN_SIMILARITY     = 0.0    # accept all in Stage 1 if no embeddings present


# ---------------------------------------------------------------------------
# Embedding utilities (Stage 1)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two pre-normalised (unit-norm) vectors.
    For unit vectors: cos(θ) = dot product.
    """
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def _unit_norm(v: list[float]) -> list[float]:
    """Returns L2-normalised copy of v."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        raise ValueError("Cannot normalise a zero vector")
    return [x / norm for x in v]


# ---------------------------------------------------------------------------
# EmbeddingFunction protocol
# ---------------------------------------------------------------------------

EmbeddingFn = Callable[[str], list[float]]
"""
Callable[str] → list[float] (unit-norm).
Injected at construction time so the registry has no hard dependency on a
specific embedding provider. In the POC this is backed by sentence-transformers
or the Anthropic embedding API.
"""


# ---------------------------------------------------------------------------
# RelevanceFilter — two-stage shortlisting
# ---------------------------------------------------------------------------

class RelevanceFilter:
    """
    Two-stage shortlisting of services relevant to a given query.

    Stage 1 — Embedding pre-filter (O(n), no LLM):
        Score each service's capability_embedding against the query embedding.
        Retain top-k by cosine similarity.
        Services without an embedding pass through to Stage 2 unconditionally
        (fallback: LLM sees their SchemaSpec-based description instead).

    Stage 2 — LLM confirmation (bounded to top-k from Stage 1):
        Send top-k ServiceDescriptor summaries to the LLM with the query.
        LLM returns a subset of service_ids it deems relevant.
        This is the only per-query LLM call in Layer 1 — and it is bounded
        regardless of mesh size.

    If no embed_fn is provided (e.g. during testing), Stage 1 is skipped
    and all active services are forwarded to Stage 2.
    """

    def __init__(
        self,
        embed_fn:        EmbeddingFn | None = None,
        llm_confirm_fn:  Callable[[str, list[ServiceDescriptor]], list[str]] | None = None,
        stage1_top_k:    int   = _DEFAULT_STAGE1_TOP_K,
        min_similarity:  float = _DEFAULT_STAGE1_MIN_SIMILARITY,
    ):
        self._embed_fn       = embed_fn
        self._llm_confirm_fn = llm_confirm_fn
        self._stage1_top_k   = stage1_top_k
        self._min_similarity = min_similarity

    def filter(
        self,
        query:       str,
        descriptors: list[ServiceDescriptor],
    ) -> list[ServiceDescriptor]:
        """
        Returns the subset of descriptors that are relevant to query.
        Preserves the order returned by Stage 2 (or Stage 1 ranking if Stage 2 absent).
        """
        if not descriptors:
            return []

        # --- Stage 1: embedding pre-filter ---
        after_stage1 = self._stage1(query, descriptors)
        log.debug("RelevanceFilter Stage 1: %d → %d candidates",
                  len(descriptors), len(after_stage1))

        # --- Stage 2: LLM confirmation ---
        if self._llm_confirm_fn is None:
            log.debug("RelevanceFilter Stage 2: no LLM confirm_fn — returning Stage 1 result")
            return after_stage1

        confirmed_ids = set(self._llm_confirm_fn(query, after_stage1))
        result = [d for d in after_stage1 if d.service_id in confirmed_ids]
        log.debug("RelevanceFilter Stage 2: %d → %d confirmed",
                  len(after_stage1), len(result))
        return result

    def _stage1(
        self,
        query:       str,
        descriptors: list[ServiceDescriptor],
    ) -> list[ServiceDescriptor]:
        """
        Embedding pre-filter.
        Services without embeddings are treated as having similarity = 1.0
        (conservative: pass them through so Stage 2 can decide).
        """
        if self._embed_fn is None:
            return descriptors  # no embed_fn — pass all through

        # Compute query embedding
        try:
            query_emb = _unit_norm(self._embed_fn(query))
        except Exception as exc:
            log.warning("Stage 1 embed_fn failed (%s) — skipping Stage 1", exc)
            return descriptors

        scored: list[tuple[float, ServiceDescriptor]] = []
        for d in descriptors:
            if d.capability_embedding is not None:
                try:
                    score = _cosine_similarity(query_emb, d.capability_embedding)
                except ValueError:
                    score = 1.0   # dimension mismatch — pass through
            else:
                score = 1.0       # no embedding — pass through conservatively
            scored.append((score, d))

        # Sort descending by similarity, keep top-k above threshold
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            d for score, d in scored[: self._stage1_top_k]
            if score >= self._min_similarity
        ]


# ---------------------------------------------------------------------------
# RegistryEntry — internal wrapper around ServiceDescriptor
# ---------------------------------------------------------------------------

class _RegistryEntry:
    __slots__ = ("descriptor", "last_heartbeat", "previous_schema_version")

    def __init__(self, descriptor: ServiceDescriptor):
        self.descriptor              = descriptor
        self.last_heartbeat: datetime = datetime.now(timezone.utc)
        self.previous_schema_version: str | None = None


# ---------------------------------------------------------------------------
# ServiceRegistry
# ---------------------------------------------------------------------------

class ServiceRegistry:
    """
    Thread-safe registry of active ServiceDescriptors.

    Lifecycle:
        registry = ServiceRegistry(embed_fn=..., llm_confirm_fn=...)
        registry.register(descriptor)       # called by each microservice at startup
        registry.heartbeat("service-id")    # called periodically by each service
        i1 = registry.discover(query_id, query_text)  # called per user query

    Cold-start policy (Foundation Doc §6):
        On first query arrival, any service with retrieval_weight == 0.0 receives
        weight = 1/|M| (uniform prior). This is set in-place on the stored descriptor.

    Schema drift (Foundation Doc §6, OQ14):
        If a re-registration arrives with a changed schema_version, only that service
        reverts to uniform weight. All other services retain their current weights.
    """

    def __init__(
        self,
        embed_fn:                  EmbeddingFn | None = None,
        llm_confirm_fn:            Callable[[str, list[ServiceDescriptor]], list[str]] | None = None,
        heartbeat_timeout_seconds: int   = _DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
        stage1_top_k:              int   = _DEFAULT_STAGE1_TOP_K,
        min_stage1_similarity:     float = _DEFAULT_STAGE1_MIN_SIMILARITY,
    ):
        self._entries:   dict[str, _RegistryEntry] = {}
        self._lock:      threading.RLock           = threading.RLock()
        self._hb_timeout = timedelta(seconds=heartbeat_timeout_seconds)
        self._filter     = RelevanceFilter(
            embed_fn        = embed_fn,
            llm_confirm_fn  = llm_confirm_fn,
            stage1_top_k    = stage1_top_k,
            min_similarity  = min_stage1_similarity,
        )
        self._first_query_seen = False
        log.info("ServiceRegistry initialised (heartbeat_timeout=%ds)", heartbeat_timeout_seconds)

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    def register(self, descriptor: ServiceDescriptor) -> None:
        """
        Register or re-register a service.

        - First registration: stores the descriptor and stamps registered_at.
        - Re-registration with same schema_version: updates metadata, keeps weight.
        - Re-registration with changed schema_version: applies partial cold start —
          reverts THIS service's weight to 1/|M|; all others are unchanged.
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            if descriptor.service_id in self._entries:
                existing_entry = self._entries[descriptor.service_id]
                existing       = existing_entry.descriptor

                if existing.schema_version != descriptor.schema_version:
                    # --- Partial cold start ---
                    log.warning(
                        "Schema version change detected for '%s': '%s' → '%s'. "
                        "Applying partial cold start — weight reset to uniform.",
                        descriptor.service_id,
                        existing.schema_version,
                        descriptor.schema_version,
                    )
                    existing_entry.previous_schema_version = existing.schema_version
                    uniform_weight = self._uniform_weight()
                    updated = descriptor.model_copy(update={
                        "registered_at":   existing.registered_at,  # preserve original registration time
                        "retrieval_weight": uniform_weight,
                        "is_active":       True,
                    })
                    self._entries[descriptor.service_id] = _RegistryEntry(updated)
                else:
                    # Re-registration, same schema — update metadata only
                    updated = descriptor.model_copy(update={
                        "registered_at":    existing.registered_at,
                        "retrieval_weight": existing.retrieval_weight,  # preserve learned weight
                        "is_active":        True,
                    })
                    self._entries[descriptor.service_id].descriptor = updated

                log.info("Re-registered service '%s' (schema_version=%s)",
                         descriptor.service_id, descriptor.schema_version)
            else:
                # First registration
                stamped = descriptor.model_copy(update={
                    "registered_at":    now,
                    "last_heartbeat":   now,
                    "retrieval_weight": 0.0,   # cold start — weight set on first query
                    "is_active":        True,
                })
                self._entries[descriptor.service_id] = _RegistryEntry(stamped)
                log.info("Registered new service '%s' (persistence=%s, schema_version=%s)",
                         descriptor.service_id,
                         descriptor.persistence_type.value,
                         descriptor.schema_version)

    def deregister(self, service_id: str) -> None:
        """Mark a service inactive (does not delete — preserves history)."""
        with self._lock:
            if service_id not in self._entries:
                log.warning("deregister: unknown service_id '%s'", service_id)
                return
            entry = self._entries[service_id]
            entry.descriptor = entry.descriptor.model_copy(update={"is_active": False})
            log.info("Deregistered service '%s'", service_id)

    # -----------------------------------------------------------------------
    # Heartbeat
    # -----------------------------------------------------------------------

    def heartbeat(self, service_id: str) -> None:
        """Record a liveness heartbeat for the named service."""
        with self._lock:
            if service_id not in self._entries:
                log.warning("heartbeat: unknown service_id '%s'", service_id)
                return
            self._entries[service_id].last_heartbeat = datetime.now(timezone.utc)

    def expire_stale_services(self) -> list[str]:
        """
        Mark services inactive if they have not sent a heartbeat within the timeout.
        Returns list of service_ids that were marked inactive.
        Called by a background thread or test harness.
        """
        now     = datetime.now(timezone.utc)
        expired = []
        with self._lock:
            for service_id, entry in self._entries.items():
                if not entry.descriptor.is_active:
                    continue
                age = now - entry.last_heartbeat
                if age > self._hb_timeout:
                    entry.descriptor = entry.descriptor.model_copy(update={"is_active": False})
                    expired.append(service_id)
                    log.warning(
                        "Service '%s' marked inactive — no heartbeat for %.1fs",
                        service_id, age.total_seconds(),
                    )
        return expired

    # -----------------------------------------------------------------------
    # Weight management
    # -----------------------------------------------------------------------

    def update_weight(self, service_id: str, weight: float) -> None:
        """
        Update the retrieval weight for a service.
        Called by the genetic planner (Paper 2) after each optimisation cycle.
        """
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Weight must be in [0, 1], got {weight}")
        with self._lock:
            if service_id not in self._entries:
                raise KeyError(f"Unknown service_id: '{service_id}'")
            entry = self._entries[service_id]
            entry.descriptor = entry.descriptor.model_copy(
                update={"retrieval_weight": weight}
            )

    def _uniform_weight(self) -> float:
        """
        Compute 1/|M| where |M| is the number of currently active services.
        Used for cold-start initialisation and partial cold-start resets.
        """
        active_count = sum(
            1 for e in self._entries.values() if e.descriptor.is_active
        )
        return 1.0 / max(active_count, 1)

    def _apply_cold_start_if_needed(self) -> None:
        """
        On first query, initialise retrieval_weight = 1/|M| for every service
        that still has weight == 0.0 (i.e. has never been touched by the planner).
        This is the maximum-entropy prior before any optimisation has occurred.
        """
        if self._first_query_seen:
            return
        self._first_query_seen = True
        uniform = self._uniform_weight()
        for entry in self._entries.values():
            if entry.descriptor.retrieval_weight == 0.0:
                entry.descriptor = entry.descriptor.model_copy(
                    update={"retrieval_weight": uniform}
                )
        log.info("Cold start: uniform retrieval weight %.4f applied to all services", uniform)

    # -----------------------------------------------------------------------
    # Discovery — produces I₁
    # -----------------------------------------------------------------------

    def discover(self, query_id: str, query_text: str) -> I1_ServiceDiscoveryOutput:
        """
        Main entry point for Layer 1.

        1. Expire stale services (heartbeat check).
        2. Apply cold-start weights if this is the first query.
        3. Run the two-stage RelevanceFilter on active services.
        4. Return I₁ (List[ServiceDescriptor]) with all I₁ producer guarantees.

        Args:
            query_id:   Opaque identifier linking this I₁ to the user query.
            query_text: The raw natural language query (used by Stage 1 embed + Stage 2 LLM).

        Returns:
            I1_ServiceDiscoveryOutput — guaranteed to contain only active, valid descriptors.
        """
        with self._lock:
            self.expire_stale_services()
            self._apply_cold_start_if_needed()

            active = [
                entry.descriptor
                for entry in self._entries.values()
                if entry.descriptor.is_active
            ]

        if not active:
            log.warning("discover(%s): no active services in registry", query_id)
            return I1_ServiceDiscoveryOutput(query_id=query_id, descriptors=[])

        # Two-stage relevance filter
        relevant = self._filter.filter(query_text, active)

        log.info(
            "discover(query_id=%s): %d active services → %d relevant after filter",
            query_id, len(active), len(relevant),
        )

        return I1_ServiceDiscoveryOutput(query_id=query_id, descriptors=relevant)

    # -----------------------------------------------------------------------
    # Inspection / admin
    # -----------------------------------------------------------------------

    def all_descriptors(self, active_only: bool = True) -> list[ServiceDescriptor]:
        """Return all (or only active) registered descriptors. Useful for admin/debug."""
        with self._lock:
            return [
                e.descriptor for e in self._entries.values()
                if (not active_only) or e.descriptor.is_active
            ]

    def get(self, service_id: str) -> ServiceDescriptor | None:
        """Retrieve a single descriptor by service_id."""
        with self._lock:
            entry = self._entries.get(service_id)
            return entry.descriptor if entry else None

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if e.descriptor.is_active)

    def __repr__(self) -> str:
        with self._lock:
            ids = [e.descriptor.service_id for e in self._entries.values()
                   if e.descriptor.is_active]
        return f"ServiceRegistry(active={ids})"
