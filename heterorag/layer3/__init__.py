"""
heterorag/layer3/__init__.py
Layer 3 — Semantic Integration

Public API:
    ParallelRetrievalExecutor   — Step 6: parallel query execution
    ConnectionRegistry           — service connection config store
    ResultNormaliser             — Step 7: heterogeneous → uniform NormalisedItems
    EntityResolver               — Step 8: cross-source deduplication + conflict detection
    Ranker                       — Step 9: scoring, ranking, I₃ construction
    SemanticIntegrator           — facade: chains Steps 7-9 from a RetrievalBatch → I₃
    I3_IntegratedContext         — formal I₃ interface (consumed by Layer 4)
"""

from .models import (
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
from .retrieval_executor import (
    ConnectionRegistry,
    ParallelRetrievalExecutor,
)
from .result_normaliser import ResultNormaliser
from .entity_resolver import EntityResolver
from .ranker import Ranker


class SemanticIntegrator:
    """
    Facade that chains Steps 7 → 8 → 9 from a RetrievalBatch into I₃.

    Usage:
        integrator = SemanticIntegrator()
        i3 = integrator.integrate(retrieval_batch, natural_query)
    """

    def __init__(
        self,
        cross_source_boost: float = 1.2,
        context_char_cap:   int   = 8_000,
        max_items:          int   = 20,
    ):
        self._normaliser = ResultNormaliser()
        self._resolver   = EntityResolver()
        self._ranker     = Ranker(
            cross_source_boost = cross_source_boost,
            context_char_cap   = context_char_cap,
            max_items          = max_items,
        )

    def integrate(
        self,
        batch:         RetrievalBatch,
        natural_query: str,
    ) -> I3_IntegratedContext:
        # Step 7
        normalised = self._normaliser.normalise(batch)
        # Step 8
        deduped    = self._resolver.deduplicate(normalised)
        # Step 9
        queried_ids = [r.service_id for r in batch.successful()]
        return self._ranker.rank(
            dedup_batch         = deduped,
            queried_service_ids = queried_ids,
            natural_query       = natural_query,
            total_wall_ms       = batch.total_wall_ms,
        )


__all__ = [
    "ConflictRecord",
    "ConnectionRegistry",
    "DeduplicatedBatch",
    "DeduplicatedItem",
    "EntityResolver",
    "ExtractedEntity",
    "I3_IntegratedContext",
    "NormalisedBatch",
    "NormalisedItem",
    "NormalisedServiceResult",
    "ParallelRetrievalExecutor",
    "RankedItem",
    "RawServiceResult",
    "Ranker",
    "RetrievalBatch",
    "RetrievalError",
    "ResultNormaliser",
    "SemanticIntegrator",
    "SourceType",
]
