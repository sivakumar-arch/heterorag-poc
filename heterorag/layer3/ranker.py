"""
heterorag/layer3/ranker.py
============================
Step 9 — Ranker.

Scores and ranks DeduplicatedItems, formats context_text for Layer 4,
and constructs the formal I₃ interface (I3_IntegratedContext).

Ranking formula (POC):
    final_score = item.score × boost

    boost factors:
      - cross_source_boost (default 1.2): items merged from ≥2 services
        receive a modest boost because cross-source corroboration is evidence
        of genuine relevance. This is architecturally motivated and stated
        in the paper.
      - single_source (1.0): no adjustment.

    Items with score = 0 after all boosts are dropped from the context.

Context formatting:
    Each ranked item is formatted as a labelled block:
      [SOURCE: SQL | RANK: 1]
      <content>

    Blocks are concatenated in rank order (highest score first).
    A total character cap (default 8000) prevents context overflow.
    The cap is applied by dropping lowest-ranked items — never by truncating
    a single item mid-content.

I₃ guarantees:
    1. items are ordered by final_score descending (rank 1 = best).
    2. context_text is ready for direct injection into the Layer 4 prompt.
    3. total_wall_ms is propagated from the RetrievalBatch (RL metric).
    4. queried_service_ids = all service_ids that returned successful results.
    5. conflict_count = ICR numerator for this query.
"""

from __future__ import annotations

import logging

from .models import (
    DeduplicatedBatch,
    I3_IntegratedContext,
    RankedItem,
    SourceType,
)

log = logging.getLogger(__name__)

_DEFAULT_CROSS_SOURCE_BOOST = 1.2
_DEFAULT_CONTEXT_CHAR_CAP   = 8_000
_DEFAULT_MAX_ITEMS          = 20


class Ranker:
    """
    Step 9: ranks DeduplicatedItems and produces I₃.

    Usage:
        ranker = Ranker()
        i3 = ranker.rank(dedup_batch, queried_service_ids, natural_query, total_wall_ms)
    """

    def __init__(
        self,
        cross_source_boost: float = _DEFAULT_CROSS_SOURCE_BOOST,
        context_char_cap:   int   = _DEFAULT_CONTEXT_CHAR_CAP,
        max_items:          int   = _DEFAULT_MAX_ITEMS,
    ):
        self._boost    = cross_source_boost
        self._char_cap = context_char_cap
        self._max_items = max_items

    def rank(
        self,
        dedup_batch:          DeduplicatedBatch,
        queried_service_ids:  list[str],
        natural_query:        str,
        total_wall_ms:        float = 0.0,
    ) -> I3_IntegratedContext:
        items = dedup_batch.items

        if not items:
            log.warning("Ranker: no items to rank — returning empty I₃")
            return I3_IntegratedContext(
                query_id            = dedup_batch.query_id,
                natural_query       = natural_query,
                items               = [],
                context_text        = "",
                queried_service_ids = queried_service_ids,
                total_wall_ms       = total_wall_ms,
                conflict_count      = dedup_batch.conflict_count,
            )

        # --- Score each item ---
        scored: list[tuple[float, object]] = []
        for item in items:
            boost      = self._boost if item.is_cross_source else 1.0
            final_score = min(item.score * boost, 1.0)
            scored.append((final_score, item))

        # --- Sort descending, assign ranks ---
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[: self._max_items]

        ranked_items: list[RankedItem] = []
        for rank, (final_score, item) in enumerate(scored, start=1):
            ranked_items.append(RankedItem(
                rank               = rank,
                content            = item.content,
                source_service_ids = item.source_service_ids,
                source_types       = item.source_types,
                final_score        = round(final_score, 4),
                is_cross_source    = item.is_cross_source,
            ))

        # --- Format context_text with character cap ---
        context_text = self._format_context(ranked_items)

        log.info(
            "Ranker: %d deduplicated → %d ranked items  chars=%d  wall=%.1fms",
            len(items), len(ranked_items), len(context_text), total_wall_ms,
        )

        return I3_IntegratedContext(
            query_id            = dedup_batch.query_id,
            natural_query       = natural_query,
            items               = ranked_items,
            context_text        = context_text,
            queried_service_ids = queried_service_ids,
            total_wall_ms       = total_wall_ms,
            conflict_count      = dedup_batch.conflict_count,
        )

    # ------------------------------------------------------------------

    def _format_context(self, ranked_items: list[RankedItem]) -> str:
        """
        Format ranked items as labelled blocks for the Layer 4 prompt.
        Applies the character cap by dropping trailing items, never truncating.
        """
        blocks: list[str] = []
        total_chars = 0

        for item in ranked_items:
            src_label = " + ".join(
                st.value.upper() for st in (item.source_types or [SourceType.SQL])
            )
            block = (
                f"[SOURCE: {src_label} | RANK: {item.rank} | SCORE: {item.final_score:.3f}]\n"
                f"{item.content}\n"
            )
            if total_chars + len(block) > self._char_cap:
                break
            blocks.append(block)
            total_chars += len(block)

        return "\n".join(blocks)
