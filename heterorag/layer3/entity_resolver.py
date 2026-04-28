"""
heterorag/layer3/entity_resolver.py
======================================
Step 8 — EntityResolver + Deduplicator.

Detects and collapses duplicate entities across structurally different result sets,
producing a DeduplicatedBatch where each real-world entity appears at most once.

Design decisions from Foundation Doc v1.6 §6:
  - Deterministic sources (SQL, Graph) have extraction_confidence = 1.0.
  - Probabilistic sources (Document) have extraction_confidence < 1.0.
  - Cross-source entity matches are weighted by the minimum confidence of the
    two contributing entities. A SQL user_id = 12345 and a Document mention of
    the same user_id is a high-confidence match (1.0 × 1.0). A Document NER
    mention matched to a SQL display_name is lower confidence (1.0 × 0.5).
  - Conflicts: when two services agree on an entity_id but disagree on an
    attribute value, a ConflictRecord is emitted. This feeds the ICR metric.

Entity matching strategy:
  Primary key match (exact):
    user_id, post_id matched across sources by integer value.
    Confidence = min(confidence_a, confidence_b).
  Tag match (exact string):
    Tag names matched across SQL tag strings and Graph Tag nodes.
  Freetext mention (heuristic):
    Only merged when confidence_a × confidence_b > 0.5 (product threshold).
    Otherwise kept as separate items to avoid false deduplication.

Conflict detection:
  Triggered when the same entity_id appears in two sources with a numeric
  attribute that differs by more than CONFLICT_TOLERANCE. In the POC this
  is primarily reputation (SQL vs Graph may differ if the data load was
  not perfectly synchronised).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .models import (
    ConflictRecord,
    DeduplicatedBatch,
    DeduplicatedItem,
    ExtractedEntity,
    NormalisedBatch,
    NormalisedItem,
    SourceType,
)

log = logging.getLogger(__name__)

# When two numeric attributes differ by less than this fraction, no conflict
CONFLICT_TOLERANCE = 0.01   # 1% relative difference
# Freetext mention pairs below this product-confidence threshold are NOT merged
FREETEXT_MERGE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Entity key — (entity_type, canonical_value) pairs used as dedup keys
# ---------------------------------------------------------------------------

def _entity_key(entity: ExtractedEntity) -> tuple[str, Any]:
    """
    Canonical key for entity deduplication.
    Integer IDs are cast to int for cross-source matching (SQL int vs ES string).
    """
    val = entity.value
    if entity.entity_type in ("user_id", "post_id"):
        try:
            val = int(val)
        except (ValueError, TypeError):
            pass
    elif entity.entity_type == "tag":
        val = str(val).lower().strip()
    return (entity.entity_type, val)


# ---------------------------------------------------------------------------
# EntityResolver + Deduplicator
# ---------------------------------------------------------------------------

class EntityResolver:
    """
    Step 8: collapses duplicate items across services.

    Items from different services that share a primary-key entity (user_id,
    post_id, or tag) are merged into a single DeduplicatedItem whose content
    is the concatenation of both sources' content strings. The merged item
    carries both service_ids in source_service_ids.

    Items with no cross-source match are passed through unchanged.
    """

    def deduplicate(self, batch: NormalisedBatch) -> DeduplicatedBatch:
        all_items = batch.all_items()

        if not all_items:
            return DeduplicatedBatch(
                query_id       = batch.query_id,
                items          = [],
                source_wall_ms = batch.source_wall_ms,
            )

        # Build an index: entity_key → list of items that carry that entity
        entity_index: dict[tuple, list[NormalisedItem]] = defaultdict(list)
        for item in all_items:
            for entity in item.entities:
                if entity.entity_type in ("user_id", "post_id", "tag"):
                    entity_index[_entity_key(entity)].append(item)
                elif entity.entity_type == "freetext_mention":
                    # Only index high-confidence freetext
                    if entity.extraction_confidence >= FREETEXT_MERGE_THRESHOLD:
                        entity_index[_entity_key(entity)].append(item)

        # Build merge groups: sets of item_ids that should be merged
        # Union-Find-style: assign each item to a canonical group
        item_to_group: dict[str, str] = {}   # item_id → canonical_id
        group_to_items: dict[str, set[str]] = {}  # canonical_id → set of item_ids

        for key, items_for_key in entity_index.items():
            if len(items_for_key) < 2:
                continue

            # Check confidence threshold for freetext merges
            entity_type = key[0]
            if entity_type == "freetext_mention":
                confidences = [
                    min(e.extraction_confidence for e in item.entities
                        if _entity_key(e) == key)
                    for item in items_for_key
                    if any(_entity_key(e) == key for e in item.entities)
                ]
                if any(c < FREETEXT_MERGE_THRESHOLD for c in confidences):
                    continue

            # Merge all items sharing this entity key into one group
            group_ids = [
                item_to_group.get(item.item_id, item.item_id)
                for item in items_for_key
            ]
            canonical = min(group_ids)  # lowest item_id as canonical
            for item in items_for_key:
                item_to_group[item.item_id] = canonical
                if canonical not in group_to_items:
                    group_to_items[canonical] = set()
                group_to_items[canonical].add(item.item_id)

        # Build lookup by item_id
        item_by_id = {item.item_id: item for item in all_items}

        # Produce deduplicated items
        dedup_items: list[DeduplicatedItem] = []
        conflicts:   list[ConflictRecord]   = []
        processed_ids: set[str] = set()

        for item in all_items:
            if item.item_id in processed_ids:
                continue

            canonical = item_to_group.get(item.item_id, item.item_id)
            group_item_ids = group_to_items.get(canonical, {item.item_id})
            group_members  = [item_by_id[iid] for iid in group_item_ids if iid in item_by_id]

            if len(group_members) == 1:
                # No merge needed
                di = self._single_item(item)
            else:
                # Merge group — detect conflicts first
                new_conflicts = self._detect_conflicts(group_members)
                conflicts.extend(new_conflicts)
                di = self._merge_items(group_members, canonical)

            dedup_items.append(di)
            processed_ids.update(group_item_ids)

        log.info(
            "EntityResolver: %d normalised items → %d deduplicated  conflicts=%d",
            len(all_items), len(dedup_items), len(conflicts),
        )

        return DeduplicatedBatch(
            query_id         = batch.query_id,
            items            = dedup_items,
            conflict_records = conflicts,
            source_wall_ms   = batch.source_wall_ms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _single_item(item: NormalisedItem) -> DeduplicatedItem:
        return DeduplicatedItem(
            canonical_id       = item.item_id,
            content            = item.content,
            entities           = item.entities,
            score              = item.score * item.retrieval_weight,
            source_service_ids = [item.source_service_id],
            source_types       = [item.source_type],
            is_cross_source    = False,
        )

    @staticmethod
    def _merge_items(items: list[NormalisedItem], canonical_id: str) -> DeduplicatedItem:
        """
        Merge multiple items about the same entity into one.

        Content: concatenate with source labels.
        Score: weighted average, weights = retrieval_weight of each source.
        Entities: union of all entity lists.
        """
        parts: list[str] = []
        all_entities: list[ExtractedEntity] = []
        svc_ids:    list[str] = []
        src_types:  list[SourceType] = []
        total_weight = sum(i.retrieval_weight for i in items)

        for item in sorted(items, key=lambda x: -x.retrieval_weight):
            parts.append(f"[{item.source_type.value.upper()}] {item.content}")
            all_entities.extend(item.entities)
            if item.source_service_id not in svc_ids:
                svc_ids.append(item.source_service_id)
            if item.source_type not in src_types:
                src_types.append(item.source_type)

        # Weighted average score
        merged_score = (
            sum(i.score * i.retrieval_weight for i in items) / total_weight
            if total_weight > 0
            else 0.0
        )
        merged_score = min(merged_score, 1.0)

        return DeduplicatedItem(
            canonical_id       = canonical_id,
            content            = "\n".join(parts),
            entities           = all_entities,
            score              = merged_score,
            source_service_ids = svc_ids,
            source_types       = src_types,
            is_cross_source    = len(svc_ids) > 1,
        )

    @staticmethod
    def _detect_conflicts(items: list[NormalisedItem]) -> list[ConflictRecord]:
        """
        Detect attribute conflicts: same entity, different numeric attribute values
        across services. Emits one ConflictRecord per detected disagreement.
        """
        conflicts: list[ConflictRecord] = []

        # Only bother for cross-source pairs
        for i, item_a in enumerate(items):
            for item_b in items[i + 1:]:
                if item_a.source_service_id == item_b.source_service_id:
                    continue

                # Find shared numeric attributes in raw_records
                raw_a = item_a.raw_record
                raw_b = item_b.raw_record
                shared_keys = set(raw_a) & set(raw_b)

                for key in shared_keys:
                    val_a, val_b = raw_a[key], raw_b[key]
                    if not (isinstance(val_a, (int, float)) and
                            isinstance(val_b, (int, float))):
                        continue
                    if val_a == 0 and val_b == 0:
                        continue
                    denom = max(abs(val_a), abs(val_b))
                    if abs(val_a - val_b) / denom > CONFLICT_TOLERANCE:
                        conflicts.append(ConflictRecord(
                            entity_type  = "numeric_attribute",
                            entity_value = key,
                            service_a    = item_a.source_service_id,
                            service_b    = item_b.source_service_id,
                            attribute    = key,
                            value_a      = val_a,
                            value_b      = val_b,
                        ))

        return conflicts
