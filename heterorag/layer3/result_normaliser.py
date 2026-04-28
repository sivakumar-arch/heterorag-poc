"""
heterorag/layer3/result_normaliser.py
========================================
Step 7 — ResultNormaliser.

Converts heterogeneous RawServiceResults into a uniform NormalisedBatch.
Each source type has its own normalisation path because the raw formats
are structurally incompatible:

  SQL rows    → structured dicts with typed columns; entities extracted
                deterministically from well-known ID columns (user_id, post_id).
                content rendered as "KEY: value | KEY: value" strings.

  Graph records → Neo4j driver dicts; node objects extracted; entity IDs
                  read from node properties deterministically.
                  content rendered with node labels and relationship context.

  BM25 hits   → Elasticsearch _source dicts; entities extracted via
                lightweight spaCy NER (probabilistic) or regex fallback.
                score normalised within the result set to [0,1].
                content = title (if present) + body excerpt (first 500 chars).

NormalisedContent asymmetry (Foundation Doc v1.6 §6):
  SQL and Graph entity extraction is DETERMINISTIC (confidence=1.0).
  Document entity extraction is PROBABILISTIC (NER confidence passed through).
  This asymmetry is propagated to the EntityResolver in Step 8 and must be
  stated explicitly in the paper.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from .models import (
    ExtractedEntity,
    NormalisedBatch,
    NormalisedItem,
    NormalisedServiceResult,
    RawServiceResult,
    RetrievalBatch,
    SourceType,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic column sets for deterministic entity extraction (SQL + Graph)
# ---------------------------------------------------------------------------

_ID_COLUMNS = frozenset({
    "id", "user_id", "post_id", "owner_user_id", "parent_id",
    "accepted_answer_id", "last_editor_user_id", "related_post_id",
})
_TAG_COLUMNS  = frozenset({"tag_name", "name", "tags"})
_TEXT_COLUMNS = frozenset({"title", "body", "display_name", "location", "about_me"})

# BM25 body excerpt maximum characters for content field
_BODY_EXCERPT_CHARS = 500

# Regex for lightweight entity extraction when spaCy is absent
_UPPER_TOKEN = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_item_id(service_id: str, index: int, row: dict) -> str:
    """
    Generate a stable, unique item_id for a normalised result.
    Prefers a database primary key when present; otherwise hashes the row.
    """
    # Prefer explicit primary key fields
    for pk_col in ("id", "_id", "user_id", "post_id"):
        if pk_col in row and row[pk_col] is not None:
            return f"{service_id}:{pk_col}:{row[pk_col]}"
    # Fallback: hash the row content
    digest = hashlib.md5(str(sorted(row.items())).encode()).hexdigest()[:8]
    return f"{service_id}:row{index}:{digest}"


def _normalise_score(score: float, max_score: float) -> float:
    """Normalise a raw score to [0, 1] given the maximum in the result set."""
    if max_score <= 0:
        return 1.0
    return min(score / max_score, 1.0)


# ---------------------------------------------------------------------------
# SQL normalisation path (deterministic entity extraction)
# ---------------------------------------------------------------------------

def _normalise_sql_row(
    row:        dict[str, Any],
    service_id: str,
    index:      int,
    retrieval_weight: float,
) -> NormalisedItem:
    entities: list[ExtractedEntity] = []

    # Extract ID-type entities deterministically
    for col in _ID_COLUMNS:
        if col in row and row[col] is not None:
            entities.append(ExtractedEntity(
                name                  = col,
                entity_type           = col,
                value                 = row[col],
                extraction_confidence = 1.0,   # deterministic
                source_type           = SourceType.SQL,
            ))

    # Extract tag entities
    for col in _TAG_COLUMNS:
        if col in row and row[col]:
            raw_tags = str(row[col])
            # Parse <python><pandas> format
            tag_names = re.findall(r"<([^>]+)>", raw_tags)
            if not tag_names:
                tag_names = [t.strip() for t in raw_tags.split(",") if t.strip()]
            for tag in tag_names:
                entities.append(ExtractedEntity(
                    name                  = "tag",
                    entity_type           = "tag",
                    value                 = tag,
                    extraction_confidence = 1.0,
                    source_type           = SourceType.SQL,
                ))

    # Render content as flat key:value string
    parts = []
    for k, v in row.items():
        if v is None:
            continue
        if k in _TEXT_COLUMNS and isinstance(v, str) and len(v) > 200:
            v = v[:200] + "…"
        parts.append(f"{k}: {v}")
    content = " | ".join(parts)

    return NormalisedItem(
        source_service_id = service_id,
        source_type       = SourceType.SQL,
        item_id           = _stable_item_id(service_id, index, row),
        content           = content,
        entities          = entities,
        score             = 1.0,   # SQL rows have no ranking — all equal weight
        raw_record        = dict(row),
        retrieval_weight  = retrieval_weight,
    )


# ---------------------------------------------------------------------------
# Graph normalisation path (deterministic entity extraction)
# ---------------------------------------------------------------------------

def _neo4j_value_to_dict(v: Any) -> Any:
    """Flatten neo4j Node/Relationship objects to plain dicts if the driver returns them."""
    if hasattr(v, "_properties"):
        d = dict(v._properties)
        if hasattr(v, "labels"):
            d["_labels"] = list(v.labels)
        if hasattr(v, "type"):
            d["_type"] = v.type
        return d
    if isinstance(v, dict):
        return {k: _neo4j_value_to_dict(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_neo4j_value_to_dict(i) for i in v]
    return v


def _normalise_graph_record(
    record:     dict[str, Any],
    service_id: str,
    index:      int,
    retrieval_weight: float,
) -> NormalisedItem:
    # Flatten any neo4j driver objects
    flat: dict[str, Any] = {k: _neo4j_value_to_dict(v) for k, v in record.items()}

    entities: list[ExtractedEntity] = []
    parts: list[str] = []

    def _extract_from_dict(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            label = f"{prefix}.{k}" if prefix else k
            if k in ("id", "user_id", "post_id") and v is not None:
                entities.append(ExtractedEntity(
                    name                  = k,
                    entity_type           = k,
                    value                 = v,
                    extraction_confidence = 1.0,
                    source_type           = SourceType.GRAPH,
                ))
            if k == "name" and isinstance(v, str):
                entities.append(ExtractedEntity(
                    name                  = "tag",
                    entity_type           = "tag",
                    value                 = v,
                    extraction_confidence = 1.0,
                    source_type           = SourceType.GRAPH,
                ))
            if isinstance(v, dict):
                _extract_from_dict(v, label)
                parts.append(f"{label}: {{{', '.join(f'{kk}={vv}' for kk, vv in v.items() if kk not in ('_labels','_type'))}}}")
            elif v is not None:
                parts.append(f"{label}: {v}")

    _extract_from_dict(flat)
    content = " | ".join(parts) if parts else str(flat)

    return NormalisedItem(
        source_service_id = service_id,
        source_type       = SourceType.GRAPH,
        item_id           = _stable_item_id(service_id, index, flat),
        content           = content,
        entities          = entities,
        score             = 1.0,
        raw_record        = flat,
        retrieval_weight  = retrieval_weight,
    )


# ---------------------------------------------------------------------------
# Document normalisation path (probabilistic entity extraction via spaCy/regex)
# ---------------------------------------------------------------------------

def _extract_entities_from_text(text: str, source_type: SourceType) -> list[ExtractedEntity]:
    """
    Attempt spaCy NER first; fall back to simple regex for PERSON/ORG tokens.
    Confidence is the spaCy score when available, else 0.5 (regex heuristic).
    This is the probabilistic path — confidence < 1.0 by definition.
    """
    entities: list[ExtractedEntity] = []

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:1000])  # limit to 1000 chars to keep NER fast
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "PRODUCT", "GPE"):
                entities.append(ExtractedEntity(
                    name                  = ent.label_,
                    entity_type           = "freetext_mention",
                    value                 = ent.text,
                    extraction_confidence = ent._.score if hasattr(ent._, "score") else 0.7,
                    source_type           = source_type,
                ))
    except Exception:
        # spaCy not installed or model not found — use regex fallback
        for token in _UPPER_TOKEN.findall(text[:500]):
            entities.append(ExtractedEntity(
                name                  = "freetext_mention",
                entity_type           = "freetext_mention",
                value                 = token,
                extraction_confidence = 0.5,   # regex heuristic
                source_type           = source_type,
            ))

    return entities


def _normalise_bm25_hit(
    hit:        dict[str, Any],
    service_id: str,
    index:      int,
    raw_score:  float,
    max_score:  float,
    retrieval_weight: float,
) -> NormalisedItem:
    title  = hit.get("title", "")
    body   = hit.get("body", "")
    excerpt = body[:_BODY_EXCERPT_CHARS] + ("…" if len(body) > _BODY_EXCERPT_CHARS else "")
    content = f"{title}  {excerpt}".strip() if title else excerpt

    entities = _extract_entities_from_text(content, SourceType.DOCUMENT)

    # Post ID from Elasticsearch metadata
    post_id = hit.get("post_id") or hit.get("_id")
    if post_id:
        entities.insert(0, ExtractedEntity(
            name                  = "post_id",
            entity_type           = "post_id",
            value                 = post_id,
            extraction_confidence = 1.0,   # _id from ES is always reliable
            source_type           = SourceType.DOCUMENT,
        ))
    user_id = hit.get("user_id")
    if user_id:
        entities.insert(0, ExtractedEntity(
            name                  = "user_id",
            entity_type           = "user_id",
            value                 = user_id,
            extraction_confidence = 1.0,
            source_type           = SourceType.DOCUMENT,
        ))

    return NormalisedItem(
        source_service_id = service_id,
        source_type       = SourceType.DOCUMENT,
        item_id           = _stable_item_id(service_id, index, hit),
        content           = content,
        entities          = entities,
        score             = _normalise_score(raw_score, max_score),
        raw_record        = dict(hit),
        retrieval_weight  = retrieval_weight,
    )


# ---------------------------------------------------------------------------
# ResultNormaliser
# ---------------------------------------------------------------------------

class ResultNormaliser:
    """
    Step 7: normalises a RetrievalBatch into a NormalisedBatch.

    Failed services (RawServiceResult.succeeded=False) are silently dropped —
    their absence from the NormalisedBatch is the signal to the metrics layer
    that source coverage was incomplete.
    """

    def normalise(self, batch: RetrievalBatch) -> NormalisedBatch:
        normalised: list[NormalisedServiceResult] = []

        for raw in batch.results:
            if not raw.succeeded:
                log.warning(
                    "ResultNormaliser: skipping failed service '%s' (%s)",
                    raw.service_id, raw.error.error_message if raw.error else "unknown error",
                )
                continue

            items = self._normalise_one(raw)
            normalised.append(NormalisedServiceResult(
                service_id  = raw.service_id,
                source_type = raw.source_type,
                items       = items,
            ))
            log.debug(
                "ResultNormaliser: '%s' → %d normalised items",
                raw.service_id, len(items),
            )

        return NormalisedBatch(
            query_id       = batch.query_id,
            results        = normalised,
            source_wall_ms = batch.total_wall_ms,
        )

    def _normalise_one(self, raw: RawServiceResult) -> list[NormalisedItem]:
        if raw.source_type == SourceType.SQL:
            return [
                _normalise_sql_row(row, raw.service_id, i, raw.retrieval_weight)
                for i, row in enumerate(raw.rows)
            ]
        elif raw.source_type == SourceType.GRAPH:
            return [
                _normalise_graph_record(rec, raw.service_id, i, raw.retrieval_weight)
                for i, rec in enumerate(raw.rows)
            ]
        elif raw.source_type == SourceType.DOCUMENT:
            max_score = max((h.get("_score", 0.0) for h in raw.rows), default=1.0)
            return [
                _normalise_bm25_hit(hit, raw.service_id, i,
                                     hit.get("_score", 0.0), max_score,
                                     raw.retrieval_weight)
                for i, hit in enumerate(raw.rows)
            ]
        else:
            log.error("ResultNormaliser: unknown source_type '%s'", raw.source_type)
            return []
