"""
heterorag/layer2/query_planner.py
===================================
QueryPlanner — orchestrates Layer 2 end-to-end, producing I₂.

Also houses build_relevance_confirm_fn() — the concrete LLM-backed Stage 2
confirmation function for the ServiceRegistry's RelevanceFilter (Step 2 of
the implementation plan).  This lives in Layer 2 because it makes an LLM call
and depends on TranslationLLM; Layer 1 must stay LLM-free except for the
injected callable.

Orchestration flow:
  I₁ (List[ServiceDescriptor])
    → [for each service in parallel] build_translation_prompt → TranslationLLM
    → [per service]                  QueryValidator (validate + optional retry)
    → [drop failed services]
    → [per valid service]            ServiceQuery
    → I₂ (I2_QueryTranslationOutput)

Uniform weights (POC):
  The QueryPlanner for Paper 1 uses the retrieval_weight already stamped on each
  ServiceDescriptor by the ServiceRegistry (uniform 1/|M| after cold-start).
  Paper 2 will replace this with learned genetic planner weights — the interface
  is identical; only the weight values change.

Parallelism:
  Translation calls are dispatched concurrently via ThreadPoolExecutor.
  The order of queries in I₂ matches the order of descriptors in I₁.
  Validation runs sequentially after translation (validation is fast; LLM retry
  is synchronous so the retry prompt reuses the same thread).

Reference: Foundation Doc v1.6 §2, §6 (parallel retrieval, QueryValidator policy).
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from heterorag.layer1.models import (
    I1_ServiceDiscoveryOutput,
    PersistenceType,
    ServiceDescriptor,
)

from .models import (
    I2_QueryTranslationOutput,
    QueryLanguage,
    ServiceQuery,
    ValidationRecord,
    ValidationStatus,
)
from .prompts import build_translation_prompt
from .query_validator import QueryValidator
from .translation_llm import TranslationLLM

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping from PersistenceType to QueryLanguage
# ---------------------------------------------------------------------------

_PERSISTENCE_TO_QUERY_LANGUAGE: dict[PersistenceType, QueryLanguage] = {
    PersistenceType.SQL:      QueryLanguage.SQL,
    PersistenceType.GRAPH:    QueryLanguage.CYPHER,
    PersistenceType.DOCUMENT: QueryLanguage.BM25,
}


# ---------------------------------------------------------------------------
# Step 2: Concrete LLM RelevanceFilter confirm function
# ---------------------------------------------------------------------------

_RELEVANCE_CONFIRM_SYSTEM = """You are a service router for a federated database query system.
Given a natural language question and a list of database services, you must decide which
services contain data relevant to answering the question.

Respond with a JSON array of service_id strings — only the IDs of services that are
relevant. Return an empty array [] if none are relevant.
Output ONLY the JSON array — no explanation, no markdown, no other text."""


def build_relevance_confirm_fn(llm: TranslationLLM):
    """
    Build and return the Stage 2 LLM confirmation function for the RelevanceFilter.

    This is injected into build_poc_registry(llm_confirm_fn=...) to give the
    ServiceRegistry its LLM-backed shortlisting capability.

    Returns:
        Callable[[str, list[ServiceDescriptor]], list[str]]
        Accepts (query_text, candidate_descriptors) → list of relevant service_ids.
    """
    import json

    def confirm_fn(query_text: str, descriptors: list[ServiceDescriptor]) -> list[str]:
        if not descriptors:
            return []

        # Build a compact service menu for the LLM
        service_entries = []
        for d in descriptors:
            summary = d.capability_summary or f"{d.persistence_type.value} service"
            service_entries.append(
                f'  {{"service_id": "{d.service_id}", '
                f'"persistence_type": "{d.persistence_type.value}", '
                f'"description": "{summary}"}}'
            )
        services_block = "[\n" + ",\n".join(service_entries) + "\n]"

        prompt = (
            f"{_RELEVANCE_CONFIRM_SYSTEM}\n\n"
            f"QUESTION: {query_text}\n\n"
            f"AVAILABLE SERVICES:\n{services_block}\n\n"
            f"Relevant service_ids (JSON array):"
        )

        reply = llm.translate(prompt)
        raw   = reply.text.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            result = json.loads(raw)
            if isinstance(result, list):
                # Filter to only valid service_ids from the candidate set
                valid_ids = {d.service_id for d in descriptors}
                return [sid for sid in result if sid in valid_ids]
        except (json.JSONDecodeError, TypeError):
            log.warning(
                "RelevanceFilter LLM confirm_fn: failed to parse JSON reply %r — "
                "falling back to all candidates",
                raw[:200],
            )
        # Safe fallback: return all candidates so no service is wrongly excluded
        return [d.service_id for d in descriptors]

    return confirm_fn


# ---------------------------------------------------------------------------
# _TranslationTask — internal per-service work unit
# ---------------------------------------------------------------------------

@dataclass
class _TranslationTask:
    descriptor:   ServiceDescriptor
    natural_query: str


@dataclass
class _TranslationResult:
    descriptor:    ServiceDescriptor
    raw_query:     str
    prompt_tokens: int
    reply_tokens:  float


# ---------------------------------------------------------------------------
# QueryPlanner
# ---------------------------------------------------------------------------

class QueryPlanner:
    """
    Layer 2 orchestrator.

    Translates a natural language query into one native query per service
    selected by the ServiceRegistry, validates each, and returns I₂.

    Args:
        llm:          TranslationLLM instance (shared with validator for retries).
        max_workers:  Thread pool size for parallel translation. Default = number
                      of services in I₁ (bounded by the OS thread limit).

    Usage:
        planner = QueryPlanner(llm=TranslationLLM())
        i2 = planner.plan(i1, natural_query="how many users have reputation > 1000?")
    """

    def __init__(
        self,
        llm:         TranslationLLM,
        max_workers: int = 4,
    ):
        self._llm        = llm
        self._validator  = QueryValidator(llm)
        self._max_workers = max_workers

    def plan(
        self,
        i1:            I1_ServiceDiscoveryOutput,
        natural_query: str,
    ) -> I2_QueryTranslationOutput:
        """
        Produce I₂ from I₁.

        Steps:
          1. Translate the natural_query into a native query for each service in I₁,
             concurrently via ThreadPoolExecutor.
          2. Validate each translation (QueryValidator policy: syntactic check +
             one LLM retry + drop on double failure).
          3. Assemble valid queries into I₂; record dropped service_ids.

        Args:
            i1:            Output of Layer 1 (ServiceRegistry.discover()).
            natural_query: The user's original NL question.

        Returns:
            I2_QueryTranslationOutput satisfying all I₂ producer guarantees.
        """
        if not i1.descriptors:
            log.warning("QueryPlanner.plan: I₁ contains no services — returning empty I₂")
            return I2_QueryTranslationOutput(
                query_id      = i1.query_id,
                natural_query = natural_query,
                queries       = [],
            )

        # --- Step 1: parallel translation ---
        translation_results = self._translate_all(i1.descriptors, natural_query)

        # --- Steps 2 + 3: validate and assemble I₂ ---
        queries:           list[ServiceQuery]      = []
        validation_log:    list[ValidationRecord]  = []
        dropped_ids:       list[str]               = []

        for tr in translation_results:
            d   = tr.descriptor
            ql  = _PERSISTENCE_TO_QUERY_LANGUAGE[d.persistence_type]

            status, final_query, records = self._validator.validate(
                service_id     = d.service_id,
                query_language = ql.value,
                native_query   = tr.raw_query,
                natural_query  = natural_query,
            )
            validation_log.extend(records)

            if status == ValidationStatus.DROPPED:
                dropped_ids.append(d.service_id)
                log.warning(
                    "QueryPlanner: service '%s' dropped from I₂ after validation failure",
                    d.service_id,
                )
                continue

            queries.append(ServiceQuery(
                query_id          = i1.query_id,
                service_id        = d.service_id,
                display_name      = d.display_name,
                query_language    = ql,
                native_query      = final_query,
                retrieval_weight  = d.retrieval_weight,
                validation_status = status,
                translation_prompt_tokens = tr.prompt_tokens,
                translation_reply_tokens  = tr.reply_tokens,
            ))

        # Maintain I₁ ordering in I₂
        id_order = {d.service_id: i for i, d in enumerate(i1.descriptors)}
        queries.sort(key=lambda q: id_order.get(q.service_id, 999))

        i2 = I2_QueryTranslationOutput(
            query_id           = i1.query_id,
            natural_query      = natural_query,
            queries            = queries,
            validation_log     = validation_log,
            dropped_service_ids = dropped_ids,
        )

        log.info(
            "QueryPlanner: query_id=%s  %d services → %d in I₂ (%d dropped)",
            i1.query_id, len(i1.descriptors), len(queries), len(dropped_ids),
        )
        return i2

    # ------------------------------------------------------------------
    # Private: parallel translation
    # ------------------------------------------------------------------

    def _translate_one(
        self,
        descriptor:    ServiceDescriptor,
        natural_query: str,
    ) -> _TranslationResult:
        prompt = build_translation_prompt(descriptor, natural_query)
        reply  = self._llm.translate(prompt)
        return _TranslationResult(
            descriptor    = descriptor,
            raw_query     = reply.text,
            prompt_tokens = reply.prompt_tokens,
            reply_tokens  = reply.reply_tokens,
        )

    def _translate_all(
        self,
        descriptors:   list[ServiceDescriptor],
        natural_query: str,
    ) -> list[_TranslationResult]:
        """
        Translate all services concurrently.
        Returns results in the same order as descriptors (sorted by service_id index).
        """
        n_workers = min(self._max_workers, len(descriptors))
        results: dict[str, _TranslationResult] = {}

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_id = {
                pool.submit(self._translate_one, d, natural_query): d.service_id
                for d in descriptors
            }
            for future in as_completed(future_to_id):
                sid = future_to_id[future]
                try:
                    results[sid] = future.result()
                except Exception as exc:
                    log.error(
                        "QueryPlanner: translation failed for '%s': %s",
                        sid, exc,
                    )
                    # Insert a placeholder that will fail validation and be dropped
                    desc = next(d for d in descriptors if d.service_id == sid)
                    results[sid] = _TranslationResult(
                        descriptor    = desc,
                        raw_query     = f"-- TRANSLATION_ERROR: {exc}",
                        prompt_tokens = 0,
                        reply_tokens  = 0,
                    )

        # Return in original descriptor order
        return [results[d.service_id] for d in descriptors if d.service_id in results]
