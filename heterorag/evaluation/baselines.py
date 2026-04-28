"""
heterorag/evaluation/baselines.py
===================================
Step 11 — Baselines B1, B2, B3, B4.

All four baselines implement the same interface:
    baseline.run(query_id, natural_query) → GenerationResult

They share the same LLM instances and ConnectionRegistry as HeteroRAG Full,
ensuring fair comparison (same model, same data, different routing logic).

B1 — SQL-Only Router:
    Hard-routes every query to the User & Activity Service (PostgreSQL).
    Represents the enterprise status quo: a single SQL data warehouse.
    Uses the same TranslationLLM for NL→SQL as HeteroRAG.

B2 — Document-Only RAG (Standard RAG):
    Hard-routes every query to the Content Service (Elasticsearch/BM25).
    Represents standard RAG as practised since 2020.
    Uses the same BM25 index and GenerationLLM as HeteroRAG.

B3 — LLM Function-Calling Router (Sequential):
    Exposes all three services as "functions" to the LLM.
    LLM decides which services to call and executes them ONE AT A TIME
    (sequential, not parallel). This is the critical structural difference
    from HeteroRAG. Implemented at FULL STRENGTH — same LLM, same
    ServiceDescriptor information — so the comparison is purely empirical,
    not argumentative. This baseline directly answers the reviewer challenge
    "why not just use LLM function calling?".

B4 — HeteroRAG Fixed-Plan Ablation:
    Full HeteroRAG architecture (parallel, semantic integration, generation),
    but the retrieval plan is FIXED to always query all three services with
    uniform weights 1/3 each, regardless of query type. Isolates the
    contribution of the learned genetic planner.
    SC = 1.0 by construction — noted as a ceiling artefact in the paper.

Reference: Foundation Doc v1.6 §7.5 (Baselines).
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from heterorag.layer1.models import (
    ConnectionConfig,
    I1_ServiceDiscoveryOutput,
    PersistenceType,
    ServiceDescriptor,
)
from heterorag.layer1.poc_descriptors import (
    CONTENT_SERVICE_DESCRIPTOR,
    KNOWLEDGE_GRAPH_DESCRIPTOR,
    USER_ACTIVITY_DESCRIPTOR,
    build_poc_registry,
)
from heterorag.layer2.models import (
    I2_QueryTranslationOutput,
    QueryLanguage,
    ServiceQuery,
    ValidationStatus,
)
from heterorag.layer2.prompts import build_translation_prompt
from heterorag.layer2.translation_llm import TranslationLLM
from heterorag.layer3 import (
    ConnectionRegistry,
    ParallelRetrievalExecutor,
    SemanticIntegrator,
)
from heterorag.layer3.models import I3_IntegratedContext
from heterorag.layer4.generation import (
    GenerationLLM,
    GenerationResult,
    _CANNOT_ANSWER,
    _SYSTEM_PROMPT,
    _build_generation_prompt,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline ABC
# ---------------------------------------------------------------------------

class BaselineSystem(ABC):
    """Abstract base for all five systems (HeteroRAG + 4 baselines)."""

    @property
    @abstractmethod
    def system_name(self) -> str:
        ...

    @abstractmethod
    def run(self, query_id: str, natural_query: str) -> GenerationResult:
        ...


# =============================================================================
# B1 — SQL-Only Router
# =============================================================================

class B1_SQLOnlyRouter(BaselineSystem):
    """
    Routes every query exclusively to the SQL (PostgreSQL) service.
    Uses the same TranslationLLM and GenerationLLM as HeteroRAG.
    """

    system_name = "B1_SQL_Only"

    def __init__(
        self,
        translation_llm: TranslationLLM,
        generation_llm:  GenerationLLM,
        conn_registry:   ConnectionRegistry,
    ):
        self._llm      = translation_llm
        self._gen      = generation_llm
        self._executor = ParallelRetrievalExecutor(conn_registry)
        self._integrator = SemanticIntegrator()

    def run(self, query_id: str, natural_query: str) -> GenerationResult:
        # Build I₁ containing only the SQL service
        i1 = I1_ServiceDiscoveryOutput(
            query_id    = query_id,
            descriptors = [USER_ACTIVITY_DESCRIPTOR.model_copy(
                update={"retrieval_weight": 1.0, "is_active": True}
            )],
        )
        # Translate
        prompt   = build_translation_prompt(USER_ACTIVITY_DESCRIPTOR, natural_query)
        reply    = self._llm.translate(prompt)
        sq = ServiceQuery(
            query_id          = query_id,
            service_id        = USER_ACTIVITY_DESCRIPTOR.service_id,
            display_name      = USER_ACTIVITY_DESCRIPTOR.display_name,
            query_language    = QueryLanguage.SQL,
            native_query      = reply.text,
            retrieval_weight  = 1.0,
            validation_status = ValidationStatus.SKIPPED,
        )
        i2 = I2_QueryTranslationOutput(
            query_id=query_id, natural_query=natural_query, queries=[sq]
        )
        # Retrieve + integrate + generate
        batch  = self._executor.execute(i2)
        i3     = self._integrator.integrate(batch, natural_query)
        result = self._gen.generate(i3)
        result.queried_service_ids = [USER_ACTIVITY_DESCRIPTOR.service_id]
        return result


# =============================================================================
# B2 — Document-Only RAG
# =============================================================================

class B2_DocumentOnlyRAG(BaselineSystem):
    """
    Routes every query exclusively to the Content Service (Elasticsearch/BM25).
    Represents standard RAG as practised since 2020.
    """

    system_name = "B2_Document_Only"

    def __init__(
        self,
        translation_llm: TranslationLLM,
        generation_llm:  GenerationLLM,
        conn_registry:   ConnectionRegistry,
    ):
        self._llm      = translation_llm
        self._gen      = generation_llm
        self._executor = ParallelRetrievalExecutor(conn_registry)
        self._integrator = SemanticIntegrator()

    def run(self, query_id: str, natural_query: str) -> GenerationResult:
        # Build I₁ containing only the Document service
        i1 = I1_ServiceDiscoveryOutput(
            query_id    = query_id,
            descriptors = [CONTENT_SERVICE_DESCRIPTOR.model_copy(
                update={"retrieval_weight": 1.0, "is_active": True}
            )],
        )
        # BM25: translate natural query to search terms
        prompt = build_translation_prompt(CONTENT_SERVICE_DESCRIPTOR, natural_query)
        reply  = self._llm.translate(prompt)
        sq = ServiceQuery(
            query_id          = query_id,
            service_id        = CONTENT_SERVICE_DESCRIPTOR.service_id,
            display_name      = CONTENT_SERVICE_DESCRIPTOR.display_name,
            query_language    = QueryLanguage.BM25,
            native_query      = reply.text,
            retrieval_weight  = 1.0,
            validation_status = ValidationStatus.SKIPPED,
        )
        i2 = I2_QueryTranslationOutput(
            query_id=query_id, natural_query=natural_query, queries=[sq]
        )
        batch  = self._executor.execute(i2)
        i3     = self._integrator.integrate(batch, natural_query)
        result = self._gen.generate(i3)
        result.queried_service_ids = [CONTENT_SERVICE_DESCRIPTOR.service_id]
        return result


# =============================================================================
# B3 — LLM Function-Calling Router (Sequential)
# =============================================================================

_B3_ROUTING_PROMPT = """\
You are a query router for a federated database system with three services.
Given the user's question, decide which services to query, then construct the query for each.

AVAILABLE SERVICES:
{service_descriptions}

QUESTION: {question}

Respond with a JSON object:
{{
  "selected_services": ["service_id", ...],
  "queries": {{
    "service_id": "native query string",
    ...
  }}
}}
Output ONLY the JSON — no explanation, no markdown fences."""


class B3_LLMFunctionCallingRouter(BaselineSystem):
    """
    LLM Function-Calling Router — the strongest possible LLM-only baseline.

    The LLM sees the same ServiceDescriptor information as HeteroRAG's
    RelevanceFilter and makes routing + translation decisions in a SINGLE call.
    Services are then queried SEQUENTIALLY (not in parallel).

    This is the empirical answer to: "Why not just use LLM function calling?"
    Expected result: highest accuracy among baselines, but higher latency
    than HeteroRAG due to sequential execution.
    """

    system_name = "B3_LLM_FunctionCalling"

    def __init__(
        self,
        translation_llm: TranslationLLM,
        generation_llm:  GenerationLLM,
        conn_registry:   ConnectionRegistry,
    ):
        self._llm        = translation_llm
        self._gen        = generation_llm
        self._conn_reg   = conn_registry
        self._integrator = SemanticIntegrator()

        self._descriptors = [
            USER_ACTIVITY_DESCRIPTOR,
            KNOWLEDGE_GRAPH_DESCRIPTOR,
            CONTENT_SERVICE_DESCRIPTOR,
        ]

    def run(self, query_id: str, natural_query: str) -> GenerationResult:
        # Step 1: single LLM call for routing + translation
        service_descriptions = "\n".join(
            f'  - {d.service_id} ({d.persistence_type.value}): '
            f'{d.capability_summary or d.display_name}'
            for d in self._descriptors
        )
        prompt = _B3_ROUTING_PROMPT.format(
            service_descriptions=service_descriptions,
            question=natural_query,
        )
        reply = self._llm.translate(prompt)

        selected_ids, queries_by_svc = self._parse_routing_reply(
            reply.text, natural_query
        )

        if not selected_ids:
            log.warning("B3: routing returned no services for query_id=%s", query_id)
            return GenerationResult(
                query_id            = query_id,
                natural_query       = natural_query,
                answer              = _CANNOT_ANSWER,
                is_cannot_answer    = True,
                queried_service_ids = [],
                model               = self._llm._model,
            )

        # Step 2: SEQUENTIAL execution (the key difference from HeteroRAG)
        all_rows: list[dict] = []
        queried: list[str]   = []
        total_rl_ms: float   = 0.0

        _ql_map = {
            "user-activity-service":   QueryLanguage.SQL,
            "knowledge-graph-service": QueryLanguage.CYPHER,
            "content-service":         QueryLanguage.BM25,
        }
        _desc_map = {d.service_id: d for d in self._descriptors}

        for svc_id in selected_ids:
            native_query = queries_by_svc.get(svc_id, natural_query)
            ql           = _ql_map.get(svc_id, QueryLanguage.BM25)
            sq = ServiceQuery(
                query_id          = query_id,
                service_id        = svc_id,
                display_name      = _desc_map[svc_id].display_name if svc_id in _desc_map else svc_id,
                query_language    = ql,
                native_query      = native_query,
                retrieval_weight  = 1.0 / len(selected_ids),
                validation_status = ValidationStatus.SKIPPED,
            )
            # Execute ONE service at a time (sequential)
            from heterorag.layer3.retrieval_executor import ParallelRetrievalExecutor as PRE
            single_executor = PRE(self._conn_reg)
            i2_single = I2_QueryTranslationOutput(
                query_id=query_id, natural_query=natural_query, queries=[sq]
            )
            t0 = time.perf_counter()
            batch_single = single_executor.execute(i2_single)
            total_rl_ms += (time.perf_counter() - t0) * 1000.0

            if batch_single.successful():
                queried.append(svc_id)
                all_rows.extend(batch_single.results[0].rows[:20])

        # Build a simple I₃ from the concatenated sequential results
        i3 = self._build_sequential_i3(
            query_id, natural_query, all_rows, queried, total_rl_ms
        )
        result = self._gen.generate(i3)
        result.queried_service_ids = queried
        result.retrieval_ms        = total_rl_ms   # sequential sum, not parallel wall
        return result

    def _parse_routing_reply(
        self,
        text: str,
        natural_query: str,
    ) -> tuple[list[str], dict[str, str]]:
        """Parse the LLM routing JSON. Falls back to all services on parse error."""
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            obj = json.loads(raw)
            selected = obj.get("selected_services", [])
            queries  = obj.get("queries", {})
            # Validate service IDs
            valid_ids = {d.service_id for d in self._descriptors}
            selected = [s for s in selected if s in valid_ids]
            return selected, queries
        except (json.JSONDecodeError, TypeError):
            log.warning("B3: failed to parse routing JSON — falling back to all services")
            return [d.service_id for d in self._descriptors], {}

    def _build_sequential_i3(
        self,
        query_id:       str,
        natural_query:  str,
        all_rows:       list[dict],
        queried:        list[str],
        total_rl_ms:    float,
    ) -> I3_IntegratedContext:
        """Build a minimal I₃ from concatenated sequential results for generation."""
        from heterorag.layer3.models import RankedItem, SourceType

        items = []
        for i, row in enumerate(all_rows[:20]):
            content = " | ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
            items.append(RankedItem(
                rank=i + 1, content=content,
                source_service_ids=queried,
                source_types=[SourceType.DOCUMENT],
                final_score=1.0 / (i + 1),
            ))

        context_text = "\n".join(
            f"[RANK {it.rank}] {it.content}" for it in items
        )
        return I3_IntegratedContext(
            query_id            = query_id,
            natural_query       = natural_query,
            items               = items,
            context_text        = context_text,
            queried_service_ids = queried,
            total_wall_ms       = total_rl_ms,
            conflict_count      = 0,
        )


# =============================================================================
# B4 — HeteroRAG Fixed-Plan Ablation
# =============================================================================

class B4_FixedPlanAblation(BaselineSystem):
    """
    Full HeteroRAG architecture with parallel retrieval and semantic integration,
    but a FIXED retrieval plan: always queries all three services with uniform
    weight 1/3 regardless of query type.

    This isolates the contribution of the learned genetic planner.
    SC = 1.0 by construction (all services always queried) — noted as a
    ceiling artefact in the paper.
    """

    system_name = "B4_Fixed_Plan"

    def __init__(
        self,
        translation_llm: TranslationLLM,
        generation_llm:  GenerationLLM,
        conn_registry:   ConnectionRegistry,
    ):
        from heterorag.layer4.pipeline import HeteroRAGPipeline
        # Build a registry that always returns all three services
        self._registry = build_poc_registry(heartbeat_timeout_seconds=86400)  # 24h
        # Force uniform weights  (they start at 0, cold-start sets 1/3)
        self._registry.discover("b4-warmup", "warmup")   # triggers cold start

        self._pipeline = HeteroRAGPipeline(
            translation_llm = translation_llm,
            generation_llm  = generation_llm,
            conn_registry   = conn_registry,
        )

    def run(self, query_id: str, natural_query: str) -> GenerationResult:
        # Always return all three services at uniform weight
        all_descriptors = self._registry.all_descriptors(active_only=True)
        i1 = I1_ServiceDiscoveryOutput(
            query_id    = query_id,
            descriptors = all_descriptors,
        )
        return self._pipeline.run(i1, natural_query)
