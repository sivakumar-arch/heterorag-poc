"""
heterorag/layer4/pipeline.py
==============================
HeteroRAGPipeline — wires all four layers into a single callable.

This is the primary object used by both the benchmark runner (Step 12) and
by the B4 baseline (fixed-plan ablation), which shares the same pipeline
with different registry configuration.

The pipeline does NOT own the ServiceRegistry — it receives I₁ externally.
This separation means the benchmark runner controls service selection
(important for B1/B2 which hard-code single services) while sharing the
translation/retrieval/integration/generation machinery.

Public interface:
    pipeline.run(query_id, natural_query, i1) → GenerationResult
"""

from __future__ import annotations

import logging
import time

from heterorag.layer1.models import I1_ServiceDiscoveryOutput
from heterorag.layer2.query_planner import QueryPlanner
from heterorag.layer2.translation_llm import TranslationLLM
from heterorag.layer3 import (
    ConnectionRegistry,
    ParallelRetrievalExecutor,
    SemanticIntegrator,
)
from heterorag.layer4.generation import GenerationLLM, GenerationResult

log = logging.getLogger(__name__)


class HeteroRAGPipeline:
    """
    Full four-layer pipeline.

    Layers 1→4:
      L1: ServiceRegistry.discover()     → I₁   (called externally, passed in)
      L2: QueryPlanner.plan()            → I₂
      L3: ParallelRetrievalExecutor      → RetrievalBatch
          SemanticIntegrator             → I₃
      L4: GenerationLLM.generate()       → GenerationResult

    Args:
        translation_llm:   Shared LLM for query translation (Layer 2).
        generation_llm:    LLM for answer generation (Layer 4).
        conn_registry:     ConnectionConfig store for live DB connections.
        max_workers:       Thread pool size for parallel retrieval.
        timeout_ms:        Per-service retrieval timeout.
    """

    def __init__(
        self,
        translation_llm: TranslationLLM,
        generation_llm:  GenerationLLM,
        conn_registry:   ConnectionRegistry,
        max_workers:     int = 4,
        timeout_ms:      int = 10_000,
    ):
        self._planner    = QueryPlanner(translation_llm, max_workers=max_workers)
        self._executor   = ParallelRetrievalExecutor(
            conn_registry,
            per_service_timeout_ms=timeout_ms,
        )
        self._integrator = SemanticIntegrator()
        self._gen        = generation_llm

    def run(
        self,
        i1:            I1_ServiceDiscoveryOutput,
        natural_query: str,
    ) -> GenerationResult:
        """
        Run the full pipeline from I₁ to GenerationResult.

        Args:
            i1:            Output of ServiceRegistry.discover() for this query.
            natural_query: The user's original NL question.

        Returns:
            GenerationResult carrying the answer, retrieval_ms (RL metric),
            queried_service_ids (SC numerator), and conflict_count (ICR).
        """
        # Layer 2: translate
        i2 = self._planner.plan(i1, natural_query)

        # Layer 3: retrieve + integrate
        batch = self._executor.execute(i2)
        i3    = self._integrator.integrate(batch, natural_query)

        # Layer 4: generate
        result = self._gen.generate(i3)

        log.info(
            "HeteroRAGPipeline: query_id=%s  services=%s  rl=%.0fms  gen=%.0fms",
            i1.query_id,
            result.queried_service_ids,
            result.retrieval_ms,
            result.generation_ms,
        )
        return result
