"""
heterorag/evaluation/systems.py
================================
System factory for the benchmark.

Design rule: every system that talks to more than one service goes through the
SAME pipeline code (planner -> executor -> SemanticIntegrator -> generator).
Systems differ only along two explicit axes:

    selection : "all"  every registered service is translated and queried
                "llm"  an LLM first picks the relevant services (the Stage-2
                       RelevanceFilter prompt, build_relevance_confirm_fn)
    schedule  : "parallel"    translate + execute services concurrently
                "sequential"  translate + execute services one at a time

    HeteroRAG_Full    selection=all  schedule=parallel   (unchanged name; this
                      is also what the old B4_Fixed_Plan ran)
    Select_Parallel   selection=llm  schedule=parallel
    Select_Sequential selection=llm  schedule=sequential

Select_Sequential is the corrected replacement for the old B3. The old B3
differed from HeteroRAG in several undisclosed ways (single routing+translation
JSON prompt with no per-service schema, no query validation, no
SemanticIntegrator, 20-row cap, fallback to the raw NL question as the native
query), so it could not isolate the effect of sequential execution. It is still
available as `B3_LLM_FunctionCalling_Legacy` for reproducing earlier numbers.

Select_Parallel vs Select_Sequential isolates the schedule.
HeteroRAG_Full vs Select_Parallel isolates LLM service selection.
"""

from __future__ import annotations

import logging
from typing import Callable

from heterorag.evaluation.baselines import (
    B1_SQLOnlyRouter,
    B2_DocumentOnlyRAG,
    B3_LLMFunctionCallingRouter,
    B4_FixedPlanAblation,
    BaselineSystem,
)
from heterorag.layer1.poc_descriptors import build_poc_registry
from heterorag.layer2.query_planner import build_relevance_confirm_fn
from heterorag.layer2.translation_llm import TranslationLLM
from heterorag.layer3 import ConnectionRegistry
from heterorag.layer4.generation import GenerationLLM, GenerationResult
from heterorag.layer4.pipeline import HeteroRAGPipeline

log = logging.getLogger(__name__)

DEFAULT_SYSTEMS = [
    "HeteroRAG_Full",
    "B1_SQL_Only",
    "B2_Document_Only",
    "Select_Parallel",
    "Select_Sequential",
]
OPTIONAL_SYSTEMS = [
    "B4_Fixed_Plan",                       # identical to HeteroRAG_Full; kept for reproduction
    "B3_LLM_FunctionCalling_Legacy",       # the old B3 (see module docstring)
]


class PipelineSystem(BaselineSystem):
    """discover (Layer 1) -> HeteroRAGPipeline (Layers 2-4)."""

    def __init__(self, name: str, pipeline: HeteroRAGPipeline, registry):
        self._name = name
        self._pipeline = pipeline
        self._registry = registry

    @property
    def system_name(self) -> str:        # type: ignore[override]
        return self._name

    def run(self, query_id: str, natural_query: str) -> GenerationResult:
        i1 = self._registry.discover(query_id, natural_query)
        return self._pipeline.run(i1, natural_query)


def _warm_registry(registry) -> None:
    """Apply cold-start weights without issuing a Stage-2 LLM call."""
    try:
        with registry._lock:
            registry._apply_cold_start_if_needed()
    except Exception:                                         # pragma: no cover
        registry.discover("warmup", "warmup")


def build_systems(
    trans_llm:  TranslationLLM,
    gen_llm:    GenerationLLM,
    conn_reg:   ConnectionRegistry,
    names:      list[str] | None = None,
    abstain_short_circuit: bool = False,
) -> dict[str, BaselineSystem]:
    names = names or DEFAULT_SYSTEMS
    confirm_fn = build_relevance_confirm_fn(trans_llm)

    def pipeline_system(name: str, selection: str, schedule: str) -> PipelineSystem:
        registry = build_poc_registry(
            llm_confirm_fn=confirm_fn if selection == "llm" else None,
            heartbeat_timeout_seconds=86400,                  # never expires mid-run
        )
        _warm_registry(registry)
        pipeline = HeteroRAGPipeline(
            trans_llm, gen_llm, conn_reg,
            execution_mode=schedule,
            abstain_short_circuit=abstain_short_circuit,
        )
        return PipelineSystem(name, pipeline, registry)

    factories: dict[str, Callable[[], BaselineSystem]] = {
        "HeteroRAG_Full":    lambda: pipeline_system("HeteroRAG_Full",    "all", "parallel"),
        "Select_Parallel":   lambda: pipeline_system("Select_Parallel",   "llm", "parallel"),
        "Select_Sequential": lambda: pipeline_system("Select_Sequential", "llm", "sequential"),
        "B1_SQL_Only":       lambda: B1_SQLOnlyRouter(trans_llm, gen_llm, conn_reg),
        "B2_Document_Only":  lambda: B2_DocumentOnlyRAG(trans_llm, gen_llm, conn_reg),
        "B4_Fixed_Plan":     lambda: B4_FixedPlanAblation(trans_llm, gen_llm, conn_reg),
        "B3_LLM_FunctionCalling_Legacy":
                             lambda: _Legacy(B3_LLMFunctionCallingRouter(trans_llm, gen_llm, conn_reg)),
    }
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise ValueError(f"Unknown system(s) {unknown}; known: {sorted(factories)}")
    return {n: factories[n]() for n in names}


class _Legacy(BaselineSystem):
    """Re-labels the old B3 so its records cannot be confused with the new systems."""

    def __init__(self, inner: BaselineSystem):
        self._inner = inner

    @property
    def system_name(self) -> str:        # type: ignore[override]
        return "B3_LLM_FunctionCalling_Legacy"

    def run(self, query_id: str, natural_query: str) -> GenerationResult:
        return self._inner.run(query_id, natural_query)
