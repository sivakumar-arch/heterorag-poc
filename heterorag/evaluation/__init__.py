# heterorag/evaluation/__init__.py
"""
HeteroRAG Evaluation — Baselines, Benchmark Runner, Metrics

Public API:
    B1_SQLOnlyRouter              — SQL-only routing baseline
    B2_DocumentOnlyRAG            — Document-only RAG baseline
    B3_LLMFunctionCallingRouter   — LLM function-calling router (sequential)
    B4_FixedPlanAblation          — Fixed-plan ablation (all services, uniform weight)
    BenchmarkRunner               — 120-question × 5-system benchmark orchestrator
    MetricsComputer               — SC / AF / RL / QTSR / ICR computation
    load_benchmark_questions      — returns the full 120-question taxonomy
"""
from .baselines import (
    B1_SQLOnlyRouter,
    B2_DocumentOnlyRAG,
    B3_LLMFunctionCallingRouter,
    B4_FixedPlanAblation,
    BaselineSystem,
)
from .benchmark_runner import BenchmarkRunner, BenchmarkQuestion, load_benchmark_questions
from .metrics import MetricsComputer

__all__ = [
    "B1_SQLOnlyRouter",
    "B2_DocumentOnlyRAG",
    "B3_LLMFunctionCallingRouter",
    "B4_FixedPlanAblation",
    "BaselineSystem",
    "BenchmarkQuestion",
    "BenchmarkRunner",
    "MetricsComputer",
    "load_benchmark_questions",
]
