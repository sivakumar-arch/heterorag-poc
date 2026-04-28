# heterorag/layer2/__init__.py
"""
Layer 2 — Query Translation

Public API:
    QueryPlanner                  — orchestrates translation + validation → I₂
    TranslationLLM                — Anthropic API wrapper for NL→native translation
    QueryValidator                — syntactic validation + LLM retry policy
    build_relevance_confirm_fn    — LLM-backed Stage 2 RelevanceFilter confirm function
    I2_QueryTranslationOutput     — formal interface I₂
    ServiceQuery                  — atomic unit of I₂
    QueryLanguage                 — SQL | CYPHER | BM25
    ValidationStatus              — VALID | FIXED | DROPPED | SKIPPED
"""
from .models import (
    I2_QueryTranslationOutput,
    QueryLanguage,
    ServiceQuery,
    ValidationRecord,
    ValidationStatus,
)
from .translation_llm import TranslationLLM, TranslationReply
from .query_validator import QueryValidator
from .query_planner import QueryPlanner, build_relevance_confirm_fn
from .prompts import build_translation_prompt

__all__ = [
    "I2_QueryTranslationOutput",
    "QueryLanguage",
    "QueryPlanner",
    "QueryValidator",
    "ServiceQuery",
    "TranslationLLM",
    "TranslationReply",
    "ValidationRecord",
    "ValidationStatus",
    "build_relevance_confirm_fn",
    "build_translation_prompt",
]
