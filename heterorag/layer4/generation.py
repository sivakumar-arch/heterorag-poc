"""
heterorag/layer4/generation.py
================================
GenerationLLM — LLM-agnostic wrapper for Layer 4 answer generation.

The POC was evaluated with Anthropic Claude (claude-sonnet-4-20250514).
Any LLMProvider can be substituted — see heterorag/llm_provider.py for
built-in providers (Anthropic, OpenAI, Azure OpenAI, Ollama) and the
LLMProvider protocol for custom implementations.

Usage — default (Anthropic):
    gen = GenerationLLM()

Usage — OpenAI:
    from heterorag.llm_provider import OpenAIProvider
    gen = GenerationLLM(provider=OpenAIProvider(model="gpt-4o"))

Usage — Ollama (local, no API key):
    from heterorag.llm_provider import OllamaProvider
    gen = GenerationLLM(provider=OllamaProvider(model="llama3"))

Usage — environment-driven (no code change):
    export HETERORAG_LLM_PROVIDER=openai
    export HETERORAG_LLM_MODEL=gpt-4o
    export OPENAI_API_KEY=sk-...
    gen = GenerationLLM()   # reads provider from env automatically

Design decisions:
  - Generation latency is EXCLUDED from the RL metric. RL is measured at I₃
    delivery (end of Layer 3). Generation time is a function of the LLM
    provider, not HeteroRAG's retrieval architecture.
  - The prompt template instructs the LLM to answer ONLY from the provided
    context and to cite which source (SQL / GRAPH / DOCUMENT) each claim
    comes from. This template is provider-agnostic.
  - Temperature = 0.0 for reproducibility across benchmark runs.
  - If I₃ has no items (empty retrieval), CANNOT_ANSWER is returned without
    an LLM call — correct behaviour distinguishable from a genuine answer.
  - Generation quality is intentionally out of scope for POC validation —
    the POC validates federated retrieval correctness (Foundation Doc §8).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from heterorag.layer3.models import I3_IntegratedContext
from heterorag.llm_provider import (
    LLMProvider,
    AnthropicProvider,
    MockProvider,
    provider_from_env,
)

log = logging.getLogger(__name__)

_MAX_TOKENS    = 1024
_TEMPERATURE   = 0.0
_CANNOT_ANSWER = "[CANNOT_ANSWER: no relevant context retrieved]"

# ---------------------------------------------------------------------------
# Layer 4 prompt template (provider-agnostic)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant for a Stack Exchange analytics system.
You are given retrieved context from one or more database services (SQL, GRAPH, DOCUMENT).
Answer the question using ONLY information present in the context.
For each factual claim, indicate which source type it comes from: [SQL], [GRAPH], or [DOCUMENT].
If the context does not contain enough information to answer, say exactly:
"I cannot answer this question from the provided context."
Do not speculate or use outside knowledge."""


def _build_generation_prompt(i3: I3_IntegratedContext) -> str:
    return (
        f"RETRIEVED CONTEXT:\n"
        f"{'=' * 60}\n"
        f"{i3.context_text}\n"
        f"{'=' * 60}\n\n"
        f"QUESTION: {i3.natural_query}\n\n"
        f"ANSWER (cite source types):"
    )


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """The complete output of one Layer 4 generation call."""
    query_id:            str
    natural_query:       str
    answer:              str
    is_cannot_answer:    bool          = False
    prompt_tokens:       int           = 0
    reply_tokens:        int           = 0
    generation_ms:       float         = 0.0   # excluded from RL metric
    retrieval_ms:        float         = 0.0   # copied from I₃.total_wall_ms (RL metric)
    queried_service_ids: list[str]     = field(default_factory=list)
    conflict_count:      int           = 0
    produced_at:         datetime      = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    model:               str           = ""


# ---------------------------------------------------------------------------
# GenerationLLM
# ---------------------------------------------------------------------------

class GenerationLLM:
    """
    LLM-agnostic wrapper for Layer 4 answer generation.

    Delegates all LLM calls to an LLMProvider instance — swap the provider
    to change the underlying model without touching any other code.

    Args:
        provider:      LLMProvider instance. If None, reads HETERORAG_LLM_PROVIDER
                       from the environment (default: AnthropicProvider).
        max_tokens:    Maximum reply tokens. Default: 1024.
        temperature:   Sampling temperature. Default: 0.0 (deterministic).
        inject_answer: If set, returns this string without hitting any LLM.
                       Used for smoke testing.
        api_key:       Convenience shortcut for default AnthropicProvider.
    """

    def __init__(
        self,
        provider:      LLMProvider | None = None,
        max_tokens:    int                = _MAX_TOKENS,
        temperature:   float              = _TEMPERATURE,
        inject_answer: str | None         = None,
        # Legacy convenience args
        api_key:       str | None         = None,
        model:         str | None         = None,
    ):
        self._max_tokens   = max_tokens
        self._temperature  = temperature
        self._inject_answer = inject_answer

        if inject_answer is not None:
            self._provider = MockProvider(reply=inject_answer)
            log.info("GenerationLLM: mock mode")
        elif provider is not None:
            self._provider = provider
        else:
            import os
            if os.environ.get("HETERORAG_LLM_PROVIDER"):
                self._provider = provider_from_env()
            else:
                self._provider = AnthropicProvider(
                    model   = model or AnthropicProvider.DEFAULT_MODEL,
                    api_key = api_key,
                )

        log.info("GenerationLLM: provider=%s", type(self._provider).__name__)

    def generate(self, i3: I3_IntegratedContext) -> GenerationResult:
        # Empty context → no LLM call
        if not i3.items:
            log.warning(
                "GenerationLLM: I₃ is empty — returning CANNOT_ANSWER for %s",
                i3.query_id,
            )
            return GenerationResult(
                query_id            = i3.query_id,
                natural_query       = i3.natural_query,
                answer              = _CANNOT_ANSWER,
                is_cannot_answer    = True,
                retrieval_ms        = i3.total_wall_ms,
                queried_service_ids = i3.queried_service_ids,
                conflict_count      = i3.conflict_count,
                model               = getattr(self._provider, "model", "unknown"),
            )

        # Build the full prompt (system instruction + context + question)
        full_prompt = _SYSTEM_PROMPT + "\n\n" + _build_generation_prompt(i3)

        t0 = time.perf_counter()
        response = self._provider.complete(
            full_prompt,
            max_tokens  = self._max_tokens,
            temperature = self._temperature,
        )
        generation_ms = (time.perf_counter() - t0) * 1000.0

        answer = response.text or _CANNOT_ANSWER

        log.info(
            "GenerationLLM: query_id=%s  gen=%.0fms  tokens=%d+%d",
            i3.query_id, generation_ms,
            response.prompt_tokens, response.reply_tokens,
        )

        return GenerationResult(
            query_id            = i3.query_id,
            natural_query       = i3.natural_query,
            answer              = answer,
            is_cannot_answer    = answer.startswith("I cannot answer"),
            prompt_tokens       = response.prompt_tokens,
            reply_tokens        = response.reply_tokens,
            generation_ms       = generation_ms,
            retrieval_ms        = i3.total_wall_ms,
            queried_service_ids = i3.queried_service_ids,
            conflict_count      = i3.conflict_count,
            model               = response.model,
        )
