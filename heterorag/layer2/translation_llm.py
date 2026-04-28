"""
heterorag/layer2/translation_llm.py
=====================================
TranslationLLM — LLM-agnostic wrapper for per-service NL→native query translation.

The POC was evaluated with Anthropic Claude (claude-sonnet-4-20250514).
Any LLMProvider can be substituted — see heterorag/llm_provider.py for
built-in providers (Anthropic, OpenAI, Azure OpenAI, Ollama) and the
LLMProvider protocol for custom implementations.

Usage — default (Anthropic):
    llm = TranslationLLM()

Usage — OpenAI:
    from heterorag.llm_provider import OpenAIProvider
    llm = TranslationLLM(provider=OpenAIProvider(model="gpt-4o"))

Usage — Ollama (local, no API key):
    from heterorag.llm_provider import OllamaProvider
    llm = TranslationLLM(provider=OllamaProvider(model="llama3"))

Usage — environment-driven (no code change):
    export HETERORAG_LLM_PROVIDER=openai
    export HETERORAG_LLM_MODEL=gpt-4o
    export OPENAI_API_KEY=sk-...
    llm = TranslationLLM()   # reads provider from env automatically

Temperature: 0.0 — translation requires determinism, not creativity.
Max tokens:  512 — queries are short; a generous ceiling avoids truncation.

Reference: Foundation Doc v1.6 §2 (query translation is per-query LLM operation).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from heterorag.llm_provider import (
    LLMProvider,
    AnthropicProvider,
    MockProvider,
    provider_from_env,
)

log = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS  = 512
_DEFAULT_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Reply dataclass
# ---------------------------------------------------------------------------

@dataclass
class TranslationReply:
    """The structured reply from one translation LLM call."""
    text:          str
    prompt_tokens: int   = 0
    reply_tokens:  int   = 0
    duration_ms:   float = 0.0
    model:         str   = ""


# ---------------------------------------------------------------------------
# TranslationLLM
# ---------------------------------------------------------------------------

class TranslationLLM:
    """
    LLM wrapper used exclusively for query translation in Layer 2.

    Delegates all LLM calls to an LLMProvider instance — swap the provider
    to change the underlying model without touching any other code.

    All translation calls go through a single instance so token usage can be
    aggregated across the full benchmark run.

    Args:
        provider:     LLMProvider instance. If None, reads HETERORAG_LLM_PROVIDER
                      from the environment (default: AnthropicProvider).
        max_tokens:   Maximum reply tokens. Default: 512.
        temperature:  Sampling temperature. Default: 0.0 (deterministic).
        inject_reply: If set, returns this string for every call without
                      hitting any LLM. Used for testing.
        api_key:      Convenience shortcut — if provided and provider is None,
                      passes api_key to the default AnthropicProvider.
    """

    def __init__(
        self,
        provider:     LLMProvider | None = None,
        max_tokens:   int                = _DEFAULT_MAX_TOKENS,
        temperature:  float              = _DEFAULT_TEMPERATURE,
        inject_reply: str | None         = None,
        # Legacy convenience arg — keeps backward compatibility
        api_key:      str | None         = None,
        model:        str | None         = None,
    ):
        self._max_tokens   = max_tokens
        self._temperature  = temperature
        self._inject_reply = inject_reply

        # Token usage counters — accumulated across all calls
        self._total_prompt_tokens = 0
        self._total_reply_tokens  = 0
        self._total_calls         = 0

        if inject_reply is not None:
            self._provider = MockProvider(reply=inject_reply)
            log.info("TranslationLLM: mock mode")
        elif provider is not None:
            self._provider = provider
        else:
            # Auto-select: environment variable → AnthropicProvider default
            import os
            if os.environ.get("HETERORAG_LLM_PROVIDER"):
                self._provider = provider_from_env()
            else:
                self._provider = AnthropicProvider(
                    model   = model or AnthropicProvider.DEFAULT_MODEL,
                    api_key = api_key,
                )

        log.info("TranslationLLM: provider=%s", type(self._provider).__name__)

    # -------------------------------------------------------------------------

    @property
    def _model(self) -> str:
        """Expose model name for compatibility with existing logging."""
        return getattr(self._provider, "model", "unknown")

    def translate(self, prompt: str) -> TranslationReply:
        """
        Send prompt to the LLM and return the cleaned reply.

        The reply text is stripped of leading/trailing whitespace and any
        accidental markdown code fences (``` or ```sql etc.) that models
        occasionally emit despite the prompt instructions.
        """
        t0       = time.perf_counter()
        response = self._provider.complete(
            prompt,
            max_tokens  = self._max_tokens,
            temperature = self._temperature,
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0

        clean = self._strip_fences(response.text)

        self._total_prompt_tokens += response.prompt_tokens
        self._total_reply_tokens  += response.reply_tokens
        self._total_calls         += 1

        log.debug(
            "TranslationLLM: %d prompt_tokens, %d reply_tokens, %.1fms",
            response.prompt_tokens, response.reply_tokens, duration_ms,
        )

        return TranslationReply(
            text          = clean,
            prompt_tokens = response.prompt_tokens,
            reply_tokens  = response.reply_tokens,
            duration_ms   = duration_ms,
            model         = response.model,
        )

    # -------------------------------------------------------------------------
    # Usage accounting
    # -------------------------------------------------------------------------

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_reply_tokens(self) -> int:
        return self._total_reply_tokens

    @property
    def total_calls(self) -> int:
        return self._total_calls

    def usage_summary(self) -> dict:
        return {
            "total_calls":         self._total_calls,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_reply_tokens":  self._total_reply_tokens,
            "total_tokens":        self._total_prompt_tokens + self._total_reply_tokens,
        }

    # -------------------------------------------------------------------------

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences that models occasionally emit."""
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
