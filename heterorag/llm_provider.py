"""
heterorag/llm_provider.py
==========================
LLM Provider abstraction for HeteroRAG.

HeteroRAG uses an LLM in two places:
  - Layer 2 (TranslationLLM): NL → SQL / Cypher / BM25 translation
  - Layer 4 (GenerationLLM): grounded answer generation

Both places call the LLM with a single user message and expect a plain text
reply. This simple interface makes it trivial to swap providers.

The POC uses Anthropic Claude by default. To use a different provider,
implement the LLMProvider protocol and pass your instance to TranslationLLM
and GenerationLLM.

BUILT-IN PROVIDERS
──────────────────
  AnthropicProvider   — Anthropic Claude (default)
  OpenAIProvider      — OpenAI GPT models (requires: pip install openai)
  OllamaProvider      — Ollama local models (requires: pip install ollama)
  AzureOpenAIProvider — Azure OpenAI Service (requires: pip install openai)

USAGE
──────────────────
Default (Anthropic):
    from heterorag.layer2.translation_llm import TranslationLLM
    llm = TranslationLLM()   # uses AnthropicProvider automatically

OpenAI:
    from heterorag.llm_provider import OpenAIProvider
    from heterorag.layer2.translation_llm import TranslationLLM
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
    llm = TranslationLLM(provider=provider)

Ollama (local):
    from heterorag.llm_provider import OllamaProvider
    from heterorag.layer2.translation_llm import TranslationLLM
    provider = OllamaProvider(model="llama3")
    llm = TranslationLLM(provider=provider)

Custom provider:
    from heterorag.llm_provider import LLMProvider, LLMResponse
    class MyProvider(LLMProvider):
        def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> LLMResponse:
            # call your LLM here
            return LLMResponse(text="...", prompt_tokens=0, reply_tokens=0)
    llm = TranslationLLM(provider=MyProvider())
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass — common to all providers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Normalised response from any LLM provider."""
    text:          str
    prompt_tokens: int   = 0
    reply_tokens:  int   = 0
    model:         str   = ""
    # Part of reply_tokens spent on internal reasoning (billed as output). 0 when the
    # model does not think or the SDK does not report it.
    thinking_tokens: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Minimal interface that any LLM provider must implement.

    HeteroRAG only needs single-turn completion — one user message in,
    one text reply out. No streaming, no tool use, no multi-turn history.
    """

    @abstractmethod
    def complete(
        self,
        prompt:      str,
        *,
        max_tokens:  int   = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        Send a single user prompt and return the LLM reply.

        Args:
            prompt:      The full prompt string (built by prompts.py or generation.py).
            max_tokens:  Maximum tokens in the reply.
            temperature: Sampling temperature. Use 0.0 for deterministic output.

        Returns:
            LLMResponse with text, token counts, and model name.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Provider 1 — Anthropic Claude (default)
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider using the official anthropic-python SDK.

    Installation: pip install anthropic

    Models: claude-opus-4-5, claude-sonnet-4-20250514, claude-haiku-4-5-20251001
    Docs:   https://docs.anthropic.com/en/api/messages

    The POC was evaluated with claude-sonnet-4-20250514.
    """

    # claude-sonnet-4-20250514, the model of the original evaluation, was retired by
    # Anthropic on 2026-06-15. The default is the current Sonnet; pin the model
    # explicitly (HETERORAG_LLM_MODEL) for anything that is reported.
    DEFAULT_MODEL = "claude-sonnet-5-5"

    def __init__(
        self,
        model:   str        = DEFAULT_MODEL,
        api_key: str | None = None,
    ):
        self.model = model
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        log.info("AnthropicProvider: model=%s", model)

    # Newer Claude models do not accept `temperature` (the API rejects any value other
    # than the default with a 400, and recent SDKs drop the argument entirely). Rather
    # than keep a list of model names that goes stale, the provider sends temperature
    # first and, if the model or SDK refuses it, retries once without it and remembers
    # that for the rest of the run. Decoding is then not under our control, so results
    # for such models should be reported over repeated runs.
    temperature_supported: bool = True

    @staticmethod
    def _rejects_temperature(exc: Exception) -> bool:
        if "temperature" not in str(exc).lower():
            return False
        if isinstance(exc, TypeError):               # SDK signature no longer has it
            return True
        return getattr(exc, "status_code", None) == 400   # API refused the value

    # Ceiling for the automatic retry when the reply budget is spent on reasoning.
    MAX_TOKENS_CEILING = int(os.environ.get("HETERORAG_MAX_TOKENS_CEILING", "16000"))

    def _create(self, kwargs: dict, temperature: float):
        """One messages.create call, sending temperature only while it is accepted."""
        if self.temperature_supported:
            try:
                return self._client.messages.create(temperature=temperature, **kwargs)
            except Exception as exc:
                if not self._rejects_temperature(exc):
                    raise
                self.temperature_supported = False
                log.warning(
                    "AnthropicProvider: the SDK or model %s rejected 'temperature' (%s); "
                    "continuing without it. Sampling is now the model default, so report "
                    "results over repeated runs.", self.model, str(exc)[:120])
        return self._client.messages.create(**kwargs)

    @staticmethod
    def _text_of(response) -> str:
        # Current models can put a thinking block before the answer, so content[0] is
        # not necessarily text. Join the text blocks only.
        return "".join(
            b.text for b in (response.content or [])
            if getattr(b, "type", "text") == "text" and hasattr(b, "text")
        )

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        kwargs = dict(
            model      = self.model,
            max_tokens = max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )
        response = self._create(kwargs, temperature)
        text = self._text_of(response)

        # Reasoning tokens count toward max_tokens. If the whole budget went on thinking
        # there is no answer text; retry once with a larger budget instead of failing the
        # run. Tokens of the discarded attempt are still billed, so they are added to the
        # usage that is reported.
        spent_in = spent_out = spent_think = 0
        if (not text.strip() and getattr(response, "stop_reason", None) == "max_tokens"
                and max_tokens < self.MAX_TOKENS_CEILING):
            d0 = getattr(response.usage, "output_tokens_details", None)
            spent_in    = response.usage.input_tokens
            spent_out   = response.usage.output_tokens
            spent_think = int(getattr(d0, "thinking_tokens", 0) or 0)
            bigger = min(max_tokens * 4, self.MAX_TOKENS_CEILING)
            log.warning("AnthropicProvider: reply budget %d spent before any answer text; "
                        "retrying once with max_tokens=%d", max_tokens, bigger)
            kwargs["max_tokens"] = bigger
            response = self._create(kwargs, temperature)
            text = self._text_of(response)

        blocks = list(response.content or [])
        if blocks and not text.strip():
            kinds = [getattr(b, "type", type(b).__name__) for b in blocks]
            raise RuntimeError(
                "Anthropic response has no text block (blocks=%s, stop_reason=%s). With "
                "thinking enabled, reasoning tokens count toward max_tokens; if "
                "stop_reason is 'max_tokens', raise the token limit."
                % (kinds, getattr(response, "stop_reason", None)))
        details = getattr(response.usage, "output_tokens_details", None)
        thinking = int(getattr(details, "thinking_tokens", 0) or 0)
        return LLMResponse(
            text            = text.strip(),
            prompt_tokens   = response.usage.input_tokens + spent_in,
            reply_tokens    = response.usage.output_tokens + spent_out,
            model           = self.model,
            thinking_tokens = thinking + spent_think,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Provider 2 — OpenAI GPT
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """
    OpenAI GPT provider using the official openai-python SDK.

    Installation: pip install openai

    Models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
    Docs:   https://platform.openai.com/docs/api-reference/chat

    Example:
        provider = OpenAIProvider(model="gpt-4o")
        llm = TranslationLLM(provider=provider)
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model:   str        = DEFAULT_MODEL,
        api_key: str | None = None,
    ):
        self.model = model
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY")
            )
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        log.info("OpenAIProvider: model=%s", model)

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        response = self._client.chat.completions.create(
            model       = self.model,
            max_tokens  = max_tokens,
            temperature = temperature,
            messages    = [{"role": "user", "content": prompt}],
        )
        choice = response.choices[0]
        text   = choice.message.content or ""
        usage  = response.usage
        return LLMResponse(
            text          = text.strip(),
            prompt_tokens = usage.prompt_tokens if usage else 0,
            reply_tokens  = usage.completion_tokens if usage else 0,
            model         = self.model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Provider 3 — Azure OpenAI Service
# ─────────────────────────────────────────────────────────────────────────────

class AzureOpenAIProvider(LLMProvider):
    """
    Azure OpenAI Service provider.

    Installation: pip install openai

    Requires:
        AZURE_OPENAI_API_KEY     — your Azure API key
        AZURE_OPENAI_ENDPOINT    — e.g. https://your-resource.openai.azure.com/
        AZURE_OPENAI_DEPLOYMENT  — your deployment name

    Example:
        provider = AzureOpenAIProvider(
            deployment="gpt-4o-deployment",
            endpoint="https://your-resource.openai.azure.com/",
            api_version="2024-02-01",
        )
        llm = TranslationLLM(provider=provider)
    """

    def __init__(
        self,
        deployment:  str        = "",
        endpoint:    str | None = None,
        api_key:     str | None = None,
        api_version: str        = "2024-02-01",
    ):
        self.model = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        try:
            import openai
            self._client = openai.AzureOpenAI(
                api_key     = api_key     or os.environ.get("AZURE_OPENAI_API_KEY"),
                azure_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                api_version = api_version,
            )
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        log.info("AzureOpenAIProvider: deployment=%s", self.model)

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        response = self._client.chat.completions.create(
            model       = self.model,
            max_tokens  = max_tokens,
            temperature = temperature,
            messages    = [{"role": "user", "content": prompt}],
        )
        choice = response.choices[0]
        text   = choice.message.content or ""
        usage  = response.usage
        return LLMResponse(
            text          = text.strip(),
            prompt_tokens = usage.prompt_tokens if usage else 0,
            reply_tokens  = usage.completion_tokens if usage else 0,
            model         = self.model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Provider 4 — Ollama (local models)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """
    Ollama provider for locally hosted open-source models.

    Installation: pip install ollama
    Ollama:       https://ollama.ai  (install Ollama, then: ollama pull llama3)

    Models: llama3, mistral, gemma2, codellama, qwen2 — any model pulled via Ollama.
    Runs entirely locally — no API key or internet connection required.

    Example:
        provider = OllamaProvider(model="llama3")
        llm = TranslationLLM(provider=provider)
    """

    DEFAULT_MODEL = "llama3"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host:  str = "http://localhost:11434",
    ):
        self.model = model
        try:
            import ollama
            self._client = ollama.Client(host=host)
        except ImportError:
            raise ImportError(
                "ollama package not installed. Run: pip install ollama"
            )
        log.info("OllamaProvider: model=%s host=%s", model, host)

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        response = self._client.chat(
            model   = self.model,
            messages= [{"role": "user", "content": prompt}],
            options = {"temperature": temperature, "num_predict": max_tokens},
        )
        text  = response["message"]["content"] if isinstance(response, dict) else response.message.content
        usage = response.get("prompt_eval_count", 0) if isinstance(response, dict) else 0
        reply = response.get("eval_count", 0) if isinstance(response, dict) else 0
        return LLMResponse(
            text          = (text or "").strip(),
            prompt_tokens = usage,
            reply_tokens  = reply,
            model         = self.model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Usage meter + retry wrapper
# ─────────────────────────────────────────────────────────────────────────────
#
# Why this exists
# ---------------
# Earlier benchmark runs had no retry/backoff: an HTTP 429 from the provider
# raised inside the translation thread pool, was converted into a placeholder
# query, failed validation, and was recorded as a *dropped service* — i.e. a
# rate-limit error was indistinguishable from a genuine translation failure.
# Every LLM call now goes through RetryingProvider, which
#   * retries transient errors (429, 5xx, 529 overloaded, connection/timeouts)
#     with exponential backoff + jitter, honouring Retry-After when present;
#   * records every call, retry and token in the process-wide LLM_USAGE meter,
#     so the benchmark runner can log per-run LLM call counts (the cost side of
#     the evaluation) without touching any call site.

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_RETRYABLE_NAMES = {
    "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
    "OverloadedError", "ServiceUnavailableError", "Timeout", "ReadTimeout",
    "ConnectTimeout", "ConnectionError", "APIStatusError_529",
}


class LLMUsageMeter:
    """Thread-safe process-wide counters. Use snapshot()/delta() around a run."""

    _FIELDS = ("calls", "retries", "failures", "prompt_tokens", "reply_tokens",
               "thinking_tokens", "llm_ms", "retry_wait_ms")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._v = {k: 0.0 for k in self._FIELDS}

    def add(self, **kw: float) -> None:
        with self._lock:
            for k, v in kw.items():
                self._v[k] += v

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._v)

    @staticmethod
    def delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
        return {k: after[k] - before[k] for k in after}


LLM_USAGE = LLMUsageMeter()


def is_retryable_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    if status in _RETRYABLE_STATUS:
        return True
    if type(exc).__name__ in _RETRYABLE_NAMES:
        return True
    msg = str(exc).lower()
    return "overloaded" in msg or "rate limit" in msg or "rate_limit" in msg


def _retry_after_s(exc: BaseException) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        try:
            v = headers.get("retry-after") or headers.get("Retry-After")
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            return None
    return None


class RetryingProvider(LLMProvider):
    """
    Wraps any LLMProvider with retry/backoff and usage metering.

    max_retries=0 disables retrying (calls are still metered). Defaults can be
    overridden with HETERORAG_LLM_MAX_RETRIES / HETERORAG_LLM_BACKOFF_BASE_S /
    HETERORAG_LLM_BACKOFF_MAX_S.
    """

    def __init__(
        self,
        inner:       LLMProvider,
        max_retries: int | None   = None,
        base_delay:  float | None = None,
        max_delay:   float | None = None,
        sleep=time.sleep,
    ):
        self.inner       = inner
        self.max_retries = int(os.environ.get("HETERORAG_LLM_MAX_RETRIES", 8)) \
            if max_retries is None else max_retries
        self.base_delay  = float(os.environ.get("HETERORAG_LLM_BACKOFF_BASE_S", 2.0)) \
            if base_delay is None else base_delay
        self.max_delay   = float(os.environ.get("HETERORAG_LLM_BACKOFF_MAX_S", 60.0)) \
            if max_delay is None else max_delay
        self._sleep      = sleep

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "unknown")

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        attempt = 0
        while True:
            t0 = time.perf_counter()
            try:
                resp = self.inner.complete(prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:                      # noqa: BLE001
                elapsed = (time.perf_counter() - t0) * 1000.0
                if attempt >= self.max_retries or not is_retryable_error(exc):
                    LLM_USAGE.add(calls=1, failures=1, llm_ms=elapsed)
                    raise
                wait = _retry_after_s(exc)
                if wait is None:
                    wait = min(self.max_delay, self.base_delay * (2 ** attempt))
                wait = min(self.max_delay, wait) * (0.75 + 0.5 * random.random())
                log.warning("LLM call failed (%s: %s); retry %d/%d in %.1fs",
                            type(exc).__name__, str(exc)[:120], attempt + 1,
                            self.max_retries, wait)
                LLM_USAGE.add(retries=1, retry_wait_ms=wait * 1000.0, llm_ms=elapsed)
                self._sleep(wait)
                attempt += 1
                continue
            LLM_USAGE.add(
                calls=1,
                llm_ms=(time.perf_counter() - t0) * 1000.0,
                prompt_tokens=resp.prompt_tokens,
                reply_tokens=resp.reply_tokens,
                thinking_tokens=getattr(resp, "thinking_tokens", 0),
            )
            return resp


def with_retries(provider: LLMProvider) -> LLMProvider:
    """Idempotently wrap a provider in RetryingProvider."""
    return provider if isinstance(provider, RetryingProvider) else RetryingProvider(provider)


# ─────────────────────────────────────────────────────────────────────────────
# Mock provider — for testing without any API
# ─────────────────────────────────────────────────────────────────────────────

class MockProvider(LLMProvider):
    """
    Mock provider that returns a fixed reply for deterministic testing.

    Usage:
        provider = MockProvider(reply="SELECT id FROM users LIMIT 10")
        llm = TranslationLLM(provider=provider)
    """

    def __init__(self, reply: str = "[MOCK]"):
        self.model = "mock"
        self._reply = reply

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse(
            text          = self._reply,
            prompt_tokens = len(prompt.split()),
            reply_tokens  = len(self._reply.split()),
            model         = "mock",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def provider_from_env() -> LLMProvider:
    """
    Build an LLM provider from environment variables.

    Reads HETERORAG_LLM_PROVIDER (default: anthropic) and returns the
    appropriate provider. Useful for switching providers without code changes.

    Environment variables:
        HETERORAG_LLM_PROVIDER   anthropic | openai | azure | ollama
        HETERORAG_LLM_MODEL      model name override (optional)

        For Anthropic: ANTHROPIC_API_KEY
        For OpenAI:    OPENAI_API_KEY
        For Azure:     AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
        For Ollama:    OLLAMA_HOST (default: http://localhost:11434)

    Example:
        export HETERORAG_LLM_PROVIDER=openai
        export HETERORAG_LLM_MODEL=gpt-4o
        export OPENAI_API_KEY=sk-...
    """
    return with_retries(_provider_from_env_unwrapped())


def resolved_model_name() -> str:
    """The model name the environment selects, without building a provider (so it works
    without an API key). Recorded in run_meta so a result states the model it used."""
    provider_name = os.environ.get("HETERORAG_LLM_PROVIDER", "anthropic").lower()
    override = os.environ.get("HETERORAG_LLM_MODEL", "")
    defaults = {
        "anthropic": AnthropicProvider.DEFAULT_MODEL,
        "openai":    OpenAIProvider.DEFAULT_MODEL,
        "azure":     os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        "ollama":    OllamaProvider.DEFAULT_MODEL,
    }
    return override or defaults.get(provider_name, "unknown")


def _provider_from_env_unwrapped() -> LLMProvider:
    provider_name = os.environ.get("HETERORAG_LLM_PROVIDER", "anthropic").lower()
    model_override = os.environ.get("HETERORAG_LLM_MODEL", "")

    if provider_name == "anthropic":
        model = model_override or AnthropicProvider.DEFAULT_MODEL
        return AnthropicProvider(model=model)

    elif provider_name == "openai":
        model = model_override or OpenAIProvider.DEFAULT_MODEL
        return OpenAIProvider(model=model)

    elif provider_name == "azure":
        return AzureOpenAIProvider(
            deployment = model_override or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        )

    elif provider_name == "ollama":
        model = model_override or OllamaProvider.DEFAULT_MODEL
        host  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaProvider(model=model, host=host)

    else:
        raise ValueError(
            f"Unknown HETERORAG_LLM_PROVIDER={provider_name!r}. "
            f"Valid options: anthropic, openai, azure, ollama"
        )
