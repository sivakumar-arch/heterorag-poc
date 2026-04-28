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

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

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

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        response = self._client.messages.create(
            model       = self.model,
            max_tokens  = max_tokens,
            temperature = temperature,
            messages    = [{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        return LLMResponse(
            text          = text.strip(),
            prompt_tokens = response.usage.input_tokens,
            reply_tokens  = response.usage.output_tokens,
            model         = self.model,
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
