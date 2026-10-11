"""
heterorag/federation/llm_config.py
==================================
Model configuration and cost metering for the planner and the translator
(Interface Specification v2.1, decision D23: cost-first model policy).

Each role has its own model. Defaults are the most cost-effective model; a larger model is
selected only by an explicit setting, so that a result can be attributed to the architecture
or to the model. The legacy variable ``HETERORAG_LLM_MODEL`` (used by the earlier
retrieve-and-generate baseline) is deliberately not read here.

    HETERORAG_PLANNER_MODEL      model of the planner
    HETERORAG_TRANSLATOR_MODEL   model of the translator
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Mapping

from ..llm_provider import LLMProvider, LLMResponse

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class RoleModels:
    planner: str = DEFAULT_MODEL
    translator: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RoleModels":
        e = env if env is not None else os.environ
        return cls(planner=e.get("HETERORAG_PLANNER_MODEL") or DEFAULT_MODEL,
                   translator=e.get("HETERORAG_TRANSLATOR_MODEL") or DEFAULT_MODEL)


@dataclass(frozen=True)
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    reply_tokens: int = 0
    thinking_tokens: int = 0
    seconds: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.calls + other.calls, self.prompt_tokens + other.prompt_tokens,
                     self.reply_tokens + other.reply_tokens,
                     self.thinking_tokens + other.thinking_tokens, self.seconds + other.seconds)

    def to_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "reply_tokens": self.reply_tokens, "thinking_tokens": self.thinking_tokens,
                "seconds": round(self.seconds, 3)}


class MeteredProvider(LLMProvider):
    """Wraps a provider and records every call. ``call()`` returns the response together with
    the usage of that one call, so concurrent callers can attribute cost per step."""

    def __init__(self, inner: LLMProvider, role: str = ""):
        self.inner, self.role = inner, role
        self._lock = threading.Lock()
        self._total = Usage()
        self._seen: set[str] = set()

    @property
    def models_seen(self) -> list[str]:
        """Model names the API reported for the calls made so far."""
        with self._lock:
            return sorted(self._seen)

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "unknown")

    @property
    def total(self) -> Usage:
        with self._lock:
            return self._total

    def call(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0):
        t0 = time.perf_counter()
        resp: LLMResponse = self.inner.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        u = Usage(1, resp.prompt_tokens, resp.reply_tokens, resp.thinking_tokens,
                  time.perf_counter() - t0)
        with self._lock:
            self._total = self._total + u
            if resp.model:
                self._seen.add(resp.model)
        return resp, u

    def complete(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        return self.call(prompt, max_tokens=max_tokens, temperature=temperature)[0]


def make_provider(model: str, role: str) -> MeteredProvider:
    """A metered Anthropic provider for one role."""
    from ..llm_provider import AnthropicProvider, RetryingProvider
    return MeteredProvider(RetryingProvider(AnthropicProvider(model=model)), role=role)
